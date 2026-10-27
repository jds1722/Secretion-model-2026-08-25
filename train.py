from __future__ import annotations

import argparse
import ast
import datetime as dt
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    balanced_accuracy_score,
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.config import MODEL_CHANNEL_NAMES, MODEL_INPUT_CHANNELS
from src.dataset import INDEX_TO_LABEL, VALID_TARGETS, SecretorTiffDataset
from src.model import MODEL_NAMES, create_binary_classifier, get_gradcam_layer
from src.preprocessed_dataset import PreprocessedSecretorDataset
from src.resilient_io import replace_file, temporary_path, write_json_atomic
from src.training_manifest import is_original_variant, source_id


class Tee:
    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--preprocessed",
        action="store_true",
        help="Read cached NPZ tensors from preprocessed_path instead of source TIFFs.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--roi-size",
        type=int,
        default=None,
        help="Square ROI side length in original TIFF pixels, centered on the detected nucleus.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--model", choices=MODEL_NAMES, default="resnet18")
    parser.add_argument(
        "--early-stopping-patience", type=int, default=0,
        help="Stop after this many epochs without validation-score improvement; 0 disables it.",
    )
    parser.add_argument(
        "--early-stopping-min-delta", type=float, default=0.001,
        help="Minimum validation-score increase counted as an improvement.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from <output-dir>/latest_checkpoint.pt.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Training log path. Defaults to <output-dir>/train.log.",
    )
    parser.add_argument(
        "--preview-images",
        type=int,
        default=50,
        help="Save this many preprocessing examples with Grad-CAM model-focus maps.",
    )
    parser.add_argument(
        "--threshold-mode",
        choices=["balanced_accuracy", "high_recall"],
        default="balanced_accuracy",
        help=(
            "Choose the high/low decision threshold from validation predictions. "
            "balanced_accuracy balances high and low recall; high_recall favors "
            "detecting high at the cost of more false positives."
        ),
    )
    parser.add_argument(
        "--min-high-recall",
        type=float,
        default=0.9,
        help="Minimum validation high recall when --threshold-mode high_recall is used.",
    )
    parser.add_argument(
        "--target",
        choices=sorted(VALID_TARGETS),
        default=None,
        help=(
            "Train one independent hormone classifier. Use insulin or glucagon "
            "when the manifest contains hormone-specific labels."
        ),
    )
    return parser.parse_args()


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def setup_logging(output_dir: Path, log_file: str | None) -> object:
    log_path = Path(log_file) if log_file is not None else output_dir / "train.log"
    if not log_path.is_absolute():
        log_path = Path.cwd() / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = open(log_path, "a", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_handle)
    sys.stderr = Tee(sys.__stderr__, log_handle)
    print(f"Logging to {log_path}")
    return log_handle


def save_checkpoint_atomic(payload: dict[str, object], path: Path) -> None:
    """Save a checkpoint without exposing a partial file to other processes."""
    temporary = temporary_path(path)
    try:
        torch.save(payload, temporary)
        replace_file(temporary, path, attempts=20)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> dict[str, float]:
    model.eval()
    labels = []
    probs = []
    preds = []
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            logits = model(x)
            if criterion is not None:
                loss = criterion(logits, y)
                total_loss += float(loss.item()) * x.size(0)
                total_count += x.size(0)
            p_high = torch.softmax(logits, dim=1)[:, 1]
            pred = torch.argmax(logits, dim=1)
            labels.extend(y.cpu().tolist())
            probs.extend(p_high.cpu().tolist())
            preds.extend(pred.cpu().tolist())

    metrics = {"accuracy": accuracy_score(labels, preds)}
    if criterion is not None and total_count > 0:
        metrics["loss"] = total_loss / total_count
    if len(set(labels)) == 2:
        metrics["auroc"] = roc_auc_score(labels, probs)
        metrics["average_precision"] = average_precision_score(labels, probs)
    else:
        metrics["auroc"] = float("nan")
        metrics["average_precision"] = float("nan")
    return metrics


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[float]]:
    model.eval()
    labels = []
    probs = []

    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            logits = model(x)
            p_high = torch.softmax(logits, dim=1)[:, 1]
            labels.extend(y.cpu().tolist())
            probs.extend(p_high.cpu().tolist())

    return labels, probs


def labels_from_threshold(probs: list[float], threshold: float) -> list[int]:
    return [1 if prob >= threshold else 0 for prob in probs]


def metrics_from_predictions(
    labels: list[int],
    probs: list[float],
    threshold: float,
) -> dict[str, float]:
    preds = labels_from_threshold(probs, threshold)
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "balanced_accuracy": balanced_accuracy_score(labels, preds),
        "threshold": threshold,
    }
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    low_total = int(cm[0].sum())
    high_total = int(cm[1].sum())
    metrics["low_recall"] = float(cm[0, 0] / low_total) if low_total else float("nan")
    metrics["high_recall"] = float(cm[1, 1] / high_total) if high_total else float("nan")
    metrics["high_precision"] = (
        float(cm[1, 1] / (cm[0, 1] + cm[1, 1]))
        if (cm[0, 1] + cm[1, 1])
        else float("nan")
    )
    if len(set(labels)) == 2:
        metrics["auroc"] = roc_auc_score(labels, probs)
        metrics["average_precision"] = average_precision_score(labels, probs)
    else:
        metrics["auroc"] = float("nan")
        metrics["average_precision"] = float("nan")
    return metrics


def threshold_table(labels: list[int], probs: list[float]) -> pd.DataFrame:
    rows = []
    for threshold in np.linspace(0.01, 0.99, 99):
        metrics = metrics_from_predictions(labels, probs, float(threshold))
        rows.append(metrics)
    return pd.DataFrame(rows)


def choose_decision_threshold(
    labels: list[int],
    probs: list[float],
    mode: str,
    min_high_recall: float,
) -> tuple[float, pd.DataFrame]:
    df = threshold_table(labels, probs)
    if mode == "high_recall":
        candidates = df[df["high_recall"] >= min_high_recall]
        if not candidates.empty:
            best_idx = candidates["balanced_accuracy"].idxmax()
            return float(candidates.loc[best_idx, "threshold"]), df

    best_idx = df["balanced_accuracy"].idxmax()
    return float(df.loc[best_idx, "threshold"]), df


def save_training_plots(history: list[dict[str, float]], output_dir: Path) -> None:
    df = pd.DataFrame(history)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(df["epoch"], df["train_loss"], marker="o", label="train")
    if "val_loss" in df:
        axes[0].plot(df["epoch"], df["val_loss"], marker="o", label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross entropy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(df["epoch"], df["val_accuracy"], marker="o", label="accuracy")
    if "val_auroc" in df:
        axes[1].plot(df["epoch"], df["val_auroc"], marker="o", label="AUROC")
    if "val_average_precision" in df:
        axes[1].plot(
            df["epoch"],
            df["val_average_precision"],
            marker="o",
            label="average precision",
        )
    axes[1].set_title("Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=200)
    plt.close(fig)


def save_test_plots(
    labels: list[int],
    preds: list[int],
    probs: list[float],
    output_dir: Path,
) -> None:
    display_labels = [INDEX_TO_LABEL[0], INDEX_TO_LABEL[1]]
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=display_labels)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title("Test Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "test_confusion_matrix.png", dpi=200)
    plt.close(fig)

    if len(set(labels)) != 2:
        return

    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(labels, probs, ax=ax)
    ax.set_title("Test ROC Curve")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "test_roc_curve.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    PrecisionRecallDisplay.from_predictions(labels, probs, ax=ax)
    ax.set_title("Test Precision-Recall Curve")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "test_precision_recall_curve.png", dpi=200)
    plt.close(fig)


def save_threshold_plot(thresholds: pd.DataFrame, threshold: float, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(thresholds["threshold"], thresholds["balanced_accuracy"], label="balanced accuracy")
    ax.plot(thresholds["threshold"], thresholds["high_recall"], label="high recall")
    ax.plot(thresholds["threshold"], thresholds["low_recall"], label="low recall")
    ax.axvline(threshold, color="black", linestyle="--", label=f"selected={threshold:.2f}")
    ax.set_title("Validation Threshold Selection")
    ax.set_xlabel("High probability threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "validation_threshold_curve.png", dpi=200)
    plt.close(fig)


def select_checkpoint_score(metrics: dict[str, float]) -> float:
    auroc = metrics.get("auroc", float("nan"))
    if math.isfinite(auroc):
        return auroc
    return metrics["accuracy"]


def compute_gradcam(
    model: nn.Module,
    x: torch.Tensor,
    class_idx: int,
    target_layer: nn.Module,
) -> np.ndarray:
    activations = None
    gradients = None

    def forward_hook(_module: nn.Module, _inputs: tuple[torch.Tensor], output: torch.Tensor) -> None:
        nonlocal activations
        activations = output

    def backward_hook(
        _module: nn.Module,
        _grad_input: tuple[torch.Tensor],
        grad_output: tuple[torch.Tensor],
    ) -> None:
        nonlocal gradients
        gradients = grad_output[0]

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(x)
        logits[:, class_idx].sum().backward()
    finally:
        forward_handle.remove()
        backward_handle.remove()

    if activations is None or gradients is None:
        raise RuntimeError("Grad-CAM hooks did not capture activations and gradients.")

    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activations).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(
        cam,
        size=x.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    cam_np = cam[0, 0].detach().cpu().numpy()
    cam_np -= float(cam_np.min())
    max_value = float(cam_np.max())
    if max_value > 0:
        cam_np /= max_value
    return cam_np


def gradcam_preview_indices(
    dataset: SecretorTiffDataset | PreprocessedSecretorDataset,
    max_images: int,
) -> list[int]:
    """Select one original image per source, balanced across available labels."""
    if max_images <= 0 or len(dataset) == 0:
        return []
    dataframe = getattr(dataset, "df", None)
    if not isinstance(dataframe, pd.DataFrame) or "path" not in dataframe.columns:
        return list(range(min(max_images, len(dataset))))

    candidates = dataframe.copy()
    candidates["_dataset_index"] = np.arange(len(candidates), dtype=int)
    candidates = candidates[
        candidates["path"].astype(str).map(is_original_variant)
    ].copy()
    if candidates.empty:
        return []

    if "source_id" in candidates.columns:
        source_groups = candidates["source_id"].fillna("").astype(str)
        missing_group = source_groups.str.strip() == ""
        source_groups.loc[missing_group] = candidates.loc[
            missing_group, "path"
        ].astype(str).map(source_id)
    else:
        source_groups = candidates["path"].astype(str).map(source_id)
    candidates["_source_group"] = source_groups

    group_columns = ["_source_group"]
    if "hormone" in candidates.columns:
        group_columns.insert(0, "hormone")
    label_column = None
    target = getattr(dataset, "target", None)
    if target and f"{target}_label" in candidates.columns:
        label_column = f"{target}_label"
    elif "label" in candidates.columns:
        label_column = "label"
    if label_column:
        group_columns.insert(0, label_column)

    originals = candidates.drop_duplicates(group_columns, keep="first")
    if not label_column:
        return originals["_dataset_index"].astype(int).head(max_images).tolist()

    label_order = list(
        dict.fromkeys(originals[label_column].astype(str).str.lower())
    )
    queues = {
        label: originals.loc[
            originals[label_column].astype(str).str.lower() == label,
            "_dataset_index",
        ].astype(int).tolist()
        for label in label_order
    }
    selected: list[int] = []
    while len(selected) < max_images and any(queues.values()):
        for label in label_order:
            if queues[label] and len(selected) < max_images:
                selected.append(queues[label].pop(0))
    return selected


def save_preprocess_attention_examples(
    model: nn.Module,
    dataset: SecretorTiffDataset | PreprocessedSecretorDataset,
    device: torch.device,
    output_dir: Path,
    max_images: int,
    threshold: float,
) -> None:
    if max_images <= 0:
        return

    examples_dir = output_dir / "preprocess_attention_examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    target_layer = get_gradcam_layer(model)
    model.eval()
    preview_indices = gradcam_preview_indices(dataset, max_images)
    if not preview_indices:
        print(
            "WARNING: no original, non-augmented images were available for "
            "Grad-CAM previews.",
            flush=True,
        )
        return
    print(
        f"Grad-CAM previews: selected {len(preview_indices)} unique original "
        f"images from {len(dataset)} test rows.",
        flush=True,
    )

    for preview_number, dataset_index in enumerate(preview_indices):
        item = dataset[dataset_index]
        x = item["image"].unsqueeze(0).to(device)
        label_idx = int(item["label"].item())
        path = Path(str(item["path"]))

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
            pred_idx = 1 if float(probs[1].item()) >= threshold else 0

        heatmap = compute_gradcam(model, x, pred_idx, target_layer)
        image_np = item["image"].detach().cpu().numpy()
        bf = image_np[0]
        nucleus = image_np[1]
        cell_mask = image_np[2]
        nucleus_bf = image_np[3]
        ignore_mask = item["ignore_mask"].detach().cpu().numpy()
        overlay_base = np.clip((bf + nucleus) / 2.0, 0.0, 1.0)

        fig, axes = plt.subplots(1, 7, figsize=(21, 3))
        axes[0].imshow(bf, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("BF preprocessed")
        axes[1].imshow(nucleus, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Nucleus preprocessed")
        axes[2].imshow(cell_mask, cmap="viridis", vmin=0, vmax=1)
        axes[2].set_title("Nucleus-guided cell mask")
        axes[3].imshow(nucleus_bf, cmap="gray", vmin=0, vmax=1)
        axes[3].set_title("Nucleus-localized BF")
        axes[4].imshow(ignore_mask, cmap="magma")
        axes[4].set_title("Ignore mask")
        axes[5].imshow(heatmap, cmap="inferno", vmin=0, vmax=1)
        axes[5].set_title("Model focus (Grad-CAM)")
        axes[6].imshow(overlay_base, cmap="gray", vmin=0, vmax=1)
        axes[6].imshow(heatmap, cmap="inferno", alpha=0.45, vmin=0, vmax=1)
        axes[6].set_title("Overlay")

        for ax in axes:
            ax.axis("off")

        label_name = INDEX_TO_LABEL[label_idx]
        pred_name = INDEX_TO_LABEL[pred_idx]
        fig.suptitle(
            f"{path.name} | true={label_name} pred={pred_name} "
            f"prob_high={float(probs[1].item()):.3f}",
            fontsize=9,
        )
        fig.tight_layout()

        out_path = examples_dir / f"{preview_number:03d}_{path.stem}_attention.png"
        fig.savefig(out_path, dpi=180)
        plt.close(fig)

        rows.append(
            {
                "path": str(path),
                "source_id": source_id(str(path)),
                "dataset_index": dataset_index,
                "augmentation": "original",
                "example_image": str(out_path),
                "label": label_name,
                "predicted_label": pred_name,
                "prob_low": float(probs[0].item()),
                "prob_high": float(probs[1].item()),
            }
        )

    pd.DataFrame(rows).to_csv(examples_dir / "examples.csv", index=False)


def main() -> None:
    args = parse_args()
    training_args = {
        **vars(args),
        "input_channels": MODEL_INPUT_CHANNELS,
        "channel_names": list(MODEL_CHANNEL_NAMES),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / "training.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(
            f"Training is already using {output_dir}, or a stale lock exists at {lock_path}. "
            "Use a different --output-dir."
        )
    with os.fdopen(lock_fd, "w", encoding="utf-8") as lock_file:
        lock_file.write(f"pid={os.getpid()}\n")
    log_handle = setup_logging(output_dir, args.log_file)
    progress_path = output_dir / "progress.json"
    progress_state: dict[str, object] = {
        "status": "initializing",
        "pid": os.getpid(),
        "target": args.target or "label",
        "model": args.model,
        "preprocessed": bool(args.preprocessed),
        "epoch": 0,
        "total_epochs": args.epochs,
        "batch": 0,
        "total_batches": 0,
        "epoch_progress": 0.0,
        "overall_progress": 0.0,
        "elapsed_seconds": 0.0,
        "eta_seconds": None,
        "val_auroc": None,
        "val_accuracy": None,
        "best_validation_score": None,
        "best_epoch": 0,
    }

    progress_write_failures = 0

    def publish_progress(**values: object) -> None:
        nonlocal progress_write_failures
        progress_state.update(values)
        try:
            write_json_atomic(progress_path, progress_state, attempts=12)
        except OSError as error:
            progress_write_failures += 1
            progress_state["progress_write_failures"] = progress_write_failures
            progress_state["last_progress_write_error"] = str(error)
            if progress_write_failures == 1 or progress_write_failures % 10 == 0:
                print(
                    "WARNING: progress.json is temporarily locked; training will "
                    f"continue and the dashboard update will retry: {error}",
                    flush=True,
                )

    publish_progress()

    try:
        with open(output_dir / "run_config.json", "w", encoding="utf-8") as config_file:
            json.dump(training_args, config_file, indent=2)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Started at {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Target: {args.target or 'label'}")
        print(f"Model: {args.model}")
        print(f"Device: {device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        publish_progress(
            status="loading_data",
            device=str(device),
            gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        )

        dataset_class = (
            PreprocessedSecretorDataset if args.preprocessed else SecretorTiffDataset
        )
        train_ds = dataset_class(
            args.manifest,
            split="train",
            image_size=args.image_size,
            target=args.target,
            roi_size=args.roi_size,
        )
        val_ds = dataset_class(
            args.manifest,
            split="val",
            image_size=args.image_size,
            target=args.target,
            roi_size=args.roi_size,
        )
        test_ds = dataset_class(
            args.manifest,
            split="test",
            image_size=args.image_size,
            target=args.target,
            roi_size=args.roi_size,
        )
        print(
            "Dataset sizes: "
            f"train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}"
        )
        publish_progress(
            train_samples=len(train_ds),
            validation_samples=len(val_ds),
            test_samples=len(test_ds),
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

        model = create_binary_classifier(
            args.model, in_channels=MODEL_INPUT_CHANNELS, pretrained=args.pretrained
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        best_auroc = -1.0
        best_epoch = 0
        epochs_without_improvement = 0
        history = []
        start_epoch = 1
        latest_checkpoint_path = output_dir / "latest_checkpoint.pt"
        if args.resume:
            resume_path = latest_checkpoint_path
            legacy_resume = False
            if not resume_path.exists() and (output_dir / "best_model.pt").exists():
                resume_path = output_dir / "best_model.pt"
                legacy_resume = True
            if not resume_path.exists():
                raise FileNotFoundError(
                    f"Cannot resume: no checkpoint found in {output_dir}"
                )
            checkpoint = torch.load(resume_path, map_location=device)
            saved_args = checkpoint.get("args", {})
            saved_input_channels = int(saved_args.get("input_channels", 3))
            if saved_input_channels != MODEL_INPUT_CHANNELS:
                raise ValueError(
                    "Cannot resume a checkpoint with a different model input: "
                    f"checkpoint={saved_input_channels} channels, "
                    f"current={MODEL_INPUT_CHANNELS} channels. Start a new run."
                )
            for name in ("target", "image_size", "roi_size", "model", "preprocessed"):
                saved_value = saved_args.get(
                    name, "resnet18" if name == "model" else None
                )
                current_value = getattr(args, name)
                if saved_value != current_value:
                    raise ValueError(
                        f"Cannot resume with a different {name}: "
                        f"checkpoint={saved_value!r}, requested={current_value!r}"
                    )
            model.load_state_dict(checkpoint["model_state"])
            if "optimizer_state" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state"])
            history = checkpoint.get("history", [])
            best_auroc = float(checkpoint.get("best_auroc", -1.0))
            best_epoch = int(checkpoint.get("best_epoch", 0))
            epochs_without_improvement = int(
                checkpoint.get("epochs_without_improvement", 0)
            )
            completed_epoch = int(checkpoint.get("epoch", 0))
            if legacy_resume:
                log_text = (output_dir / "train.log").read_text(
                    encoding="utf-8", errors="ignore"
                )
                completed = [
                    int(value)
                    for value in re.findall(r"Epoch\s+(\d+)/\d+\s+done", log_text)
                ]
                completed_epoch = max(completed, default=0)
                for line in log_text.splitlines():
                    if line.startswith("{'epoch':"):
                        try:
                            record = ast.literal_eval(line)
                        except (SyntaxError, ValueError):
                            continue
                        if isinstance(record, dict) and "epoch" in record:
                            history.append(record)
                print(
                    "Legacy resume: loaded best_model.pt; optimizer state was not "
                    "available and has been freshly initialized."
                )
            start_epoch = completed_epoch + 1
            print(
                f"Resumed after epoch {start_epoch - 1}; "
                f"continuing at epoch {start_epoch}/{args.epochs}"
            )
        training_start = time.time()
        publish_progress(status="running", started_at=dt.datetime.now().isoformat(timespec="seconds"))

        for epoch in range(start_epoch, args.epochs + 1):
            epoch_start = time.time()
            model.train()
            running_loss = 0.0

            total_batches = len(train_loader)
            progress_interval = max(total_batches // 100, 1)
            for batch_index, batch in enumerate(
                tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"), start=1
            ):
                x = batch["image"].to(device)
                y = batch["label"].to(device)

                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                running_loss += float(loss.item()) * x.size(0)
                if batch_index % progress_interval == 0 or batch_index == total_batches:
                    batch_percent = 100.0 * batch_index / max(total_batches, 1)
                    elapsed_seconds = time.time() - training_start
                    completed_fraction = (
                        (epoch - 1) + batch_index / max(total_batches, 1)
                    ) / max(args.epochs, 1)
                    eta_seconds = (
                        elapsed_seconds * (1.0 - completed_fraction) / completed_fraction
                        if completed_fraction > 0 else None
                    )
                    publish_progress(
                        status="running",
                        epoch=epoch,
                        batch=batch_index,
                        total_batches=total_batches,
                        epoch_progress=batch_index / max(total_batches, 1),
                        overall_progress=completed_fraction,
                        elapsed_seconds=elapsed_seconds,
                        eta_seconds=eta_seconds,
                        running_train_loss=running_loss / max(batch_index * args.batch_size, 1),
                    )
                    print(
                        f"Batch progress epoch {epoch}/{args.epochs}: "
                        f"{batch_index}/{total_batches} ({batch_percent:.1f}%)",
                        flush=True,
                    )

            train_loss = running_loss / len(train_ds)
            publish_progress(status="validating", epoch=epoch, batch=total_batches)
            val_metrics = evaluate(model, val_loader, device, criterion)
            epoch_seconds = time.time() - epoch_start
            elapsed_seconds = time.time() - training_start
            epochs_this_session = epoch - start_epoch + 1
            average_epoch_seconds = elapsed_seconds / epochs_this_session
            remaining_epochs = args.epochs - epoch
            eta_seconds = average_epoch_seconds * remaining_epochs
            finish_time = dt.datetime.now() + dt.timedelta(seconds=eta_seconds)

            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "epoch_seconds": epoch_seconds,
                "elapsed_seconds": elapsed_seconds,
                "eta_seconds": eta_seconds,
            }
            record.update({f"val_{key}": value for key, value in val_metrics.items()})
            history.append(record)
            print(record)
            print(
                f"Epoch {epoch}/{args.epochs} done | "
                f"epoch={format_duration(epoch_seconds)} | "
                f"elapsed={format_duration(elapsed_seconds)} | "
                f"remaining={format_duration(eta_seconds)} | "
                f"ETA finish={finish_time.strftime('%Y-%m-%d %H:%M:%S')}"
            )

            score = select_checkpoint_score(val_metrics)
            improved = score > best_auroc + args.early_stopping_min_delta
            if improved:
                best_auroc = score
                best_epoch = epoch
                epochs_without_improvement = 0
                val_labels, val_probs = collect_predictions(model, val_loader, device)
                decision_threshold, threshold_df = choose_decision_threshold(
                    val_labels,
                    val_probs,
                    args.threshold_mode,
                    args.min_high_recall,
                )
                threshold_metrics = metrics_from_predictions(
                    val_labels,
                    val_probs,
                    decision_threshold,
                )
                save_checkpoint_atomic(
                    {
                        "model_state": model.state_dict(),
                        "args": training_args,
                        "target": args.target,
                        "val_metrics": val_metrics,
                        "decision_threshold": decision_threshold,
                        "threshold_mode": args.threshold_mode,
                        "threshold_metrics": threshold_metrics,
                    },
                    output_dir / "best_model.pt",
                )
                threshold_df.to_csv(output_dir / "validation_thresholds.csv", index=False)
                save_threshold_plot(threshold_df, decision_threshold, output_dir)
                print(
                    "Saved new best_model.pt with "
                    f"checkpoint score {score:.6f}, "
                    f"threshold={decision_threshold:.2f}, "
                    f"val balanced accuracy={threshold_metrics['balanced_accuracy']:.4f}"
                )
            else:
                epochs_without_improvement += 1
                if args.early_stopping_patience > 0:
                    print(
                        "Early stopping wait: "
                        f"{epochs_without_improvement}/{args.early_stopping_patience} "
                        f"epochs without improvement > {args.early_stopping_min_delta:g}"
                    )

            publish_progress(
                status="running",
                epoch=epoch,
                batch=total_batches,
                total_batches=total_batches,
                epoch_progress=1.0,
                overall_progress=epoch / max(args.epochs, 1),
                elapsed_seconds=elapsed_seconds,
                eta_seconds=eta_seconds,
                train_loss=float(train_loss),
                val_loss=float(val_metrics.get("loss", float("nan"))),
                val_auroc=float(val_metrics.get("auroc", float("nan"))),
                val_accuracy=float(val_metrics.get("accuracy", float("nan"))),
                best_validation_score=float(best_auroc),
                best_epoch=int(best_epoch),
            )

            latest_checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_auroc": best_auroc,
                "best_epoch": best_epoch,
                "epochs_without_improvement": epochs_without_improvement,
                "history": history,
                "args": training_args,
            }
            save_checkpoint_atomic(latest_checkpoint, latest_checkpoint_path)
            with open(output_dir / "history.json", "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
            pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
            print(f"Saved resume checkpoint after epoch {epoch}")

            if (
                args.early_stopping_patience > 0
                and epochs_without_improvement >= args.early_stopping_patience
            ):
                print(
                    f"Early stopping at epoch {epoch}; best epoch={best_epoch}, "
                    f"best validation score={best_auroc:.6f}"
                )
                break

        completed_epoch = int(history[-1]["epoch"]) if history else 0
        stopped_early = completed_epoch < args.epochs
        training_summary = {
            "model": args.model,
            "target": args.target,
            "input_channels": MODEL_INPUT_CHANNELS,
            "channel_names": list(MODEL_CHANNEL_NAMES),
            "configured_max_epochs": args.epochs,
            "completed_epoch": completed_epoch,
            "best_epoch": best_epoch,
            "best_validation_score": best_auroc,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_min_delta": args.early_stopping_min_delta,
            "stopped_early": stopped_early,
            "stop_reason": "early_stopping" if stopped_early else "max_epochs",
        }
        with open(output_dir / "training_summary.json", "w", encoding="utf-8") as f:
            json.dump(training_summary, f, indent=2)

        publish_progress(
            status="evaluating_test",
            epoch=completed_epoch,
            best_validation_score=float(best_auroc),
            best_epoch=int(best_epoch),
        )
        checkpoint = torch.load(output_dir / "best_model.pt", map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        decision_threshold = float(checkpoint.get("decision_threshold", 0.5))
        test_labels, test_probs = collect_predictions(model, test_loader, device)
        test_preds = labels_from_threshold(test_probs, decision_threshold)
        test_metrics = metrics_from_predictions(test_labels, test_probs, decision_threshold)
        test_loss_metrics = evaluate(model, test_loader, device, criterion)
        if "loss" in test_loss_metrics:
            test_metrics["loss"] = test_loss_metrics["loss"]

        with open(output_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
        with open(output_dir / "test_metrics.json", "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, indent=2)
        pd.DataFrame(
            {
                "target": args.target or "label",
                "label": [INDEX_TO_LABEL[label] for label in test_labels],
                "predicted_label": [INDEX_TO_LABEL[pred] for pred in test_preds],
                "prob_high": test_probs,
                "decision_threshold": decision_threshold,
            }
        ).to_csv(output_dir / "test_predictions.csv", index=False)

        save_training_plots(history, output_dir)
        save_test_plots(test_labels, test_preds, test_probs, output_dir)
        save_preprocess_attention_examples(
            model,
            test_ds,
            device,
            output_dir,
            args.preview_images,
            decision_threshold,
        )

        print("test", test_metrics)
        print(f"decision_threshold {decision_threshold:.2f}")
        print(f"saved plots to {output_dir}")
        print(f"Finished at {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        publish_progress(
            status="complete",
            epoch=completed_epoch,
            batch=0,
            total_batches=0,
            epoch_progress=1.0,
            overall_progress=1.0,
            eta_seconds=0.0,
            best_validation_score=float(best_auroc),
            best_epoch=int(best_epoch),
            test_metrics=test_metrics,
            finished_at=dt.datetime.now().isoformat(timespec="seconds"),
        )
    except BaseException as error:
        publish_progress(
            status="failed",
            error=f"{type(error).__name__}: {error}",
            failed_at=dt.datetime.now().isoformat(timespec="seconds"),
        )
        raise
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_handle.close()
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
