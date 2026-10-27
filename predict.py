from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from src.dataset import INDEX_TO_LABEL
from src.model import create_binary_classifier, get_input_channels
from src.preprocess import build_model_input
from src.tiff_io import read_multilayer_tiff


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", help="A single model checkpoint.")
    parser.add_argument("--insulin-checkpoint", help="Independent insulin model checkpoint.")
    parser.add_argument("--glucagon-checkpoint", help="Independent glucagon model checkpoint.")
    parser.add_argument("--input", help="An unlabeled TIFF file or folder.")
    parser.add_argument("--high-input", help="A folder containing known-high TIFF files.")
    parser.add_argument("--low-input", help="A folder containing known-low TIFF files.")
    for target in ("insulin", "glucagon"):
        parser.add_argument(
            f"--{target}-high-input",
            help=f"A folder containing known-high {target} TIFF files.",
        )
        parser.add_argument(
            f"--{target}-low-input",
            help=f"A folder containing known-low {target} TIFF files.",
        )
    parser.add_argument("--out", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--roi-size",
        type=int,
        default=None,
        help="Override the checkpoint ROI size in original TIFF pixels.",
    )
    args = parser.parse_args()
    if not any([args.checkpoint, args.insulin_checkpoint, args.glucagon_checkpoint]):
        parser.error("Provide --checkpoint or at least one hormone-specific checkpoint.")
    if args.checkpoint and any([args.insulin_checkpoint, args.glucagon_checkpoint]):
        parser.error("Use --checkpoint by itself, or use hormone-specific checkpoints.")
    target_inputs = any([
        args.insulin_high_input, args.insulin_low_input,
        args.glucagon_high_input, args.glucagon_low_input,
    ])
    labeled_inputs = any([args.high_input, args.low_input]) or target_inputs
    if not args.input and not labeled_inputs:
        parser.error("Provide --input, or provide --high-input/--low-input.")
    if args.input and labeled_inputs:
        parser.error("Use --input by itself, or use labeled high/low inputs.")
    if target_inputs and any([args.high_input, args.low_input]):
        parser.error("Do not mix shared and hormone-specific labeled inputs.")
    if any([args.insulin_high_input, args.insulin_low_input]) and not args.insulin_checkpoint:
        parser.error("Insulin-labeled inputs require --insulin-checkpoint.")
    if any([args.glucagon_high_input, args.glucagon_low_input]) and not args.glucagon_checkpoint:
        parser.error("Glucagon-labeled inputs require --glucagon-checkpoint.")
    return args


def iter_tiffs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(list(path.rglob("*.tif")) + list(path.rglob("*.tiff")))


def load_model(
    checkpoint_path: str, device: torch.device
) -> tuple[str | None, torch.nn.Module, float, int | None]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_args = checkpoint.get("args", {})
    model_name = checkpoint_args.get("model", "resnet18")
    state = checkpoint["model_state"]
    input_weight_keys = (
        "conv1.weight",
        "features.0.0.weight",
    )
    input_channels = next(
        (int(state[key].shape[1]) for key in input_weight_keys if key in state),
        3,
    )
    model = create_binary_classifier(
        model_name, in_channels=input_channels, pretrained=False
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    checkpoint_roi_size = checkpoint.get("args", {}).get("roi_size")
    return (
        checkpoint.get("target"),
        model,
        float(checkpoint.get("decision_threshold", 0.5)),
        checkpoint_roi_size,
    )


def prediction_columns(
    target: str | None,
    probs: torch.Tensor,
    pred_idx: int,
    threshold: float,
) -> dict[str, str | float]:
    if target is None:
        return {
            "predicted_label": INDEX_TO_LABEL[pred_idx],
            "prob_low": float(probs[0].item()),
            "prob_high": float(probs[1].item()),
            "decision_threshold": threshold,
        }

    return {
        f"{target}_predicted_label": INDEX_TO_LABEL[pred_idx],
        f"{target}_prob_low": float(probs[0].item()),
        f"{target}_prob_high": float(probs[1].item()),
        f"{target}_decision_threshold": threshold,
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_inputs = any([
        args.insulin_high_input,
        args.insulin_low_input,
        args.glucagon_high_input,
        args.glucagon_low_input,
    ])

    models = []
    if args.checkpoint:
        models.append(load_model(args.checkpoint, device))
    else:
        if args.insulin_checkpoint:
            _, model, threshold, roi_size = load_model(args.insulin_checkpoint, device)
            models.append(("insulin", model, threshold, roi_size))
        if args.glucagon_checkpoint:
            _, model, threshold, roi_size = load_model(args.glucagon_checkpoint, device)
            models.append(("glucagon", model, threshold, roi_size))

    inputs: list[tuple[Path, str | None, str | None]] = []
    if args.input:
        inputs.extend((path, None, None) for path in iter_tiffs(Path(args.input)))
    elif target_inputs:
        for target in ("insulin", "glucagon"):
            for label in ("high", "low"):
                folder = getattr(args, f"{target}_{label}_input")
                if folder:
                    inputs.extend(
                        (path, label, target) for path in iter_tiffs(Path(folder))
                    )
    else:
        if args.high_input:
            inputs.extend((path, "high", None) for path in iter_tiffs(Path(args.high_input)))
        if args.low_input:
            inputs.extend((path, "low", None) for path in iter_tiffs(Path(args.low_input)))

    rows = []
    for path, true_label, input_target in inputs:
        stack = read_multilayer_tiff(path)
        row = {"path": str(path)}
        if true_label is not None:
            row["true_label"] = true_label
        if input_target is not None:
            row["target"] = input_target
        for target, model, threshold, checkpoint_roi_size in models:
            if input_target is not None and target != input_target:
                continue
            roi_size = args.roi_size if args.roi_size is not None else checkpoint_roi_size
            image, _ = build_model_input(stack, roi_size=roi_size)
            image = image[: get_input_channels(model)]
            x = torch.from_numpy(image).unsqueeze(0)
            x = F.interpolate(
                x,
                size=(args.image_size, args.image_size),
                mode="bilinear",
                align_corners=False,
            ).to(device)
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1)[0]
                pred_idx = 1 if float(probs[1].item()) >= threshold else 0
            row.update(prediction_columns(target, probs, pred_idx, threshold))
        rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote predictions to {out}")


if __name__ == "__main__":
    main()

