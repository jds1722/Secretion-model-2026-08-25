from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.dataset import INDEX_TO_LABEL
from src.model import create_binary_classifier, get_input_channels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict only from NPZ tensors produced by preprocess_dataset.py."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--insulin-checkpoint")
    parser.add_argument("--glucagon-checkpoint")
    parser.add_argument("--out", required=True)
    parser.add_argument("--image-size", type=int, default=None)
    args = parser.parse_args()
    checkpoints = [
        args.checkpoint, args.insulin_checkpoint, args.glucagon_checkpoint
    ]
    if not any(checkpoints):
        parser.error("Provide --checkpoint or at least one hormone checkpoint.")
    if args.checkpoint and any(
        [args.insulin_checkpoint, args.glucagon_checkpoint]
    ):
        parser.error("Use --checkpoint alone or hormone-specific checkpoints.")
    return args


def load_model(
    checkpoint_path: str,
    device: torch.device,
    forced_target: str | None = None,
) -> tuple[str | None, torch.nn.Module, float, int]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_args = checkpoint.get("args", {})
    if not saved_args.get("preprocessed", False):
        raise ValueError(
            f"Checkpoint was not trained in preprocessed mode: {checkpoint_path}"
        )
    model_name = saved_args.get("model", "resnet18")
    state = checkpoint["model_state"]
    input_weight_keys = ("conv1.weight", "features.0.0.weight")
    input_channels = next(
        (int(state[key].shape[1]) for key in input_weight_keys if key in state),
        int(saved_args.get("input_channels", 3)),
    )
    model = create_binary_classifier(
        model_name, in_channels=input_channels, pretrained=False
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    target = forced_target or checkpoint.get("target") or saved_args.get("target")
    return (
        target,
        model,
        float(checkpoint.get("decision_threshold", 0.5)),
        int(saved_args.get("image_size", 224)),
    )


def prediction_columns(
    target: str | None,
    probabilities: torch.Tensor,
    prediction: int,
    threshold: float,
) -> dict[str, str | float]:
    prefix = f"{target}_" if target else ""
    return {
        f"{prefix}predicted_label": INDEX_TO_LABEL[prediction],
        f"{prefix}prob_low": float(probabilities[0].item()),
        f"{prefix}prob_high": float(probabilities[1].item()),
        f"{prefix}decision_threshold": threshold,
    }


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    if "preprocessed_path" not in manifest.columns:
        raise ValueError("Manifest must contain a preprocessed_path column")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    if args.checkpoint:
        models.append(load_model(args.checkpoint, device))
    else:
        if args.insulin_checkpoint:
            models.append(
                load_model(args.insulin_checkpoint, device, "insulin")
            )
        if args.glucagon_checkpoint:
            models.append(
                load_model(args.glucagon_checkpoint, device, "glucagon")
            )

    rows = []
    total = len(manifest)
    for position, (_, source_row) in enumerate(manifest.iterrows(), 1):
        cache_path = Path(str(source_row["preprocessed_path"]))
        with np.load(cache_path, allow_pickle=False) as cached:
            image = np.asarray(cached["image"], dtype=np.float32)
        row = source_row.to_dict()
        input_target = str(source_row.get("hormone", "")).lower() or None
        for target, model, threshold, saved_image_size in models:
            if input_target and target and input_target != target:
                continue
            image_size = args.image_size or saved_image_size
            expected_channels = get_input_channels(model)
            if image.ndim != 3 or image.shape[0] != expected_channels:
                raise ValueError(
                    f"Tensor/checkpoint channel mismatch for {cache_path}: tensor "
                    f"shape={image.shape}, checkpoint expects {expected_channels} channels."
                )
            x = F.interpolate(
                torch.from_numpy(image).unsqueeze(0),
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False,
            ).to(device)
            with torch.no_grad():
                probabilities = torch.softmax(model(x), dim=1)[0]
            prediction = (
                1 if float(probabilities[1].item()) >= threshold else 0
            )
            row.update(
                prediction_columns(target, probabilities, prediction, threshold)
            )
        rows.append(row)
        if position % max(total // 100, 1) == 0 or position == total:
            print(f"Prediction progress: {position}/{total}", flush=True)

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Wrote {len(rows)} predictions to {output}")


if __name__ == "__main__":
    main()
