from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from src.model import MODEL_NAMES, MODEL_TRAINING_DEFAULTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train all supported architectures sequentially.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--preprocessed",
        action="store_true",
        help="Use NPZ tensors from preprocessed_path instead of source TIFFs.",
    )
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--target", choices=("both", "insulin", "glucagon"), required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, required=True)
    parser.add_argument("--roi-size", type=int, required=True)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--preview-images", type=int, required=True)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.001)
    for model_name in MODEL_NAMES:
        option = model_name.replace("_", "-")
        parser.add_argument(f"--{option}-epochs", type=int, default=None)
        parser.add_argument(f"--{option}-patience", type=int, default=None)
        parser.add_argument(f"--{option}-lr", type=float, default=None)
        parser.add_argument(f"--{option}-batch-size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(root / ".matplotlib"))
    targets = ("insulin", "glucagon") if args.target == "both" else (args.target,)

    for model_name in MODEL_NAMES:
        defaults = MODEL_TRAINING_DEFAULTS[model_name]
        dest_prefix = model_name
        epochs = getattr(args, f"{dest_prefix}_epochs") or args.epochs or defaults["max_epochs"]
        patience = getattr(args, f"{dest_prefix}_patience")
        if patience is None:
            patience = defaults["patience"]
        learning_rate = getattr(args, f"{dest_prefix}_lr") or args.lr or defaults["learning_rate"]
        batch_size = (
            getattr(args, f"{dest_prefix}_batch_size")
            or args.batch_size
            or defaults["batch_size"]
        )
        for target in targets:
            suffix = f"_{model_name}_{target}" if args.target == "both" else f"_{model_name}"
            output_dir = f"{args.output_prefix}{suffix}"
            command = [
                sys.executable, "train.py",
                "--manifest", args.manifest,
                "--target", target,
                "--model", model_name,
                "--output-dir", output_dir,
                "--epochs", str(epochs),
                "--batch-size", str(batch_size),
                "--image-size", str(args.image_size),
                "--roi-size", str(args.roi_size),
                "--lr", str(learning_rate),
                "--weight-decay", str(args.weight_decay),
                "--num-workers", str(args.num_workers),
                "--preview-images", str(args.preview_images),
                "--early-stopping-patience", str(patience),
                "--early-stopping-min-delta", str(args.early_stopping_min_delta),
            ]
            if args.preprocessed:
                command.append("--preprocessed")
            if args.pretrained:
                command.append("--pretrained")
            result = subprocess.run(command, cwd=root, env=env, check=False)
            if result.returncode != 0:
                raise SystemExit(
                    f"{model_name}/{target} failed with exit code {result.returncode}; "
                    "remaining comparison runs were not started."
                )


if __name__ == "__main__":
    main()
