from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from src.model import MODEL_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train independent insulin and glucagon models sequentially."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--preprocessed",
        action="store_true",
        help="Use NPZ tensors from preprocessed_path instead of source TIFFs.",
    )
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--image-size", type=int, required=True)
    parser.add_argument("--roi-size", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--preview-images", type=int, required=True)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--model", choices=MODEL_NAMES, default="resnet18")
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(root / ".matplotlib"))

    for target in ("insulin", "glucagon"):
        output_dir = f"{args.output_prefix}_{target}"
        command = [
            sys.executable,
            "train.py",
            "--manifest", args.manifest,
            "--target", target,
            "--output-dir", output_dir,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--image-size", str(args.image_size),
            "--roi-size", str(args.roi_size),
            "--lr", str(args.lr),
            "--weight-decay", str(args.weight_decay),
            "--num-workers", str(args.num_workers),
            "--preview-images", str(args.preview_images),
            "--model", args.model,
            "--early-stopping-patience", str(args.early_stopping_patience),
            "--early-stopping-min-delta", str(args.early_stopping_min_delta),
        ]
        if args.preprocessed:
            command.append("--preprocessed")
        if args.pretrained:
            command.append("--pretrained")
        result = subprocess.run(command, cwd=root, env=env, check=False)
        if result.returncode != 0:
            raise SystemExit(
                f"{target} training failed with exit code {result.returncode}; "
                "the next target was not started."
            )


if __name__ == "__main__":
    main()
