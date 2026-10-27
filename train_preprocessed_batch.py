from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
from pathlib import Path

from src.model import MODEL_NAMES, MODEL_TRAINING_DEFAULTS
from src.resilient_io import write_json_atomic


def write_progress(path: Path, payload: dict[str, object]) -> None:
    """Publish queue state without letting a dashboard file lock stop training."""
    try:
        write_json_atomic(path, payload, attempts=12)
    except OSError as error:
        failures = int(payload.get("progress_write_failures", 0) or 0) + 1
        payload["progress_write_failures"] = failures
        payload["last_progress_write_error"] = str(error)
        if failures == 1 or failures % 10 == 0:
            print(
                "WARNING: batch progress file is temporarily locked; the training "
                f"queue will continue and retry on the next update: {error}",
                flush=True,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train selected target/model combinations sequentially from NPZ tensors."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument(
        "--targets", nargs="+", choices=("insulin", "glucagon"), required=True
    )
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, required=True)
    parser.add_argument(
        "--schedule", choices=("recommended", "custom"), required=True
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--preview-images", type=int, default=25)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.001)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    environment = os.environ.copy()
    environment.setdefault("MPLCONFIGDIR", str(root / ".matplotlib"))
    total = len(args.models) * len(args.targets)
    progress_path = Path(f"{args.output_prefix}_batch_progress.json")
    jobs = [
        {
            "index": index,
            "model": model_name,
            "target": target,
            "output_dir": f"{args.output_prefix}_{model_name}_{target}",
            "status": "waiting",
        }
        for index, (model_name, target) in enumerate(
            (
                (model_name, target)
                for model_name in args.models
                for target in args.targets
            ),
            1,
        )
    ]
    batch_state: dict[str, object] = {
        "status": "running",
        "pid": os.getpid(),
        "manifest": str(Path(args.manifest).resolve()),
        "schedule": args.schedule,
        "total_combinations": total,
        "current_combination": 0,
        "completed_combinations": 0,
        "current_model": None,
        "current_target": None,
        "current_output_dir": None,
        "jobs": jobs,
        "started_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    if not args.dry_run:
        write_progress(progress_path, batch_state)
    current = 0
    for model_name in args.models:
        defaults = MODEL_TRAINING_DEFAULTS[model_name]
        if args.schedule == "recommended":
            epochs = int(defaults["max_epochs"])
            batch_size = int(defaults["batch_size"])
            learning_rate = float(defaults["learning_rate"])
            patience = int(defaults["patience"])
        else:
            epochs = args.epochs
            batch_size = args.batch_size
            learning_rate = args.lr
            patience = args.patience
        for target in args.targets:
            current += 1
            output_dir = Path(
                f"{args.output_prefix}_{model_name}_{target}"
            )
            print(
                f"Starting combination {current}/{total}: {model_name}/{target} -> {output_dir}",
                flush=True,
            )
            if not args.dry_run:
                jobs[current - 1]["status"] = "running"
                batch_state.update(
                    {
                        "current_combination": current,
                        "current_model": model_name,
                        "current_target": target,
                        "current_output_dir": str(output_dir),
                    }
                )
                write_progress(progress_path, batch_state)
            command = [
                sys.executable,
                "train.py",
                "--preprocessed",
                "--manifest", args.manifest,
                "--target", target,
                "--model", model_name,
                "--output-dir", str(output_dir),
                "--epochs", str(epochs),
                "--batch-size", str(batch_size),
                "--image-size", str(args.image_size),
                "--lr", str(learning_rate),
                "--weight-decay", str(args.weight_decay),
                "--num-workers", str(args.num_workers),
                "--preview-images", str(args.preview_images),
                "--early-stopping-patience", str(patience),
                "--early-stopping-min-delta", str(args.early_stopping_min_delta),
            ]
            if args.pretrained:
                command.append("--pretrained")
            if args.dry_run:
                print(subprocess.list2cmdline(command), flush=True)
                continue
            result = subprocess.run(command, cwd=root, env=environment, check=False)
            if result.returncode != 0:
                jobs[current - 1]["status"] = "failed"
                batch_state.update(
                    {
                        "status": "failed",
                        "error": f"{model_name}/{target} exited with code {result.returncode}",
                        "failed_at": dt.datetime.now().isoformat(timespec="seconds"),
                    }
                )
                write_progress(progress_path, batch_state)
                raise SystemExit(
                    f"{model_name}/{target} failed with exit code "
                    f"{result.returncode}; remaining combinations were not started."
                )
            jobs[current - 1]["status"] = "complete"
            batch_state["completed_combinations"] = current
            write_progress(progress_path, batch_state)
    if not args.dry_run:
        batch_state.update(
            {
                "status": "complete",
                "current_model": None,
                "current_target": None,
                "current_output_dir": None,
                "finished_at": dt.datetime.now().isoformat(timespec="seconds"),
            }
        )
        write_progress(progress_path, batch_state)
    print(f"Completed all {total} target/model combinations.", flush=True)


if __name__ == "__main__":
    main()
