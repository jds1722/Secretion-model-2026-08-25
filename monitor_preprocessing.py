from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

PROGRESS_FILENAME = "preprocess_progress.json"
LOG_FILENAME = "preprocess.log"
TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor one or more independent preprocessing jobs."
    )
    parser.add_argument(
        "--output-dir",
        action="append",
        required=True,
        help="Preprocessing output folder. Repeat for multiple simultaneous jobs.",
    )
    parser.add_argument("--watch", action="store_true", help="Refresh until all jobs finish.")
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--pause-on-exit", action="store_true")
    parser.add_argument("--log-lines", type=int, default=8)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def format_duration(value: object) -> str:
    if value is None:
        return "-"
    try:
        seconds = max(0, int(float(value)))
    except (TypeError, ValueError):
        return "-"
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def render(output_dirs: list[Path], log_lines: int) -> bool:
    print("\033[2J\033[H", end="")
    print(f"Preprocessing monitor | {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("Press Ctrl+C to stop monitoring; background jobs will continue.\n")
    all_terminal = True
    for output_dir in output_dirs:
        progress = load_json(output_dir / PROGRESS_FILENAME)
        if not progress:
            all_terminal = False
            print(f"{output_dir.name}: waiting for {PROGRESS_FILENAME}")
            continue
        status = str(progress.get("status", "unknown"))
        total = int(progress.get("total", 0) or 0)
        completed = int(progress.get("completed", 0) or 0)
        percent = 100.0 * completed / total if total else 0.0
        all_terminal = all_terminal and status in TERMINAL_STATUSES
        print(f"{output_dir.name} | {status.upper()} | PID {progress.get('pid', '-')}")
        print(
            f"  {completed:,}/{total:,} ({percent:5.1f}%) | "
            f"processed {int(progress.get('processed', 0) or 0):,} | "
            f"cached {int(progress.get('cached', 0) or 0):,} | "
            f"ok {int(progress.get('ok', 0) or 0):,} | "
            f"excluded {int(progress.get('excluded_qc', 0) or 0):,} | "
            f"errors {int(progress.get('errors', 0) or 0):,}"
        )
        print(
            f"  elapsed {format_duration(progress.get('elapsed_seconds'))} | "
            f"ETA {format_duration(progress.get('eta_seconds'))}"
        )
        current = progress.get("current_file")
        if current:
            print(f"  current: {current}")
        if progress.get("last_error"):
            print(f"  last error: {progress['last_error']}")
        recovered_writes = int(progress.get("progress_write_failures", 0) or 0)
        if recovered_writes:
            print(f"  recovered progress-file conflicts: {recovered_writes:,}")

        if status in {"failed", "cancelled"} and log_lines > 0:
            log_path = output_dir / LOG_FILENAME
            try:
                tail = log_path.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()[-log_lines:]
            except OSError:
                tail = []
            if tail:
                print("  recent log:")
                for line in tail:
                    print(f"    {line}")
        print()
    return all_terminal


def main() -> None:
    args = parse_args()
    if args.interval <= 0:
        raise ValueError("--interval must be positive")
    output_dirs = [Path(value).expanduser().resolve() for value in args.output_dir]
    try:
        while True:
            all_terminal = render(output_dirs, args.log_lines)
            if not args.watch or all_terminal:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nMonitor closed. Background preprocessing jobs were not stopped.")
    if args.pause_on_exit:
        input("Monitoring finished. Press Enter to close this window...")


if __name__ == "__main__":
    main()
