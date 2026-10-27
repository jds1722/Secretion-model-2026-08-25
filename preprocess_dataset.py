from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, TypeVar

import numpy as np
import pandas as pd

from src.config import MODEL_CHANNEL_NAMES, MODEL_INPUT_CHANNELS
from src.preprocess_v2 import ObjectPreprocessConfig, build_model_input_v2
from src.tiff_io import read_multilayer_tiff


PROGRESS_FILENAME = "preprocess_progress.json"
LOG_FILENAME = "preprocess.log"
LOCK_FILENAME = "preprocessing.lock"
TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
PROGRESS_WRITE_INTERVAL_SECONDS = 1.0
IO_RETRY_ATTEMPTS = 12
T = TypeVar("T")


class Tee:
    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            try:
                stream.write(data)
                stream.flush()
            except OSError:
                # A temporarily unavailable synced log file must not stop the job.
                continue
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            try:
                stream.flush()
            except OSError:
                continue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess TIFFs once into independent NPZ tensors."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", help="Source TIFF manifest CSV")
    source.add_argument(
        "--input-dir",
        help="Folder to search recursively for .tif and .tiff files.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--out-manifest", default=None)
    parser.add_argument("--roi-size", type=int, default=128)
    parser.add_argument("--bead-min-diameter", type=float, default=5.0)
    parser.add_argument("--bead-max-diameter", type=float, default=20.0)
    parser.add_argument("--bead-dilation-px", type=int, default=2)
    parser.add_argument("--nucleus-min-diameter", type=float, default=10.0)
    parser.add_argument("--nucleus-max-diameter", type=float, default=100.0)
    parser.add_argument("--cell-opening-px", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--fail-on-qc",
        action="store_true",
        help="Exclude samples without a valid nucleus from the trainable manifest.",
    )
    parser.add_argument(
        "--background",
        action="store_true",
        help=(
            "Launch an independent background process and return immediately. "
            f"Progress and logs are written to {PROGRESS_FILENAME} and {LOG_FILENAME}."
        ),
    )
    return parser.parse_args()


def retry_os_operation(
    operation: Callable[[], T],
    *,
    description: str,
    attempts: int = IO_RETRY_ATTEMPTS,
    announce: bool = False,
) -> T:
    """Retry transient filesystem failures common in synced/network folders."""
    delay = 0.05
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except OSError as error:
            if attempt >= attempts:
                raise
            if announce and (attempt == 1 or attempt % 4 == 0):
                print(
                    f"Transient I/O error during {description}; retry "
                    f"{attempt}/{attempts}: {error}",
                    flush=True,
                )
            time.sleep(delay)
            delay = min(delay * 2.0, 2.0)
    raise RuntimeError(f"Unreachable retry state for {description}")


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")


def write_text_atomic(
    path: Path,
    text: str,
    *,
    attempts: int = IO_RETRY_ATTEMPTS,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def operation() -> None:
        temporary = _temporary_path(path)
        try:
            temporary.write_text(text, encoding="utf-8")
            os.replace(temporary, path)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    retry_os_operation(
        operation,
        description=f"writing {path.name}",
        attempts=attempts,
    )


def write_json_atomic(
    path: Path,
    payload: dict[str, object],
    *,
    attempts: int = IO_RETRY_ATTEMPTS,
) -> None:
    payload = {**payload, "updated_at": dt.datetime.now().isoformat(timespec="seconds")}
    write_text_atomic(path, json.dumps(payload, indent=2), attempts=attempts)


def write_csv_atomic(
    dataframe: pd.DataFrame,
    path: Path,
    *,
    attempts: int = IO_RETRY_ATTEMPTS,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def operation() -> None:
        temporary = _temporary_path(path)
        try:
            dataframe.to_csv(temporary, index=False)
            os.replace(temporary, path)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    retry_os_operation(
        operation,
        description=f"writing {path.name}",
        attempts=attempts,
    )


def save_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def operation() -> None:
        temporary = path.with_name(
            f".{path.stem}.{os.getpid()}.{time.time_ns()}.tmp.npz"
        )
        try:
            np.savez_compressed(temporary, **arrays)
            os.replace(temporary, path)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    retry_os_operation(operation, description=f"writing {path.name}", announce=True)


def read_json_with_retry(path: Path) -> dict[str, object]:
    return retry_os_operation(
        lambda: json.loads(path.read_text(encoding="utf-8")),
        description=f"reading {path.name}",
        announce=True,
    )


def write_csv_with_fallback(dataframe: pd.DataFrame, path: Path) -> Path:
    """Publish a CSV, falling back to a unique name if a sync client holds the target."""
    try:
        write_csv_atomic(dataframe, path, attempts=20)
        return path
    except OSError as error:
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback = path.with_name(f"{path.stem}_{timestamp}_{os.getpid()}{path.suffix}")
        write_csv_atomic(dataframe, fallback, attempts=20)
        print(
            f"WARNING: could not replace {path} because it remained locked: {error}. "
            f"Published {fallback} instead.",
            flush=True,
        )
        return fallback


def acquire_lock(path: Path) -> int:
    delay = 0.05
    for attempt in range(1, IO_RETRY_ATTEMPTS + 1):
        try:
            return os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise
        except OSError:
            if attempt >= IO_RETRY_ATTEMPTS:
                raise
            time.sleep(delay)
            delay = min(delay * 2.0, 2.0)
    raise RuntimeError(f"Unable to acquire lock: {path}")


def remove_file_resilient(path: Path) -> None:
    try:
        retry_os_operation(
            lambda: path.unlink(missing_ok=True),
            description=f"removing {path.name}",
        )
    except OSError as error:
        print(f"WARNING: could not remove {path}: {error}", flush=True)


def cache_name(source: Path) -> str:
    digest = hashlib.sha1(str(source.resolve()).encode("utf-8")).hexdigest()[:12]
    return f"{source.stem}_{digest}.npz"


def discover_tiffs(input_dir: Path) -> list[Path]:
    """Return all TIFF files below a folder in deterministic path order."""
    if not input_dir.is_dir():
        raise ValueError(f"Input folder does not exist: {input_dir}")
    return sorted(
        path.resolve()
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    )


def cache_has_current_channels(path: Path) -> bool:
    if not path.is_file():
        return False

    def inspect() -> bool:
        with np.load(path, allow_pickle=False) as cached:
            image = cached["image"]
            return image.ndim == 3 and image.shape[0] == MODEL_INPUT_CHANNELS

    try:
        return retry_os_operation(
            inspect,
            description=f"reading {path.name}",
            attempts=6,
        )
    except (OSError, ValueError, KeyError):
        return False


def launch_background_process(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / LOCK_FILENAME
    progress_path = output_dir / PROGRESS_FILENAME
    if lock_path.exists():
        raise RuntimeError(
            f"This output folder already has a preprocessing lock: {lock_path}. "
            "Monitor the existing job or choose a different output folder."
        )

    child_args = [argument for argument in sys.argv[1:] if argument != "--background"]
    command = [sys.executable, "-u", str(Path(__file__).resolve()), *child_args]
    popen_kwargs: dict[str, object] = {
        "cwd": str(Path(__file__).resolve().parent),
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NO_WINDOW
        )
    else:
        popen_kwargs["start_new_session"] = True

    requested_at = dt.datetime.now().isoformat(timespec="seconds")
    launch_state = {
        "status": "launching",
        "pid": None,
        "launch_requested_at": requested_at,
        "source": str(args.manifest or args.input_dir),
        "output_dir": str(output_dir),
        "total": 0,
        "completed": 0,
        "processed": 0,
        "cached": 0,
        "ok": 0,
        "excluded_qc": 0,
        "errors": 0,
        "current_file": None,
        "elapsed_seconds": 0.0,
        "eta_seconds": None,
        "last_error": None,
    }
    try:
        write_json_atomic(progress_path, launch_state)
    except OSError as error:
        print(
            f"WARNING: launch status could not be written yet; the job will still "
            f"start and retry progress updates: {error}",
            flush=True,
        )
    try:
        process = subprocess.Popen(command, **popen_kwargs)
    except Exception as error:
        try:
            write_json_atomic(
                progress_path,
                {**launch_state, "status": "failed", "last_error": str(error)},
            )
        except OSError:
            pass
        raise

    job_pid = process.pid
    deadline = time.time() + 8.0
    while time.time() < deadline:
        try:
            state = json.loads(progress_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            state = {}
        if state.get("pid") and state.get("status") != "launching":
            job_pid = int(state["pid"])
            break
        time.sleep(0.1)
    print(f"Launched background preprocessing job PID {job_pid}")
    print(f"Progress: {progress_path}")
    print(f"Log: {output_dir / LOG_FILENAME}")
    return job_pid


def run_preprocessing(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / PROGRESS_FILENAME
    lock_path = output_dir / LOCK_FILENAME
    log_path = output_dir / LOG_FILENAME
    started_at = dt.datetime.now()
    started_clock = time.time()
    source_description = str(args.manifest or args.input_dir)
    progress: dict[str, object] = {
        "status": "initializing",
        "pid": os.getpid(),
        "started_at": started_at.isoformat(timespec="seconds"),
        "source": source_description,
        "output_dir": str(output_dir),
        "total": 0,
        "completed": 0,
        "processed": 0,
        "cached": 0,
        "ok": 0,
        "excluded_qc": 0,
        "errors": 0,
        "current_file": None,
        "elapsed_seconds": 0.0,
        "eta_seconds": None,
        "last_error": None,
        "progress_write_failures": 0,
        "last_progress_write_error": None,
    }

    try:
        lock_fd = acquire_lock(lock_path)
    except FileExistsError as error:
        raise RuntimeError(
            f"Preprocessing is already using {output_dir}, or a stale lock exists at "
            f"{lock_path}."
        ) from error
    try:
        with os.fdopen(lock_fd, "w", encoding="utf-8") as lock_file:
            lock_file.write(f"pid={os.getpid()}\n")
    except OSError:
        remove_file_resilient(lock_path)
        raise

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    try:
        log_handle = retry_os_operation(
            lambda: open(log_path, "a", encoding="utf-8"),
            description=f"opening {log_path.name}",
            announce=True,
        )
    except OSError:
        remove_file_resilient(lock_path)
        raise
    sys.stdout = Tee(original_stdout, log_handle)
    sys.stderr = Tee(original_stderr, log_handle)

    last_progress_write = 0.0
    progress_write_failures = 0

    def publish(*, force: bool = False, **values: object) -> None:
        nonlocal last_progress_write, progress_write_failures
        progress.update(values)
        progress["elapsed_seconds"] = time.time() - started_clock
        now = time.monotonic()
        if not force and now - last_progress_write < PROGRESS_WRITE_INTERVAL_SECONDS:
            return
        try:
            write_json_atomic(progress_path, progress, attempts=8 if not force else 20)
            last_progress_write = now
        except OSError as error:
            progress_write_failures += 1
            progress["progress_write_failures"] = progress_write_failures
            progress["last_progress_write_error"] = str(error)
            if progress_write_failures == 1 or progress_write_failures % 10 == 0:
                print(
                    "WARNING: progress file is temporarily unavailable; image "
                    f"processing will continue and status will retry: {error}",
                    flush=True,
                )

    try:
        publish(force=True)
        print("=" * 72)
        print(f"Preprocessing started at {started_at:%Y-%m-%d %H:%M:%S}")
        print(f"PID: {os.getpid()}")
        print(f"Source: {source_description}")
        print(f"Output: {output_dir}")

        out_manifest = (
            Path(args.out_manifest).expanduser().resolve()
            if args.out_manifest
            else output_dir / "preprocessed_manifest.csv"
        )
        publish(force=True, status="discovering")
        if args.manifest:
            source_manifest = Path(args.manifest).expanduser().resolve()
            df = retry_os_operation(
                lambda: pd.read_csv(source_manifest),
                description=f"reading {source_manifest.name}",
                announce=True,
            )
        else:
            paths = discover_tiffs(Path(args.input_dir).expanduser().resolve())
            if not paths:
                raise ValueError(f"No TIFF files found under: {args.input_dir}")
            df = pd.DataFrame({"path": [str(path) for path in paths]})
            source_manifest = output_dir / "source_manifest.csv"
            try:
                published_source_manifest = write_csv_with_fallback(df, source_manifest)
                print(
                    f"Found {len(df)} TIFF files; source manifest: "
                    f"{published_source_manifest}"
                )
            except OSError as error:
                print(
                    f"WARNING: source manifest could not be published, but the "
                    f"discovered TIFF list is retained in memory: {error}",
                    flush=True,
                )
        if "path" not in df.columns:
            raise ValueError("Source manifest must contain a path column")
        if df.empty:
            raise ValueError("Source manifest contains no TIFF rows")

        object_config = ObjectPreprocessConfig(
            bead_min_diameter=args.bead_min_diameter,
            bead_max_diameter=args.bead_max_diameter,
            bead_dilation_px=args.bead_dilation_px,
            nucleus_min_diameter=args.nucleus_min_diameter,
            nucleus_max_diameter=args.nucleus_max_diameter,
            cell_opening_px=args.cell_opening_px,
        )
        if object_config.bead_min_diameter > object_config.bead_max_diameter:
            raise ValueError("bead minimum diameter cannot exceed maximum diameter")
        if object_config.nucleus_min_diameter > object_config.nucleus_max_diameter:
            raise ValueError("nucleus minimum diameter cannot exceed maximum diameter")
        try:
            write_json_atomic(
                output_dir / "preprocess_run_config.json",
                {
                    **vars(args),
                    "background": False,
                    "input_channels": MODEL_INPUT_CHANNELS,
                    "channel_names": list(MODEL_CHANNEL_NAMES),
                },
                attempts=20,
            )
        except OSError as error:
            print(
                f"WARNING: run configuration could not be published, but image "
                f"processing will continue: {error}",
                flush=True,
            )

        total = len(df)
        counters = {
            "processed": 0,
            "cached": 0,
            "ok": 0,
            "excluded_qc": 0,
            "errors": 0,
        }
        output_rows: list[dict[str, object]] = []
        publish(force=True, status="running", total=total)

        for position, (_, row) in enumerate(df.iterrows(), start=1):
            result = row.to_dict()
            source = Path(str(row["path"]))
            destination = output_dir / "tensors" / cache_name(source)
            publish(current_file=str(source), current_index=position)
            action = "processed"
            try:
                def prepare_sample() -> tuple[dict[str, object], str]:
                    qc_path = destination.with_suffix(".qc.json")
                    needs_processing = (
                        args.overwrite
                        or not cache_has_current_channels(destination)
                        or not qc_path.exists()
                    )
                    if needs_processing:
                        stack = read_multilayer_tiff(source)
                        image, masks, sample_qc = build_model_input_v2(
                            stack,
                            object_config=object_config,
                            roi_size=args.roi_size,
                        )
                        save_npz_atomic(
                            destination,
                            image=image,
                            ignore_mask=masks["ignore"].astype(np.uint8),
                            cell_mask=masks["cell"].astype(np.uint8),
                            nucleus_mask=masks["primary_nucleus"].astype(np.uint8),
                        )
                        write_json_atomic(qc_path, sample_qc, attempts=20)
                        return sample_qc, "processed"
                    return read_json_with_retry(qc_path), "cached"

                qc, action = retry_os_operation(
                    prepare_sample,
                    description=f"processing {source.name}",
                    attempts=8,
                    announce=True,
                )
                if action == "processed":
                    counters["processed"] += 1
                else:
                    counters["cached"] += 1
                result.update(qc)
                result["preprocessed_path"] = str(destination.resolve())
                result["preprocess_status"] = (
                    "excluded_qc"
                    if args.fail_on_qc and not qc["nucleus_valid"]
                    else "ok"
                )
                result["preprocess_error"] = ""
                counters[str(result["preprocess_status"])] += 1
            except Exception as error:
                result["preprocessed_path"] = ""
                result["preprocess_status"] = "error"
                result["preprocess_error"] = str(error)
                counters["errors"] += 1
                progress["last_error"] = f"{source}: {error}"
                action = "error"
            output_rows.append(result)

            elapsed = time.time() - started_clock
            eta = elapsed * (total - position) / position if position else None
            publish(completed=position, eta_seconds=eta, **counters)
            print(
                f"[{position:,}/{total:,}] {action}: {source} | "
                f"ok={counters['ok']:,} excluded={counters['excluded_qc']:,} "
                f"errors={counters['errors']:,}",
                flush=True,
            )

        full_df = pd.DataFrame(output_rows)
        trainable_df = full_df[full_df["preprocess_status"] == "ok"].copy()
        published_manifest = write_csv_with_fallback(trainable_df, out_manifest)
        full_report = output_dir / "preprocess_qc_all.csv"
        published_report = write_csv_with_fallback(full_df, full_report)
        publish(
            force=True,
            status="completed",
            completed=total,
            current_file=None,
            eta_seconds=0.0,
            finished_at=dt.datetime.now().isoformat(timespec="seconds"),
            output_manifest=str(published_manifest),
            qc_report=str(published_report),
            **counters,
        )
        print(f"Wrote {len(trainable_df)} trainable rows to {published_manifest}")
        print(f"Full QC report: {published_report}")
        print("Preprocessing completed successfully.")
    except KeyboardInterrupt:
        publish(
            force=True,
            status="cancelled",
            finished_at=dt.datetime.now().isoformat(timespec="seconds"),
            eta_seconds=None,
        )
        print("Preprocessing cancelled.")
        raise
    except Exception as error:
        progress["last_error"] = str(error)
        publish(
            force=True,
            status="failed",
            finished_at=dt.datetime.now().isoformat(timespec="seconds"),
            eta_seconds=None,
        )
        traceback.print_exc()
        raise
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        try:
            log_handle.close()
        except OSError:
            pass
        remove_file_resilient(lock_path)


def main() -> None:
    args = parse_args()
    if args.background:
        launch_background_process(args)
        return
    run_preprocessing(args)


if __name__ == "__main__":
    main()
