from __future__ import annotations

import datetime as dt
import json
import os
import time
from pathlib import Path
from typing import Callable, TypeVar


T = TypeVar("T")
DEFAULT_IO_RETRY_ATTEMPTS = 12


def retry_os_operation(
    operation: Callable[[], T],
    *,
    description: str,
    attempts: int = DEFAULT_IO_RETRY_ATTEMPTS,
) -> T:
    """Retry short-lived Windows file locks with exponential backoff."""
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    delay = 0.05
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except OSError:
            if attempt >= attempts:
                raise
            time.sleep(delay)
            delay = min(delay * 2.0, 1.0)
    raise RuntimeError(f"Unreachable retry state for {description}")


def temporary_path(path: Path) -> Path:
    return path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )


def write_text_atomic(
    path: str | Path,
    text: str,
    *,
    attempts: int = DEFAULT_IO_RETRY_ATTEMPTS,
) -> None:
    """Atomically publish text while tolerating readers that briefly lock it."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    def operation() -> None:
        temporary = temporary_path(destination)
        try:
            temporary.write_text(text, encoding="utf-8")
            os.replace(temporary, destination)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    retry_os_operation(
        operation,
        description=f"writing {destination.name}",
        attempts=attempts,
    )


def write_json_atomic(
    path: str | Path,
    payload: dict[str, object],
    *,
    attempts: int = DEFAULT_IO_RETRY_ATTEMPTS,
    add_timestamp: bool = True,
) -> None:
    """Serialize and atomically publish a JSON object."""
    published = dict(payload)
    if add_timestamp:
        published["updated_at"] = dt.datetime.now().isoformat(timespec="seconds")
    write_text_atomic(
        path,
        json.dumps(published, indent=2),
        attempts=attempts,
    )


def replace_file(
    source: str | Path,
    destination: str | Path,
    *,
    attempts: int = DEFAULT_IO_RETRY_ATTEMPTS,
) -> None:
    """Replace a file with retry protection for transient destination locks."""
    source_path = Path(source)
    destination_path = Path(destination)
    retry_os_operation(
        lambda: os.replace(source_path, destination_path),
        description=f"replacing {destination_path.name}",
        attempts=attempts,
    )
