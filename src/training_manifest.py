from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path, PureWindowsPath
from typing import Iterable

import pandas as pd


HORMONES = ("insulin", "glucagon")
LABELS = ("high", "low")
DEFAULT_TRAINING_MANIFEST_NAME = "training_manifest_balanced_test.csv"
AUGMENTATION_SUFFIX_RE = re.compile(
    r"(?:_(?:orig|rot90|rot180|rot270|flip_lr|shift_(?:up|down|left|right)_[0-9]+px))+$"
)


def source_id(path: str | Path) -> str:
    """Strip trailing rotation, flip and integer-pixel shift variants as one family.

    Numeric suffixes belonging to the source name are preserved: they cannot
    safely be treated as copies without provenance metadata.
    """
    text = str(path)
    name = PureWindowsPath(text).stem if "\\" in text else Path(text).stem
    return AUGMENTATION_SUFFIX_RE.sub("", name)


def is_original_variant(path: object) -> bool:
    """Return true for an unaugmented path or the explicit ``_orig`` variant."""
    text = str(path)
    stem = PureWindowsPath(text).stem if "\\" in text else Path(text).stem
    match = AUGMENTATION_SUFFIX_RE.search(stem)
    return match is None or match.group(0) == "_orig"


def tensor_filename(value: object) -> str:
    """Extract an NPZ filename from either a Windows or POSIX CSV path."""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return PureWindowsPath(text).name if "\\" in text else Path(text).name


def rebase_preprocessed_paths(
    dataframe: pd.DataFrame,
    preprocessing_folder: str | Path,
) -> tuple[pd.DataFrame, list[str]]:
    """Point manifest rows at the tensors stored under the selected dataset root.

    Preprocessing manifests can be copied from another computer or OneDrive
    location.  Only the NPZ filename is portable, so the selected folder is the
    source of truth and ``<folder>/tensors/<filename>`` is always preferred.
    """
    folder = Path(preprocessing_folder).expanduser().resolve()
    tensor_folder = folder / "tensors"
    rebased = dataframe.copy()
    filenames = rebased["preprocessed_path"].map(tensor_filename)
    local_by_name = {
        path.name: path.resolve()
        for path in tensor_folder.glob("*.npz")
        if path.is_file()
    } if tensor_folder.is_dir() else {}
    rebased["preprocessed_path"] = filenames.map(
        lambda name: str(
            local_by_name.get(name, (tensor_folder / name).resolve())
        )
    )
    missing = sorted(
        name for name in filenames.unique()
        if not name or name not in local_by_name
    )
    return rebased, missing


def _test_group_options(
    identifiers: list[str], sizes: dict[str, int], limit: int,
) -> tuple[int, dict[int, tuple[int, str]]]:
    """Reachable image counts using whole source groups, with backtracking."""
    reachable = 1  # Bit zero represents the empty subset.
    bounded = (1 << (limit + 1)) - 1
    parents: dict[int, tuple[int, str]] = {}
    for identifier in identifiers:
        size = sizes[identifier]
        if size > limit:
            continue
        additions = ((reachable << size) & bounded) & ~reachable
        reachable |= additions
        while additions:
            bit = additions & -additions
            total = bit.bit_length() - 1
            parents[total] = (total - size, identifier)
            additions ^= bit
    return reachable, parents


def assign_manifest_splits(
    dataframe: pd.DataFrame,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> pd.DataFrame:
    """Balance test image counts per hormone while keeping source groups intact.

    The test budget per class is floor(smaller_class_rows * test_fraction).
    Choose the largest shared reachable count at or below that budget. Preserve
    the previous validation allocation; assign all other non-test groups to train.
    """
    if validation_fraction <= 0 or test_fraction <= 0:
        raise ValueError("Validation and test fractions must both be positive.")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must sum to less than 1.")

    assignments: list[dict[str, str]] = []
    if dataframe.empty or dataframe[["hormone", "label", "source_id"]].isna().any().any():
        raise ValueError("Splitting requires nonempty hormone, label and source_id values.")
    for hormone, hormone_rows in dataframe.groupby("hormone"):
        if set(hormone_rows["label"]) != set(LABELS):
            raise ValueError(f"{hormone}: balanced test requires both high and low labels.")
        budget = int(hormone_rows.groupby("label").size().min() * test_fraction)
        options = {}
        for label in LABELS:
            class_rows = hormone_rows[hormone_rows["label"] == label]
            sizes = class_rows.groupby("source_id").size().to_dict()
            identifiers = sorted(
                sizes,
                key=lambda item: hashlib.sha1(
                    f"{seed}:{hormone}:{label}:{item}".encode("utf-8")
                ).hexdigest(),
            )
            count = len(identifiers)
            if count < 3:
                raise ValueError(
                    f"{hormone}/{label} needs at least 3 independent source groups; found {count}."
                )
            # Retain the validation groups selected by the previous splitter.
            old_test_count = max(1, int(round(count * test_fraction)))
            validation_count = max(1, int(round(count * validation_fraction)))
            while old_test_count + validation_count >= count:
                if old_test_count >= validation_count and old_test_count > 1:
                    old_test_count -= 1
                else:
                    validation_count -= 1
            validation_ids = set(
                identifiers[old_test_count:old_test_count + validation_count]
            )
            candidates = [item for item in identifiers if item not in validation_ids]
            # Reserve one group for training even for very small datasets.
            reachable, parents = _test_group_options(candidates[:-1], sizes, budget)
            options[label] = (identifiers, validation_ids, reachable, parents)
        shared = options["high"][2] & options["low"][2]
        test_images = shared.bit_length() - 1
        if test_images <= 0:
            raise ValueError(
                f"{hormone}: cannot form equal nonempty high/low test sets within "
                f"the {budget}-image budget while keeping source groups intact. "
                "Increase the test fraction or provide more source groups."
            )
        for label, (identifiers, validation_ids, _reachable, parents) in options.items():
            test_ids = set()
            total = test_images
            while total:
                total, identifier = parents[total]
                test_ids.add(identifier)
            for identifier in identifiers:
                assignments.append({
                    "hormone": hormone,
                    "label": label,
                    "source_id": identifier,
                    "split": "test" if identifier in test_ids else (
                        "val" if identifier in validation_ids else "train"
                    ),
                })
    assignment_df = pd.DataFrame(assignments)
    return dataframe.drop(columns=["split"], errors="ignore").merge(
        assignment_df,
        on=["hormone", "label", "source_id"],
        how="left",
        validate="many_to_one",
    )


def build_dataset_root_manifest(
    dataset_root: str | Path,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 17,
    hormones: Iterable[str] = HORMONES,
) -> tuple[pd.DataFrame | None, pd.DataFrame, list[str]]:
    """Discover the canonical hormone/label tree and make a trainable manifest."""
    root = Path(dataset_root).expanduser().resolve()
    selected_hormones = tuple(dict.fromkeys(str(item).lower() for item in hormones))
    errors: list[str] = []
    status_rows: list[dict[str, object]] = []
    categories: list[pd.DataFrame] = []

    if not root.is_dir():
        return None, pd.DataFrame(), [f"Dataset folder does not exist: {root}"]

    unknown = sorted(set(selected_hormones) - set(HORMONES))
    if unknown:
        return None, pd.DataFrame(), [
            "Unknown hormone folder(s): " + ", ".join(unknown)
        ]

    for hormone in selected_hormones:
        for label in LABELS:
            folder = root / hormone / label
            manifest_path = folder / "preprocessed_manifest.csv"
            status: dict[str, object] = {
                "hormone": hormone,
                "label": label,
                "folder": str(folder),
                "manifest_rows": 0,
                "ready_rows": 0,
                "source_groups": 0,
                "local_npz": 0,
                "missing_npz": 0,
                "status": "not ready",
            }
            if not manifest_path.is_file():
                errors.append(f"Missing manifest: {manifest_path}")
                status["status"] = "missing manifest"
                status_rows.append(status)
                continue
            try:
                category = pd.read_csv(manifest_path)
            except Exception as error:
                errors.append(f"Cannot read {manifest_path}: {error}")
                status["status"] = "unreadable manifest"
                status_rows.append(status)
                continue

            status["manifest_rows"] = len(category)
            required = {"path", "preprocessed_path"}
            missing_columns = sorted(required - set(category.columns))
            if missing_columns:
                errors.append(
                    f"{manifest_path} missing columns: {', '.join(missing_columns)}"
                )
                status["status"] = "invalid manifest"
                status_rows.append(status)
                continue

            if "preprocess_status" in category.columns:
                ready_mask = (
                    category["preprocess_status"].astype(str).str.lower() == "ok"
                )
                category = category.loc[ready_mask].copy()
            else:
                category = category.copy()

            category, missing_npz = rebase_preprocessed_paths(category, folder)
            category["hormone"] = hormone
            category["label"] = label
            category["source_id"] = category["path"].astype(str).map(source_id)
            status["ready_rows"] = len(category)
            status["source_groups"] = int(category["source_id"].nunique())
            status["local_npz"] = len(category) - len(missing_npz)
            status["missing_npz"] = len(missing_npz)
            if category.empty:
                errors.append(f"No successful preprocessing rows in {manifest_path}")
                status["status"] = "empty"
            elif missing_npz:
                errors.append(
                    f"{hormone}/{label}: {len(missing_npz):,} manifest NPZ "
                    "files are missing from the local tensors folder."
                )
                status["status"] = "missing NPZ"
            else:
                status["status"] = "ready"
                categories.append(category)
            status_rows.append(status)

    status_df = pd.DataFrame(status_rows)
    if errors or not categories:
        return None, status_df, errors or ["No trainable categories were found."]

    combined = pd.concat(categories, ignore_index=True)
    try:
        combined = assign_manifest_splits(
            combined,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            seed=seed,
        )
    except ValueError as error:
        return None, status_df, [str(error)]
    return combined, status_df, []


def write_training_manifest(dataframe: pd.DataFrame, output_path: str | Path) -> Path:
    """Atomically write a generated training manifest."""
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    dataframe.to_csv(temporary, index=False)
    os.replace(temporary, output)
    return output
