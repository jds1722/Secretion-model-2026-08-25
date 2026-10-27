from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training_manifest import assign_manifest_splits, source_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        required=True,
        help=(
            "Folder with high/ and low/ TIFFs, or hormone folders like "
            "insulin/high, insulin/low, glucagon/high, glucagon/low."
        ),
    )
    parser.add_argument("--out", required=True, help="Output manifest CSV path.")
    parser.add_argument("--val-fraction", type=float, default=1 / 6)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def find_tiffs(folder: Path) -> list[Path]:
    return sorted(list(folder.glob("*.tif")) + list(folder.glob("*.tiff")))


def collect_rows(data_root: Path) -> list[dict[str, str]]:
    rows = []
    hormone_dirs = [name for name in ("insulin", "glucagon") if (data_root / name).is_dir()]

    if hormone_dirs:
        for hormone in hormone_dirs:
            for label in ("high", "low"):
                for path in find_tiffs(data_root / hormone / label):
                    rows.append(
                        {
                            "path": str(path.resolve()),
                            "hormone": hormone,
                            "label": label,
                            "source_id": source_id(path),
                        }
                    )
        return rows

    for label in ("high", "low"):
        for path in find_tiffs(data_root / label):
            rows.append(
                {
                    "path": str(path.resolve()),
                    "label": label,
                    "source_id": source_id(path),
                }
            )
    return rows


def assign_splits(
    df: pd.DataFrame,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> pd.DataFrame:
    if test_fraction > 0:
        has_hormone = "hormone" in df.columns
        prepared = df if has_hormone else df.assign(hormone="all")
        result = assign_manifest_splits(prepared, val_fraction, test_fraction, seed)
        return result if has_hormone else result.drop(columns="hormone")
    groups = df[["label", "source_id"]].drop_duplicates().reset_index(drop=True)
    groups = groups.assign(split="train")

    if test_fraction > 0:
        train_val_groups, test_groups = train_test_split(
            groups,
            test_size=test_fraction,
            stratify=groups["label"],
            random_state=seed,
        )
        test_groups = test_groups.assign(split="test")
    else:
        train_val_groups = groups
        test_groups = groups.iloc[0:0]

    val_size = val_fraction / (1.0 - test_fraction) if test_fraction > 0 else val_fraction
    train_groups, val_groups = train_test_split(
        train_val_groups,
        test_size=val_size,
        stratify=train_val_groups["label"],
        random_state=seed,
    )

    split_groups = pd.concat(
        [
            train_groups.assign(split="train"),
            val_groups.assign(split="val"),
            test_groups,
        ],
        ignore_index=True,
    )
    split_groups = split_groups[["label", "source_id", "split"]]
    return df.drop(columns=["split"], errors="ignore").merge(
        split_groups,
        on=["label", "source_id"],
        how="left",
        validate="many_to_one",
    )


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    rows = collect_rows(data_root)

    if not rows:
        raise SystemExit(
            f"No TIFF files found under {data_root}/high and {data_root}/low or "
            f"{data_root}/insulin|glucagon/high|low"
        )

    df = pd.DataFrame(rows)
    if "hormone" in df.columns:
        out_df = pd.concat(
            [
                assign_splits(group, args.val_fraction, args.test_fraction, args.seed)
                for _, group in df.groupby("hormone")
            ],
            ignore_index=True,
        )
    else:
        out_df = assign_splits(df, args.val_fraction, args.test_fraction, args.seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    print(f"Wrote {len(out_df)} rows to {out}")
    group_cols = ["split", "label"]
    if "hormone" in out_df.columns:
        group_cols = ["hormone", "split", "label"]
    print(out_df.groupby(group_cols).size())


if __name__ == "__main__":
    main()
