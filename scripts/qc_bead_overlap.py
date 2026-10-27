from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.preprocess import bead_nucleus_overlap_fraction
from src.tiff_io import read_multilayer_tiff


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--exclude-threshold",
        type=float,
        default=0.0,
        help="Flag images with bead/nucleus overlap above this fraction.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.manifest)
    rows = []

    for _, row in df.iterrows():
        path = Path(row["path"])
        stack = read_multilayer_tiff(path)
        overlap = bead_nucleus_overlap_fraction(stack)
        rows.append(
            {
                **row.to_dict(),
                "bead_nucleus_overlap": overlap,
                "qc_flag": "review_or_exclude" if overlap > args.exclude_threshold else "ok",
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote QC table to {out}")


if __name__ == "__main__":
    main()
