from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .config import MODEL_INPUT_CHANNELS
from .dataset import LABEL_TO_INDEX, VALID_TARGETS


class PreprocessedSecretorDataset(Dataset):
    """Read cached NPZ tensors without opening the original TIFF files."""

    def __init__(
        self,
        manifest_csv: str | Path,
        split: str | None,
        image_size: int,
        target: str | None = None,
        roi_size: int | None = None,
    ) -> None:
        del roi_size  # ROI was fixed during preprocessing.
        df = pd.read_csv(manifest_csv)
        self.target = target.lower() if target else None
        if "preprocessed_path" not in df.columns:
            raise ValueError("Preprocessed manifest needs a preprocessed_path column")
        if self.target:
            if self.target not in VALID_TARGETS:
                raise ValueError(f"Unknown target {target!r}")
            if "hormone" in df.columns:
                df = df[df["hormone"].astype(str).str.lower() == self.target]
        if split is not None and "split" in df.columns:
            df = df[df["split"] == split]
        if df.empty:
            raise ValueError(f"No rows for split={split!r}, target={target!r}")
        self.df = df.reset_index(drop=True)
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.df.iloc[index]
        cache_path = Path(row["preprocessed_path"])
        with np.load(cache_path, allow_pickle=False) as cached:
            image = np.asarray(cached["image"], dtype=np.float32)
            ignore = np.asarray(cached["ignore_mask"], dtype=np.float32)
        if image.ndim != 3 or image.shape[0] != MODEL_INPUT_CHANNELS:
            raise ValueError(
                f"Cached tensor {cache_path} has shape {image.shape}; expected "
                f"({MODEL_INPUT_CHANNELS}, H, W). Re-run preprocessing to create "
                "the four-channel cache."
            )
        x = F.interpolate(
            torch.from_numpy(image).unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        ignore_mask = F.interpolate(
            torch.from_numpy(ignore).unsqueeze(0).unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="nearest",
        ).squeeze().bool()
        label_column = f"{self.target}_label" if self.target else "label"
        if label_column not in row:
            label_column = "label"
        label_name = str(row[label_column]).lower()
        if label_name not in LABEL_TO_INDEX:
            raise ValueError(f"Unknown label {label_name!r} in {cache_path}")
        return {
            "image": x,
            "label": torch.tensor(LABEL_TO_INDEX[label_name], dtype=torch.long),
            "path": str(row.get("path", cache_path)),
            "ignore_mask": ignore_mask,
        }
