from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .config import MODEL_INPUT_CHANNELS
from .preprocess import build_model_input
from .tiff_io import read_multilayer_tiff


LABEL_TO_INDEX = {"low": 0, "high": 1}
INDEX_TO_LABEL = {0: "low", 1: "high"}
VALID_TARGETS = {"insulin", "glucagon"}


class SecretorTiffDataset(Dataset):
    def __init__(
        self,
        manifest_csv: str | Path,
        split: str | None = None,
        image_size: int = 224,
        target: str | None = None,
        roi_size: int | None = None,
    ) -> None:
        self.manifest_csv = Path(manifest_csv)
        df = pd.read_csv(self.manifest_csv)
        self.target = target.lower() if target is not None else None

        if self.target is not None:
            if self.target not in VALID_TARGETS:
                raise ValueError(
                    f"Unknown target {target!r}. Expected one of {sorted(VALID_TARGETS)}."
                )
            if "hormone" in df.columns and "label" in df.columns:
                df = df[df["hormone"].astype(str).str.lower() == self.target]
            elif f"{self.target}_label" not in df.columns:
                raise ValueError(
                    f"Manifest must contain either hormone/label columns or "
                    f"{self.target}_label when target={self.target!r}."
                )

        if split is not None and "split" in df.columns:
            df = df[df["split"] == split]
        if df.empty:
            raise ValueError(
                f"No rows found for split={split!r}, target={self.target!r} "
                f"in {manifest_csv}"
            )
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.roi_size = roi_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.df.iloc[index]
        path = Path(row["path"])
        stack = read_multilayer_tiff(path)
        image, masks = build_model_input(stack, roi_size=self.roi_size)
        if image.ndim != 3 or image.shape[0] != MODEL_INPUT_CHANNELS:
            raise ValueError(
                f"Expected {MODEL_INPUT_CHANNELS} model channels, got shape "
                f"{image.shape} for {path}"
            )

        x = torch.from_numpy(image)
        x = F.interpolate(
            x.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        ignore_mask = torch.from_numpy(masks["ignore"].astype("float32"))
        ignore_mask = F.interpolate(
            ignore_mask.unsqueeze(0).unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="nearest",
        ).squeeze(0).squeeze(0)

        if self.target is not None and f"{self.target}_label" in row:
            label_name = str(row[f"{self.target}_label"]).lower()
        else:
            label_name = str(row["label"]).lower()
        if label_name not in LABEL_TO_INDEX:
            raise ValueError(f"Unknown label {label_name!r} in {path}")
        y = torch.tensor(LABEL_TO_INDEX[label_name], dtype=torch.long)

        return {
            "image": x,
            "label": y,
            "path": str(path),
            "ignore_mask": ignore_mask.bool(),
        }
