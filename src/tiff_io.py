from pathlib import Path

import numpy as np
from PIL import Image, ImageSequence


def read_multilayer_tiff(path: str | Path) -> np.ndarray:
    """Read a multi-page TIFF as an array shaped (layers, height, width)."""
    path = Path(path)
    try:
        import tifffile

        arr = tifffile.imread(path)
        arr = np.asarray(arr)
        if arr.ndim == 2:
            arr = arr[None, ...]
        if arr.ndim == 3:
            return arr
        raise ValueError(f"Unsupported TIFF shape from tifffile: {arr.shape}")
    except Exception:
        image = Image.open(path)
        frames = [np.asarray(frame) for frame in ImageSequence.Iterator(image)]
        if not frames:
            raise ValueError(f"No frames found in TIFF: {path}")
        return np.stack(frames, axis=0)


def robust_normalize(
    image: np.ndarray,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
) -> np.ndarray:
    """Percentile-normalize one image to float32 range [0, 1]."""
    image = image.astype(np.float32, copy=False)
    lo, hi = np.percentile(image, [lower_percentile, upper_percentile])
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    image = np.clip((image - lo) / (hi - lo), 0.0, 1.0)
    return image.astype(np.float32, copy=False)

