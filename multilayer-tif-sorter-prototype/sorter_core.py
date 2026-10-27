from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
import tifffile


@dataclass
class AnalysisResult:
    path: Path
    selected: bool
    overlap_percent: float
    nucleus_pixels: int
    signal_pixels: int
    overlap_pixels: int
    nucleus_object_count: int
    touching_nucleus_count: int
    preview_rgb: np.ndarray


def read_channels(path: Path) -> np.ndarray:
    """Read common TIFF layouts and return a C,Y,X array (UI uses zero-based indices)."""
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        data = series.asarray()
        axes = series.axes.upper()

    if "Y" not in axes or "X" not in axes:
        raise ValueError(f"Y/X 축을 찾을 수 없습니다: {axes}")

    data = np.moveaxis(data, (axes.index("Y"), axes.index("X")), (-2, -1))
    remaining_axes = [axis for axis in axes if axis not in "YX"]

    if "C" in remaining_axes:
        channel_axis = remaining_axes.index("C")
        data = np.moveaxis(data, channel_axis, 0)
        # Bright fluorescence may exist outside the first Z/T plane. Preserve it
        # by maximum-intensity projection over every non-channel/non-spatial axis.
        if data.ndim > 3:
            data = np.max(data, axis=tuple(range(1, data.ndim - 2)))
    else:
        # Multi-page grayscale TIFF: treat the leading page dimension as channels.
        leading_count = int(np.prod(data.shape[:-2])) if data.ndim > 2 else 1
        data = data.reshape((leading_count,) + data.shape[-2:])

    return np.asarray(data)


def otsu_threshold(image: np.ndarray) -> float:
    values = np.asarray(image, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0 or values.max() <= values.min():
        return float("inf")
    histogram, edges = np.histogram(values, bins=256, range=(values.min(), values.max()))
    centers = (edges[:-1] + edges[1:]) / 2
    weight_left = np.cumsum(histogram)
    weight_right = np.cumsum(histogram[::-1])[::-1]
    mean_left = np.cumsum(histogram * centers) / np.maximum(weight_left, 1)
    mean_right = (np.cumsum((histogram * centers)[::-1]) / np.maximum(weight_right[::-1], 1))[::-1]
    score = weight_left[:-1] * weight_right[1:] * (mean_left[:-1] - mean_right[1:]) ** 2
    return float(centers[int(np.argmax(score))])


def object_mask(image: np.ndarray, min_object_pixels: int) -> np.ndarray:
    mask = np.asarray(image) > otsu_threshold(image)
    labels, count = ndi.label(mask)
    if count:
        sizes = np.bincount(labels.ravel())
        mask = sizes[labels] >= min_object_pixels
        mask[labels == 0] = False
    return ndi.binary_fill_holes(mask)


def subtract_local_background(image: np.ndarray, radius_pixels: int) -> np.ndarray:
    """Remove slowly varying fluorescence background with a white top-hat transform."""
    values = np.asarray(image, dtype=np.float32)
    if radius_pixels <= 0:
        return values
    window = 2 * int(radius_pixels) + 1
    background = ndi.grey_opening(values, size=(window, window))
    return np.maximum(values - background, 0.0)


def normalize_u8(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=np.uint8)
    low, high = np.percentile(finite, (1, 99.8))
    if high <= low:
        return np.zeros(values.shape, dtype=np.uint8)
    return np.clip((values - low) * 255 / (high - low), 0, 255).astype(np.uint8)


def analyze_tiff(
    path: Path,
    min_object_pixels: int = 25,
    max_overlap_percent: float = 0.0,
    contact_margin_pixels: int = 0,
    background_radius_pixels: int = 10,
) -> AnalysisResult:
    channels = read_channels(path)
    if channels.shape[0] < 7:
        raise ValueError(f"채널이 {channels.shape[0]}개입니다. 인덱스 6까지 최소 7개가 필요합니다.")

    nucleus = object_mask(channels[1], min_object_pixels)
    corrected5 = subtract_local_background(channels[5], background_radius_pixels)
    corrected6 = subtract_local_background(channels[6], background_radius_pixels)
    layer5 = object_mask(corrected5, min_object_pixels)
    layer6 = object_mask(corrected6, min_object_pixels)
    signal = layer5 | layer6
    comparison = ndi.binary_dilation(signal, iterations=contact_margin_pixels) if contact_margin_pixels else signal

    nucleus_pixels = int(nucleus.sum())
    overlap_pixels = int((nucleus & comparison).sum())
    overlap_percent = 100.0 * overlap_pixels / max(nucleus_pixels, 1)
    selected = nucleus_pixels > 0 and overlap_percent <= max_overlap_percent

    nucleus_labels, nucleus_object_count = ndi.label(nucleus)
    touching_labels = np.unique(nucleus_labels[nucleus & comparison])
    touching_nucleus_count = int(np.count_nonzero(touching_labels))

    composite = np.zeros((*nucleus.shape, 3), dtype=np.uint8)
    composite[..., 2] = normalize_u8(channels[1])  # Index 1: blue
    composite[..., 0] = normalize_u8(corrected5)  # Index 5 after background removal: red
    composite[..., 1] = normalize_u8(corrected6)  # Index 6 after background removal: green

    # Diagnostic 1: detected masks. Index 1=blue, index 5=red, index 6=green.
    masks = np.zeros_like(composite)
    masks[..., 2] = nucleus.astype(np.uint8) * 255
    masks[..., 0] = layer5.astype(np.uint8) * 255
    masks[..., 1] = layer6.astype(np.uint8) * 255
    # Diagnostic 2: dim source with the tested overlap highlighted in yellow.
    overlap_view = (composite.astype(np.float32) * 0.22).astype(np.uint8)
    overlap_mask = nucleus & comparison
    overlap_view[overlap_mask] = (255, 255, 0)
    # White outline shows the complete detected layer-1 object boundary.
    nucleus_edge = nucleus ^ ndi.binary_erosion(nucleus)
    overlap_view[nucleus_edge & ~overlap_mask] = (235, 235, 255)

    separator = np.full((composite.shape[0], 4, 3), 70, dtype=np.uint8)
    preview = np.concatenate((composite, separator, masks, separator, overlap_view), axis=1)
    return AnalysisResult(
        path, selected, overlap_percent, nucleus_pixels, int(signal.sum()), overlap_pixels,
        int(nucleus_object_count), touching_nucleus_count, preview
    )
