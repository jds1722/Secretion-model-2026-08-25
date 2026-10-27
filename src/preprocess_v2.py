from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage import feature, filters, measure, morphology, segmentation

from .config import LAYERS, PREPROCESS, LayerConfig, PreprocessConfig
from .preprocess import (
    attenuate_cell_background,
    crop_square_roi,
    dynamic_cell_roi_size,
    fill_masked_regions,
    make_autofluorescence_mask,
    make_cell_candidate_mask,
    make_nucleus_preserve_mask,
    mask_beads_in_model_channel,
)
from .tiff_io import robust_normalize


@dataclass(frozen=True)
class ObjectPreprocessConfig:
    bead_min_diameter: float = 5.0
    bead_max_diameter: float = 20.0
    bead_min_peak_distance: int = 7
    bead_dilation_px: int = 2
    nucleus_min_diameter: float = 10.0
    nucleus_max_diameter: float = 100.0
    nucleus_min_peak_distance: int = 7
    discard_border_nuclei: bool = True
    cell_opening_px: int = 4


OBJECT_PREPROCESS = ObjectPreprocessConfig()


def _safe_layer(stack: np.ndarray, index: int) -> np.ndarray:
    if index >= stack.shape[0]:
        raise IndexError(f"TIFF has {stack.shape[0]} layers; index {index} is required")
    return np.asarray(stack[index])


def _threshold(image: np.ndarray, method: str) -> float:
    values = image[np.isfinite(image)]
    if values.size == 0 or float(values.max()) <= float(values.min()):
        return float("inf")
    try:
        return float(filters.threshold_li(values)) if method == "li" else float(filters.threshold_otsu(values))
    except (ValueError, RuntimeError):
        return float(filters.threshold_otsu(values))


def identify_round_objects(
    image: np.ndarray,
    *,
    threshold_method: str,
    min_diameter: float,
    max_diameter: float,
    smoothing_sigma: float,
    min_peak_distance: int,
    discard_border: bool,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Threshold, declump, and size-filter fluorescence objects."""
    normalized = robust_normalize(image)
    detection_image = (
        filters.gaussian(normalized, sigma=smoothing_sigma, preserve_range=True)
        if smoothing_sigma > 0
        else normalized
    )
    binary = detection_image > _threshold(detection_image, threshold_method)
    min_area = max(4, int(np.pi * (min_diameter / 2) ** 2 / 3))
    binary = morphology.remove_small_objects(binary, max_size=max(0, min_area - 1))
    binary = ndi.binary_fill_holes(binary)
    if not binary.any():
        empty = np.zeros(binary.shape, bool)
        return empty, np.zeros(binary.shape, np.int32), 0

    distance = ndi.distance_transform_edt(binary)
    coordinates = feature.peak_local_max(
        distance, labels=binary, min_distance=max(1, int(min_peak_distance)), exclude_border=False
    )
    markers = np.zeros(binary.shape, dtype=np.int32)
    for marker_id, (row, col) in enumerate(coordinates, 1):
        markers[row, col] = marker_id
    if not markers.any():
        markers = measure.label(binary)
    labels = segmentation.watershed(-distance, markers, mask=binary)

    accepted = np.zeros(binary.shape, dtype=bool)
    for region in measure.regionprops(labels):
        diameter = float(region.equivalent_diameter_area)
        if not min_diameter <= diameter <= max_diameter:
            continue
        min_row, min_col, max_row, max_col = region.bbox
        touches_border = (
            min_row == 0 or min_col == 0
            or max_row == binary.shape[0] or max_col == binary.shape[1]
        )
        if discard_border and touches_border:
            continue
        accepted[labels == region.label] = True
    accepted_labels = measure.label(accepted)
    return accepted, accepted_labels, int(accepted_labels.max())


def make_object_bead_mask(
    stack: np.ndarray,
    layer_config: LayerConfig = LAYERS,
    object_config: ObjectPreprocessConfig = OBJECT_PREPROCESS,
) -> tuple[np.ndarray, dict[str, int]]:
    """Detect INS and GCG beads independently and combine their masks."""
    masks = []
    counts: dict[str, int] = {}
    for name, index, method in (
        ("ins", layer_config.bead_ins, "otsu"),
        ("gcg", layer_config.bead_gcg, "li"),
    ):
        mask, _labels, count = identify_round_objects(
            _safe_layer(stack, index),
            threshold_method=method,
            min_diameter=object_config.bead_min_diameter,
            max_diameter=object_config.bead_max_diameter,
            smoothing_sigma=0.0,
            min_peak_distance=object_config.bead_min_peak_distance,
            # Border beads remain artifacts and must still be hidden from the CNN.
            discard_border=False,
        )
        if object_config.bead_dilation_px > 0:
            mask = morphology.dilation(mask, morphology.disk(object_config.bead_dilation_px))
        masks.append(mask)
        counts[f"{name}_bead_count"] = count
    return masks[0] | masks[1], counts


def find_primary_nucleus_v2(
    stack: np.ndarray,
    layer_config: LayerConfig = LAYERS,
    object_config: ObjectPreprocessConfig = OBJECT_PREPROCESS,
) -> tuple[tuple[float, float], np.ndarray, dict[str, float | int | bool]]:
    nucleus = _safe_layer(stack, layer_config.nucleus).astype(np.float32)
    _mask, labels, count = identify_round_objects(
        nucleus,
        threshold_method="otsu",
        min_diameter=object_config.nucleus_min_diameter,
        max_diameter=object_config.nucleus_max_diameter,
        smoothing_sigma=0.0,
        min_peak_distance=object_config.nucleus_min_peak_distance,
        discard_border=object_config.discard_border_nuclei,
    )
    regions = measure.regionprops(labels, intensity_image=nucleus)
    if not regions:
        center = ((nucleus.shape[0] - 1) / 2.0, (nucleus.shape[1] - 1) / 2.0)
        return center, np.zeros(nucleus.shape, bool), {
            "nucleus_count": 0, "nucleus_valid": False, "nucleus_area": 0,
            "nucleus_eccentricity": float("nan"), "nucleus_solidity": float("nan"),
        }
    image_center = np.asarray([(nucleus.shape[0] - 1) / 2, (nucleus.shape[1] - 1) / 2])
    diagonal = max(float(np.linalg.norm(nucleus.shape)), 1.0)

    def score(region: measure._regionprops.RegionProperties) -> float:
        centrality = 1.0 - min(
            float(np.linalg.norm(np.asarray(region.centroid) - image_center)) / diagonal, 1.0
        )
        return float(region.area * region.intensity_mean * (0.5 + centrality))

    primary = max(regions, key=score)
    primary_mask = labels == primary.label
    return (float(primary.centroid[0]), float(primary.centroid[1])), primary_mask, {
        "nucleus_count": count,
        "nucleus_valid": True,
        "nucleus_area": int(primary.area),
        "nucleus_eccentricity": float(primary.eccentricity),
        "nucleus_solidity": float(primary.solidity),
    }


def _quality_metrics(
    raw_bf: np.ndarray, bf: np.ndarray, nucleus_mask: np.ndarray, cell_mask: np.ndarray
) -> dict[str, float | int | bool]:
    nucleus_area = int(nucleus_mask.sum())
    cell_area = int(cell_mask.sum())
    border = bool(
        cell_mask[0].any() or cell_mask[-1].any()
        or cell_mask[:, 0].any() or cell_mask[:, -1].any()
    )
    raw = np.asarray(raw_bf)
    dtype_max = np.iinfo(raw.dtype).max if np.issubdtype(raw.dtype, np.integer) else float(np.nanmax(raw))
    saturation = float(np.mean(raw >= dtype_max)) if dtype_max > 0 else 0.0
    return {
        "cell_area": cell_area,
        "cell_nucleus_area_ratio": float(cell_area / max(nucleus_area, 1)),
        "cell_touches_border": border,
        "bf_blur_laplacian_variance": float(np.var(ndi.laplace(bf.astype(np.float32)))),
        "bf_saturation_fraction": saturation,
    }


def build_model_input_v2(
    stack: np.ndarray,
    layer_config: LayerConfig = LAYERS,
    preprocess_config: PreprocessConfig = PREPROCESS,
    object_config: ObjectPreprocessConfig = OBJECT_PREPROCESS,
    roi_size: int | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, object]]:
    """Build a reusable CNN tensor, masks, and quantitative QC metadata."""
    bead_mask, bead_counts = make_object_bead_mask(stack, layer_config, object_config)
    autofluorescence = make_autofluorescence_mask(stack, layer_config, preprocess_config)
    nucleus_preserve = make_nucleus_preserve_mask(stack, layer_config, preprocess_config)
    autofluorescence_removal = autofluorescence & ~nucleus_preserve
    ignore_mask = bead_mask | autofluorescence_removal

    raw_bf = _safe_layer(stack, layer_config.bf)
    bf = fill_masked_regions(
        raw_bf, bead_mask, preprocess_config.fill_mode,
        0.0,
    )
    nucleus = fill_masked_regions(
        _safe_layer(stack, layer_config.nucleus), bead_mask,
        preprocess_config.fill_mode, 0.0,
    )
    bf = robust_normalize(bf, preprocess_config.lower_percentile, preprocess_config.upper_percentile)
    nucleus = robust_normalize(
        nucleus, preprocess_config.lower_percentile, preprocess_config.upper_percentile
    )
    bf = fill_masked_regions(
        bf, autofluorescence_removal, preprocess_config.fill_mode,
        0.0,
    )
    nucleus = fill_masked_regions(
        nucleus, autofluorescence_removal, preprocess_config.fill_mode,
        0.0,
    )

    center, primary_nucleus, nucleus_qc = find_primary_nucleus_v2(
        stack, layer_config, object_config
    )
    max_roi = int(roi_size or preprocess_config.roi_size or min(stack.shape[-2:]))
    if primary_nucleus.any():
        cell_mask = make_cell_candidate_mask(bf, primary_nucleus, center, max_roi)
        cell_mask &= ~bead_mask
        if object_config.cell_opening_px > 0:
            opened = morphology.opening(
                cell_mask, morphology.disk(object_config.cell_opening_px)
            )
            if opened.any():
                cell_mask = ndi.binary_fill_holes(opened | primary_nucleus)
        # Keep autofluorescence artifacts excluded even when the cell watershed
        # falls back to a nucleus-centered circular mask or restores the nucleus.
        cell_mask &= ~autofluorescence_removal
    else:
        cell_mask = np.zeros(bf.shape, bool)

    effective_roi = dynamic_cell_roi_size(
        primary_nucleus, cell_mask, max_roi, preprocess_config
    )
    roi_bounds = (0, 0, stack.shape[-2], stack.shape[-1])
    arrays = [
        bf, nucleus, bead_mask, autofluorescence, nucleus_preserve,
        autofluorescence_removal, ignore_mask, primary_nucleus, cell_mask,
    ]
    cropped = []
    for position, array in enumerate(arrays):
        crop, bounds = crop_square_roi(array, center, effective_roi)
        if position == 0:
            roi_bounds = bounds
        cropped.append(crop)
    (
        bf, nucleus, bead_mask, autofluorescence, nucleus_preserve,
        autofluorescence_removal, ignore_mask, primary_nucleus, cell_mask,
    ) = cropped
    raw_bf_roi, _ = crop_square_roi(raw_bf, center, effective_roi)

    # Apply the nucleus detected in Layer 1 to the same coordinates in Layer 0.
    nucleus_bf = np.where(primary_nucleus, bf, 0.0).astype(np.float32)

    bf, soft_cell = attenuate_cell_background(bf, cell_mask, preprocess_config)
    bf = mask_beads_in_model_channel(
        bf, bead_mask, cell_mask, preprocess_config.artifact_mask_feather_px
    )
    model_input = np.stack(
        [bf, nucleus, soft_cell, nucleus_bf], axis=0
    ).astype(np.float32)
    qc: dict[str, object] = {
        **bead_counts,
        **nucleus_qc,
        **_quality_metrics(raw_bf_roi, bf, primary_nucleus, cell_mask),
        "effective_roi_size": int(effective_roi),
        "preprocess_version": "object_qc_v2",
        "object_config": asdict(object_config),
    }
    masks = {
        "bead": bead_mask,
        "autofluorescence": autofluorescence,
        "nucleus_preserve": nucleus_preserve,
        "autofluorescence_removal": autofluorescence_removal,
        "ignore": ignore_mask,
        "primary_nucleus": primary_nucleus,
        "cell": cell_mask,
        "soft_cell": soft_cell,
        "nucleus_bf": nucleus_bf,
        "roi_bounds": np.asarray(roi_bounds, dtype=np.int32),
    }
    return model_input, masks, qc
