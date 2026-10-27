from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage import filters, measure, morphology, segmentation

from .config import LAYERS, PREPROCESS, LayerConfig, PreprocessConfig
from .tiff_io import robust_normalize


def _safe_layer(stack: np.ndarray, layer_index: int) -> np.ndarray:
    if layer_index >= stack.shape[0]:
        raise IndexError(
            f"TIFF has {stack.shape[0]} layers, but layer {layer_index} was requested."
        )
    return stack[layer_index]


def make_bead_mask(
    stack: np.ndarray,
    layer_config: LayerConfig = LAYERS,
    preprocess_config: PreprocessConfig = PREPROCESS,
) -> np.ndarray:
    """Detect INS/GCG bead regions from array-indexed layers 7 and 5."""
    ins = _safe_layer(stack, layer_config.bead_ins).astype(np.float32)
    gcg = _safe_layer(stack, layer_config.bead_gcg).astype(np.float32)

    bead_signal = np.maximum(ins, gcg)
    threshold = np.percentile(bead_signal, preprocess_config.bead_threshold_percentile)
    mask = bead_signal > threshold

    mask = morphology.remove_small_objects(mask, min_size=4)
    if preprocess_config.bead_dilation_px > 0:
        footprint = morphology.disk(preprocess_config.bead_dilation_px)
        mask = morphology.binary_dilation(mask, footprint)
    return mask.astype(bool)


def make_autofluorescence_mask(
    stack: np.ndarray,
    layer_config: LayerConfig = LAYERS,
    preprocess_config: PreprocessConfig = PREPROCESS,
) -> np.ndarray:
    """Detect autofluorescent bead/nanowell structures from layer 2."""
    autofluorescence = _safe_layer(stack, layer_config.autofluorescence).astype(np.float32)
    background = np.percentile(
        autofluorescence,
        preprocess_config.autofluorescence_background_percentile,
    )
    signal = np.percentile(
        autofluorescence,
        preprocess_config.autofluorescence_signal_percentile,
    )
    threshold = background + (
        preprocess_config.autofluorescence_threshold_fraction * (signal - background)
    )
    mask = autofluorescence > threshold

    mask = morphology.remove_small_objects(mask, min_size=4)
    return mask.astype(bool)


def make_nucleus_preserve_mask(
    stack: np.ndarray,
    layer_config: LayerConfig = LAYERS,
    preprocess_config: PreprocessConfig = PREPROCESS,
) -> np.ndarray:
    """Detect layer-1 signal regions that should survive layer-2 removal."""
    nucleus = _safe_layer(stack, layer_config.nucleus).astype(np.float32)
    background = np.percentile(
        nucleus,
        preprocess_config.nucleus_preserve_background_percentile,
    )
    signal = np.percentile(
        nucleus,
        preprocess_config.nucleus_preserve_signal_percentile,
    )
    threshold = background + (
        preprocess_config.nucleus_preserve_threshold_fraction * (signal - background)
    )
    mask = nucleus > threshold

    mask = morphology.remove_small_objects(mask, min_size=8)
    mask = morphology.binary_closing(mask, morphology.disk(1))
    if preprocess_config.nucleus_preserve_dilation_px > 0:
        footprint = morphology.disk(preprocess_config.nucleus_preserve_dilation_px)
        mask = morphology.binary_dilation(mask, footprint)
    return mask.astype(bool)


def find_primary_nucleus(
    stack: np.ndarray,
    layer_config: LayerConfig = LAYERS,
    preprocess_config: PreprocessConfig = PREPROCESS,
) -> tuple[tuple[float, float], np.ndarray]:
    """Find the most likely cell nucleus and return its (row, col) centroid."""
    nucleus = _safe_layer(stack, layer_config.nucleus).astype(np.float32)
    background = np.percentile(
        nucleus, preprocess_config.nucleus_preserve_background_percentile
    )
    signal = np.percentile(
        nucleus, preprocess_config.nucleus_preserve_signal_percentile
    )
    threshold = background + (
        preprocess_config.nucleus_preserve_threshold_fraction * (signal - background)
    )
    mask = nucleus > threshold
    mask = morphology.remove_small_objects(
        mask, min_size=preprocess_config.nucleus_object_min_size
    )
    mask = morphology.binary_closing(mask, morphology.disk(1))

    labels = measure.label(mask)
    objects = measure.regionprops(labels, intensity_image=nucleus)
    if not objects:
        height, width = nucleus.shape
        return ((height - 1) / 2.0, (width - 1) / 2.0), mask.astype(bool)

    image_center = np.asarray([(nucleus.shape[0] - 1) / 2, (nucleus.shape[1] - 1) / 2])
    image_diagonal = max(float(np.linalg.norm(nucleus.shape)), 1.0)

    def object_score(obj: measure._regionprops.RegionProperties) -> float:
        # Prefer a large, bright nucleus while mildly favoring the nanowell center.
        distance = float(np.linalg.norm(np.asarray(obj.centroid) - image_center))
        centrality = 1.0 - min(distance / image_diagonal, 1.0)
        return float(obj.area * max(obj.mean_intensity - background, 0.0) * (0.5 + centrality))

    primary = max(objects, key=object_score)
    primary_mask = labels == primary.label
    return (float(primary.centroid[0]), float(primary.centroid[1])), primary_mask


def crop_square_roi(
    image: np.ndarray,
    center: tuple[float, float],
    roi_size: int,
    fill_value: float = 0.0,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop a fixed square around center, padding when it crosses an edge."""
    if roi_size <= 0:
        raise ValueError(f"roi_size must be positive, got {roi_size}")
    center_row, center_col = center
    top = int(round(center_row - (roi_size - 1) / 2))
    left = int(round(center_col - (roi_size - 1) / 2))
    bottom = top + roi_size
    right = left + roi_size

    output = np.full((roi_size, roi_size), fill_value, dtype=image.dtype)
    src_top, src_left = max(top, 0), max(left, 0)
    src_bottom, src_right = min(bottom, image.shape[0]), min(right, image.shape[1])
    if src_bottom > src_top and src_right > src_left:
        dst_top, dst_left = src_top - top, src_left - left
        output[
            dst_top : dst_top + (src_bottom - src_top),
            dst_left : dst_left + (src_right - src_left),
        ] = image[src_top:src_bottom, src_left:src_right]
    return output, (top, left, bottom, right)


def make_cell_candidate_mask(
    bf: np.ndarray,
    nucleus_mask: np.ndarray,
    nucleus_center: tuple[float, float],
    max_roi_size: int,
) -> np.ndarray:
    """Segment the BF cell using the detected nucleus as a mandatory foreground seed."""
    height, width = bf.shape
    rows, cols = np.ogrid[:height, :width]
    radius = max(max_roi_size / 2.0, 1.0)
    search_mask = (
        (rows - nucleus_center[0]) ** 2 + (cols - nucleus_center[1]) ** 2
    ) <= radius**2
    seed = morphology.binary_dilation(nucleus_mask, morphology.disk(2)) & search_mask
    if not seed.any():
        seed[int(round(nucleus_center[0])), int(round(nucleus_center[1]))] = True

    bf_smooth = filters.gaussian(bf.astype(np.float32), sigma=1.2, preserve_range=True)
    elevation = filters.sobel(bf_smooth)
    markers = np.zeros(bf.shape, dtype=np.int32)
    markers[seed] = 1
    background = search_mask & ~morphology.binary_erosion(
        search_mask, morphology.disk(max(2, int(round(radius * 0.15))))
    )
    markers[background] = 2
    labels = segmentation.watershed(elevation, markers, mask=search_mask)
    cell_mask = labels == 1
    cell_mask = morphology.binary_closing(cell_mask, morphology.disk(2))
    cell_mask = ndi.binary_fill_holes(cell_mask)

    # Reject a collapsed or runaway watershed and retain a nucleus-centered fallback.
    nucleus_area = max(int(nucleus_mask.sum()), 1)
    area = int(cell_mask.sum())
    max_area = int(np.pi * radius**2 * 0.9)
    if area < nucleus_area * 1.5 or area > max_area:
        fallback_radius = max(4, int(round(np.sqrt(nucleus_area / np.pi) * 4.0)))
        cell_mask = morphology.binary_dilation(nucleus_mask, morphology.disk(fallback_radius))
        cell_mask &= search_mask
    return cell_mask.astype(bool)


def dynamic_cell_roi_size(
    nucleus_mask: np.ndarray,
    cell_mask: np.ndarray,
    max_roi_size: int,
    preprocess_config: PreprocessConfig = PREPROCESS,
) -> int:
    """Choose a tight square crop from nucleus and segmented-cell dimensions."""
    nucleus_area = max(int(nucleus_mask.sum()), 1)
    nucleus_diameter = 2.0 * np.sqrt(nucleus_area / np.pi)
    nucleus_based = nucleus_diameter * preprocess_config.nucleus_roi_scale
    coords = np.argwhere(cell_mask)
    if coords.size:
        cell_span = float(np.max(np.ptp(coords, axis=0)) + 1)
    else:
        cell_span = nucleus_based
    cell_based = cell_span * (1.0 + 2.0 * preprocess_config.cell_roi_margin_fraction)
    size = int(np.ceil(max(nucleus_based, cell_based, preprocess_config.dynamic_roi_min_size)))
    return max(1, min(size, int(max_roi_size)))


def attenuate_cell_background(
    image: np.ndarray,
    cell_mask: np.ndarray,
    preprocess_config: PreprocessConfig = PREPROCESS,
) -> tuple[np.ndarray, np.ndarray]:
    """Dim background smoothly while retaining context near the segmented cell."""
    feather = max(float(preprocess_config.cell_mask_feather_px), 0.1)
    distance = ndi.distance_transform_edt(~cell_mask)
    soft_mask = np.exp(-distance / feather).astype(np.float32)
    weight = preprocess_config.cell_background_strength + (
        1.0 - preprocess_config.cell_background_strength
    ) * soft_mask
    return (image * weight).astype(np.float32), soft_mask


def mask_beads_in_model_channel(
    image: np.ndarray,
    bead_mask: np.ndarray,
    cell_mask: np.ndarray,
    feather_px: float = 3.0,
) -> np.ndarray:
    """Blend bead pixels into non-cell background without a hard mask edge."""
    image = image.astype(np.float32, copy=True)
    if not bead_mask.any():
        return image

    background = image[~bead_mask & ~cell_mask]
    if background.size == 0:
        background = image[~bead_mask]
    fill_value = float(np.median(background)) if background.size else 0.0
    return blend_mask_with_fill(image, bead_mask, fill_value, feather_px)


def make_autofluorescence_removal_mask(
    stack: np.ndarray,
    layer_config: LayerConfig = LAYERS,
    preprocess_config: PreprocessConfig = PREPROCESS,
) -> np.ndarray:
    """Remove broad layer-2 signal while preserving layer-1 signal regions."""
    autofluorescence_mask = make_autofluorescence_mask(
        stack,
        layer_config,
        preprocess_config,
    )
    nucleus_preserve_mask = make_nucleus_preserve_mask(
        stack,
        layer_config,
        preprocess_config,
    )
    return autofluorescence_mask & ~nucleus_preserve_mask


def make_ignore_mask(
    stack: np.ndarray,
    layer_config: LayerConfig = LAYERS,
    preprocess_config: PreprocessConfig = PREPROCESS,
) -> np.ndarray:
    """Combine all non-biological structures that should be hidden from the model."""
    bead_mask = make_bead_mask(stack, layer_config, preprocess_config)
    autofluorescence_mask = make_autofluorescence_removal_mask(
        stack,
        layer_config,
        preprocess_config,
    )
    return bead_mask | autofluorescence_mask


def make_nucleus_mask(
    stack: np.ndarray,
    layer_config: LayerConfig = LAYERS,
    preprocess_config: PreprocessConfig = PREPROCESS,
) -> np.ndarray:
    """Create a rough nucleus mask for QC, not for final scientific quantification."""
    nucleus = _safe_layer(stack, layer_config.nucleus).astype(np.float32)
    threshold = np.percentile(nucleus, preprocess_config.bead_threshold_percentile)
    mask = nucleus > threshold
    mask = morphology.remove_small_objects(mask, min_size=8)
    mask = morphology.binary_closing(mask, morphology.disk(1))
    return mask.astype(bool)


def bead_nucleus_overlap_fraction(stack: np.ndarray) -> float:
    """Return area(bead_mask intersect nucleus_mask) / area(nucleus_mask)."""
    bead_mask = make_ignore_mask(stack)
    nucleus_mask = make_nucleus_mask(stack)
    nucleus_area = int(nucleus_mask.sum())
    if nucleus_area == 0:
        return 0.0
    return float((bead_mask & nucleus_mask).sum() / nucleus_area)


def fill_masked_regions(
    image: np.ndarray,
    mask: np.ndarray,
    mode: str = "median",
    feather_px: float = 0.0,
) -> np.ndarray:
    """Replace masked pixels and optionally feather the replacement boundary."""
    image = image.astype(np.float32, copy=True)
    if not mask.any():
        return image

    valid = image[~mask]
    if valid.size == 0:
        fill_value = float(np.median(image))
    elif mode == "zero":
        fill_value = 0.0
    elif mode == "mean":
        fill_value = float(np.mean(valid))
    elif mode == "median":
        fill_value = float(np.median(valid))
    else:
        raise ValueError(f"Unknown fill mode: {mode}")

    return blend_mask_with_fill(image, mask, fill_value, feather_px)


def blend_mask_with_fill(
    image: np.ndarray,
    mask: np.ndarray,
    fill_value: float,
    feather_px: float,
) -> np.ndarray:
    """Fill the mask completely and blend its outside edge into nearby pixels."""
    image = image.astype(np.float32, copy=True)
    if not mask.any():
        return image
    if feather_px <= 0:
        image[mask] = fill_value
        return image

    distance = ndi.distance_transform_edt(~mask)
    alpha = np.zeros(mask.shape, dtype=np.float32)
    alpha[mask] = 1.0
    feather_region = (~mask) & (distance <= 4.0 * feather_px)
    alpha[feather_region] = np.exp(
        -0.5 * (distance[feather_region] / feather_px) ** 2
    ).astype(np.float32)
    return (image * (1.0 - alpha) + float(fill_value) * alpha).astype(np.float32)


def build_model_input(
    stack: np.ndarray,
    layer_config: LayerConfig = LAYERS,
    preprocess_config: PreprocessConfig = PREPROCESS,
    roi_size: int | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return BF, nucleus, cell-mask, and fluorescence-localized nucleus-BF channels."""
    bead_mask = make_bead_mask(stack, layer_config, preprocess_config)
    autofluorescence_mask = make_autofluorescence_mask(
        stack,
        layer_config,
        preprocess_config,
    )
    nucleus_preserve_mask = make_nucleus_preserve_mask(
        stack,
        layer_config,
        preprocess_config,
    )
    autofluorescence_removal_mask = autofluorescence_mask & ~nucleus_preserve_mask
    ignore_mask = bead_mask | autofluorescence_removal_mask

    bf = _safe_layer(stack, layer_config.bf)
    nucleus = _safe_layer(stack, layer_config.nucleus)

    bf = fill_masked_regions(
        bf,
        bead_mask,
        preprocess_config.fill_mode,
        preprocess_config.artifact_mask_feather_px,
    )
    nucleus = fill_masked_regions(
        nucleus,
        bead_mask,
        preprocess_config.fill_mode,
        preprocess_config.artifact_mask_feather_px,
    )

    bf = robust_normalize(
        bf,
        preprocess_config.lower_percentile,
        preprocess_config.upper_percentile,
    )
    nucleus = robust_normalize(
        nucleus,
        preprocess_config.lower_percentile,
        preprocess_config.upper_percentile,
    )

    bf = fill_masked_regions(
        bf,
        autofluorescence_removal_mask,
        preprocess_config.fill_mode,
        preprocess_config.artifact_mask_feather_px,
    )
    nucleus = fill_masked_regions(
        nucleus,
        autofluorescence_removal_mask,
        preprocess_config.fill_mode,
        preprocess_config.artifact_mask_feather_px,
    )

    nucleus_center, primary_nucleus_mask = find_primary_nucleus(
        stack, layer_config, preprocess_config
    )
    max_roi_size = roi_size if roi_size is not None else preprocess_config.roi_size
    if max_roi_size is None:
        max_roi_size = min(stack.shape[-2:])
    cell_mask = make_cell_candidate_mask(
        bf, primary_nucleus_mask, nucleus_center, int(max_roi_size)
    )
    # A bead attached to the cell must never become part of the segmented cell.
    cell_mask &= ~bead_mask
    effective_roi_size = dynamic_cell_roi_size(
        primary_nucleus_mask, cell_mask, int(max_roi_size), preprocess_config
    )
    roi_bounds = (0, 0, stack.shape[1], stack.shape[2])
    if effective_roi_size is not None:
        bf, roi_bounds = crop_square_roi(bf, nucleus_center, effective_roi_size)
        nucleus, _ = crop_square_roi(nucleus, nucleus_center, effective_roi_size)
        bead_mask, _ = crop_square_roi(bead_mask, nucleus_center, effective_roi_size)
        autofluorescence_mask, _ = crop_square_roi(
            autofluorescence_mask, nucleus_center, effective_roi_size
        )
        nucleus_preserve_mask, _ = crop_square_roi(
            nucleus_preserve_mask, nucleus_center, effective_roi_size
        )
        autofluorescence_removal_mask, _ = crop_square_roi(
            autofluorescence_removal_mask, nucleus_center, effective_roi_size
        )
        ignore_mask, _ = crop_square_roi(ignore_mask, nucleus_center, effective_roi_size)
        primary_nucleus_mask, _ = crop_square_roi(
            primary_nucleus_mask, nucleus_center, effective_roi_size
        )
        cell_mask, _ = crop_square_roi(cell_mask, nucleus_center, effective_roi_size)

    # Transfer the Layer-1 nucleus coordinates directly onto the cleaned Layer-0
    # BF image.  This is sampled before whole-cell background attenuation so the
    # fourth channel contains the actual normalized BF intensities at the nucleus.
    nucleus_bf = np.where(primary_nucleus_mask, bf, 0.0).astype(np.float32)

    bf, soft_cell_mask = attenuate_cell_background(bf, cell_mask, preprocess_config)
    # Re-mask after normalization, cropping, and background attenuation so the
    # exact tensor seen by the model contains no bead-shaped BF structure.
    bf = mask_beads_in_model_channel(
        bf,
        bead_mask,
        cell_mask,
        preprocess_config.artifact_mask_feather_px,
    )
    model_input = np.stack(
        [bf, nucleus, soft_cell_mask, nucleus_bf], axis=0
    ).astype(np.float32)
    masks = {
        "bead": bead_mask,
        "autofluorescence": autofluorescence_mask,
        "nucleus_preserve": nucleus_preserve_mask,
        "autofluorescence_removal": autofluorescence_removal_mask,
        "ignore": ignore_mask,
        "primary_nucleus": primary_nucleus_mask,
        "cell": cell_mask,
        "soft_cell": soft_cell_mask,
        "nucleus_bf": nucleus_bf,
        "nucleus_center": np.asarray(nucleus_center, dtype=np.float32),
        "roi_bounds": np.asarray(roi_bounds, dtype=np.int32),
        "effective_roi_size": np.asarray(effective_roi_size, dtype=np.int32),
    }
    return model_input, masks
