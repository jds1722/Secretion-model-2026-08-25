from dataclasses import dataclass


@dataclass(frozen=True)
class LayerConfig:
    bf: int = 0
    nucleus: int = 1
    autofluorescence: int = 2
    bead_ins: int = 7
    bead_gcg: int = 5


@dataclass(frozen=True)
class PreprocessConfig:
    lower_percentile: float = 0.5
    upper_percentile: float = 99.5
    bead_threshold_percentile: float = 99.0
    autofluorescence_background_percentile: float = 5.0
    autofluorescence_signal_percentile: float = 99.5
    autofluorescence_threshold_fraction: float = 0.3
    nucleus_preserve_background_percentile: float = 10.0
    nucleus_preserve_signal_percentile: float = 99.5
    nucleus_preserve_threshold_fraction: float = 0.35
    bead_dilation_px: int = 3
    nucleus_preserve_dilation_px: int = 1
    fill_mode: str = "median"
    roi_size: int | None = None
    nucleus_object_min_size: int = 8
    dynamic_roi_min_size: int = 48
    nucleus_roi_scale: float = 4.0
    cell_roi_margin_fraction: float = 0.15
    cell_background_strength: float = 0.2
    cell_mask_feather_px: float = 3.0
    artifact_mask_feather_px: float = 3.0


LAYERS = LayerConfig()
PREPROCESS = PreprocessConfig()

# Model channels are ordered and versioned here so TIFF, cached-NPZ, training,
# and prediction paths cannot silently disagree about the input tensor.
MODEL_CHANNEL_NAMES = ("bf", "nucleus", "soft_cell_mask", "nucleus_bf")
MODEL_INPUT_CHANNELS = len(MODEL_CHANNEL_NAMES)
