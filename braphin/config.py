from dataclasses import dataclass, field


@dataclass
class InputConfig:
    """Configuration for input data handling."""

    allowed_fmri_extensions: list[str] = field(default_factory=lambda: [".nii", ".nii.gz"])
    allowed_aux_extensions: list[str] = field(
        default_factory=lambda: [".json", ".tsv", ".csv", ".npy"]
    )
    allow_tabular_inputs: bool = True


@dataclass
class PreprocessConfig:
    """Configuration for the preprocessing stage."""

    apply_motion_correction: bool = False
    apply_slice_timing: bool = False  # Requires tr; set tr when enabling
    apply_outlier_detection: bool = False
    apply_voxel_zscore: bool = False
    apply_smoothing: bool = False

    # Smoothing parameter
    smoothing_fwhm: float = 6.0  # FWHM in mm

    # Slice-timing parameters
    tr: float | None = None  # Repetition time in seconds (required for slice timing)
    slice_order: str = "sequential"  # "sequential" or "interleaved"
    slice_timing_ref_slice: int = 0  # Reference slice index (0 = first, -1 = middle)

    # Slice-axis parameter
    slice_axis: int = 2  # 0=X, 1=Y, 2=Z (axial, default)

    # Outlier detection parameters
    outlier_threshold_dvars: float = 1.5  # IQR multiplier for DVARS threshold
    scrubbing_strategy: str = "interpolate"  # "interpolate" or "mark"

    def __post_init__(self) -> None:
        if self.apply_slice_timing and self.tr is None:
            raise ValueError("PreprocessConfig: 'tr' must be set when apply_slice_timing=True.")
        if self.apply_smoothing and self.smoothing_fwhm <= 0:
            raise ValueError("PreprocessConfig: 'smoothing_fwhm' must be positive.")
        if self.outlier_threshold_dvars <= 0:
            raise ValueError("PreprocessConfig: 'outlier_threshold_dvars' must be positive.")
        if self.scrubbing_strategy not in ("interpolate", "mark"):
            raise ValueError(
                "PreprocessConfig: 'scrubbing_strategy' must be 'interpolate' or 'mark'."
            )
        if self.slice_axis not in (0, 1, 2):
            raise ValueError("PreprocessConfig: 'slice_axis' must be 0, 1, or 2.")


@dataclass
class DenoiseConfig:
    """Configuration for the denoising stage."""

    regress_confounds: bool = True
    apply_scrubbing: bool = False
    apply_bandpass: bool = False

    # Bandpass parameters
    tr: float | None = None  # Repetition time in seconds (required for bandpass)
    bandpass_low: float = 0.008  # Low cut-off in Hz
    bandpass_high: float = 0.1  # High cut-off in Hz (Biswal et al. 1995: 0.008–0.1 Hz)

    def __post_init__(self) -> None:
        if self.apply_bandpass and self.tr is None:
            raise ValueError("DenoiseConfig: 'tr' must be set when apply_bandpass=True.")
        if self.bandpass_low <= 0:
            raise ValueError("DenoiseConfig: 'bandpass_low' must be > 0 Hz.")
        if self.bandpass_high <= self.bandpass_low:
            raise ValueError("DenoiseConfig: 'bandpass_high' must be > 'bandpass_low'.")


@dataclass
class AtlasConfig:
    """Configuration for atlas handling."""

    atlas_name: str | None = None
    atlas_path: str | None = None
    roi_labels: list[str] | None = None


@dataclass
class ConnectivityConfig:
    """Configuration for connectivity computation."""

    method: str = "pearson_correlation"
    window_size: float | None = None
    threshold: float | None = None
    tr: float | None = None
    model_order: int = 1
    step_size: float | None = None

    def __post_init__(self) -> None:
        if self.threshold is not None and self.threshold < 0:
            raise ValueError("ConnectivityConfig: 'threshold' must be >= 0.")
        if self.window_size is not None and self.window_size <= 0:
            raise ValueError("ConnectivityConfig: 'window_size' must be > 0 when set.")
        if self.tr is not None and self.tr <= 0:
            raise ValueError("ConnectivityConfig: 'tr' must be > 0 when set.")
        if self.model_order < 1:
            raise ValueError("ConnectivityConfig: 'model_order' must be >= 1.")
        if self.step_size is not None and self.step_size <= 0:
            raise ValueError("ConnectivityConfig: 'step_size' must be > 0 when set.")


@dataclass
class BRAPHINConfig:
    """
    Convenience aggregator for all pipeline configs. Not currently consumed by
    any pipeline stage; provided for user-side configuration management.
    """

    input_config: InputConfig = field(default_factory=InputConfig)
    preprocess_config: PreprocessConfig = field(default_factory=PreprocessConfig)
    denoise_config: DenoiseConfig = field(default_factory=DenoiseConfig)
    atlas_config: AtlasConfig = field(default_factory=AtlasConfig)
    connectivity_config: ConnectivityConfig = field(default_factory=ConnectivityConfig)
