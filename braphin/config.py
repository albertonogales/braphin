from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class InputConfig:
    """
    Configuration for input data handling.

    Controls:
    - which file extensions are accepted,
    - whether auxiliary files are required,
    - whether pre-derived tabular inputs are allowed.
    """
    allowed_fmri_extensions: List[str] = field(default_factory=lambda: [".nii", ".nii.gz"])
    allowed_aux_extensions: List[str] = field(default_factory=lambda: [".json", ".tsv", ".csv", ".npy"])
    allow_tabular_inputs: bool = True


@dataclass
class PreprocessConfig:
    """
    Configuration for the preprocessing stage.

    All steps are implemented. Relevant parameters per step:

    apply_slice_timing      → requires tr (repetition time in seconds).
    apply_motion_correction → no additional parameters; uses volume 0 as reference.
    apply_outlier_detection → uses outlier_threshold_dvars (IQR multiplier)
                              and scrubbing_strategy ("interpolate" or "mark").
    apply_normalization     → per-voxel temporal z-score normalisation (mean and
                              std computed along the time axis for each voxel).
    apply_smoothing         → requires smoothing_fwhm (mm); uses voxel sizes
                              from NIfTI metadata when available.
    """
    apply_motion_correction: bool = False
    apply_slice_timing: bool = False   # Requires tr; set tr when enabling
    apply_outlier_detection: bool = False
    apply_normalization: bool = True
    apply_smoothing: bool = False

    # Smoothing parameter
    smoothing_fwhm: float = 6.0          # FWHM in mm

    # Slice-timing parameters
    tr: Optional[float] = None           # Repetition time in seconds (required for slice timing)
    slice_order: str = "sequential"      # "sequential" or "interleaved"
    slice_timing_ref_slice: int = 0      # Reference slice index (0 = first, -1 = middle)

    # Outlier detection parameters
    outlier_threshold_dvars: float = 1.5  # IQR multiplier for DVARS threshold
    scrubbing_strategy: str = "interpolate"  # "interpolate" or "mark"

    def __post_init__(self) -> None:
        if self.apply_slice_timing and self.tr is None:
            raise ValueError(
                "PreprocessConfig: 'tr' must be set when apply_slice_timing=True."
            )
        if self.apply_smoothing and self.smoothing_fwhm <= 0:
            raise ValueError(
                "PreprocessConfig: 'smoothing_fwhm' must be positive."
            )
        if self.outlier_threshold_dvars <= 0:
            raise ValueError(
                "PreprocessConfig: 'outlier_threshold_dvars' must be positive."
            )
        if self.scrubbing_strategy not in ("interpolate", "mark"):
            raise ValueError(
                "PreprocessConfig: 'scrubbing_strategy' must be 'interpolate' or 'mark'."
            )


@dataclass
class DenoiseConfig:
    """
    Configuration for the denoising stage.

    All steps are implemented:
    - regress_confounds: OLS regression of confounds (motion parameters, etc.).
    - apply_scrubbing: removes/interpolates volumes flagged as outliers by
      the preprocessing stage (outlier_mask in metadata).
    - apply_bandpass: Butterworth band-pass filter. Requires tr.
    """
    regress_confounds: bool = True
    apply_scrubbing: bool = False
    apply_bandpass: bool = False

    # Bandpass parameters
    tr: Optional[float] = None           # Repetition time in seconds (required for bandpass)
    bandpass_low: float = 0.008          # Low cut-off in Hz
    bandpass_high: float = 0.1           # High cut-off in Hz (Biswal et al. 1995: 0.008–0.1 Hz)

    def __post_init__(self) -> None:
        if self.apply_bandpass and self.tr is None:
            raise ValueError(
                "DenoiseConfig: 'tr' must be set when apply_bandpass=True."
            )
        if self.bandpass_low <= 0:
            raise ValueError(
                "DenoiseConfig: 'bandpass_low' must be > 0 Hz."
            )
        if self.bandpass_high <= self.bandpass_low:
            raise ValueError(
                "DenoiseConfig: 'bandpass_high' must be > 'bandpass_low'."
            )


@dataclass
class AtlasConfig:
    """
    Configuration for atlas handling.

    Controls:
    - which atlas to use,
    - whether it is specified by a supported name or a custom path,
    - whether manual ROI labels are provided.
    """
    atlas_name: Optional[str] = None
    atlas_path: Optional[str] = None
    roi_labels: Optional[List[str]] = None


@dataclass
class ConnectivityConfig:
    """
    Configuration for connectivity computation.

    Fields
    ------
    method : str
        Connectivity measure to use. Supported canonical names:
        ``pearson_correlation``, ``cross_correlation``,
        ``corr_cross_correlation``, ``partial_correlation``,
        ``plv``, ``pli``, ``wpli``, ``coherence``, ``imag_coherence``.
        Aliases are accepted (see ``braphin.tools.CONNECTIVITY_ALIASES``).
    threshold : float or None
        Absolute threshold applied after connectivity computation.
        Edges with |value| < threshold are zeroed. None = no thresholding.
    window_size : float or None
        None = static connectivity. A positive float (seconds) requests
        windowed dynamic connectivity, which is not yet implemented and is
        flagged as pending in the output bundle.
    tr : float or None
        Repetition time in seconds (= 1 / sample_rate). Required for
        ``coherence`` and ``imag_coherence``; ignored for all other methods.
    """
    method: str = "pearson_correlation"
    window_size: Optional[float] = None
    threshold: Optional[float] = None
    tr: Optional[float] = None
    model_order: int = 1

    def __post_init__(self) -> None:
        if self.threshold is not None and self.threshold < 0:
            raise ValueError(
                "ConnectivityConfig: 'threshold' must be >= 0."
            )
        if self.window_size is not None and self.window_size <= 0:
            raise ValueError(
                "ConnectivityConfig: 'window_size' must be > 0 when set."
            )
        if self.tr is not None and self.tr <= 0:
            raise ValueError(
                "ConnectivityConfig: 'tr' must be > 0 when set."
            )
        if self.model_order < 1:
            raise ValueError(
                "ConnectivityConfig: 'model_order' must be >= 1."
            )


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
