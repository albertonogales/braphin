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
    apply_smoothing         → requires smoothing_fwhm (mm); uses voxel sizes
                              from NIfTI metadata when available.
    apply_voxel_zscore      → per-voxel temporal z-score normalisation. Disabled by
                              default. Do NOT enable when using AEC or AEC-orth
                              connectivity measures — z-scoring removes amplitude
                              information and will produce incorrect results.
    """
    apply_motion_correction: bool = False
    apply_slice_timing: bool = False   # Requires tr; set tr when enabling
    apply_outlier_detection: bool = False
    apply_voxel_zscore: bool = False
    apply_smoothing: bool = False

    # Smoothing parameter
    smoothing_fwhm: float = 6.0          # FWHM in mm

    # Slice-timing parameters
    tr: Optional[float] = None           # Repetition time in seconds (required for slice timing)
    slice_order: str = "sequential"      # "sequential" or "interleaved"
    slice_timing_ref_slice: int = 0      # Reference slice index (0 = first, -1 = middle)

    # Slice-axis parameter
    slice_axis: int = 2
    """Spatial axis that corresponds to the slice dimension.

    0 = X (sagittal acquisition), 1 = Y (coronal acquisition),
    2 = Z (axial acquisition, default).  Change this when the fMRI data were
    acquired in a non-axial orientation so that slice-timing correction
    iterates over the correct dimension.
    Must be one of {0, 1, 2}.
    """

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
        if self.slice_axis not in (0, 1, 2):
            raise ValueError(
                "PreprocessConfig: 'slice_axis' must be 0, 1, or 2."
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
        Connectivity measure to use.  The 15 supported canonical names are:
        ``pearson_correlation``, ``cross_correlation``,
        ``corr_cross_correlation``, ``partial_correlation``,
        ``aec``, ``aec_orth``, ``mutual_information``, ``sync_likelihood``,
        ``coherence``, ``imag_coherence``, ``lagged_coherence``,
        ``granger_causality``, ``transfer_entropy``, ``pdc``, ``psi``.
        Aliases are accepted (see ``braphin.tools.CONNECTIVITY_ALIASES``).
        For a programmatic list call ``braphin.tools.list_fmri_connectivity_measures()``.

        Note: EEG-only phase measures (``plv``, ``pli``, ``wpli``, ``dwpli``,
        ``ppc``) are **not** implemented in this fMRI pipeline and will raise
        ``ConnectivityError`` if used.
    threshold : float or None
        Absolute threshold applied after connectivity computation.
        Edges with |value| < threshold are zeroed. None = no thresholding.
    window_size : float or None
        None = static connectivity. A positive float (seconds) enables
        sliding-window dynamic functional connectivity (dFC).
        Requires ``tr`` to be set.
    step_size : float or None
        Step between successive windows in seconds. Defaults to
        ``window_size / 2`` (50 % overlap). Requires ``window_size`` to be set.
    tr : float or None
        Repetition time in seconds (= 1 / sample_rate). Required for
        ``coherence`` and ``imag_coherence``; ignored for all other methods.
    """
    method: str = "pearson_correlation"
    window_size: Optional[float] = None
    threshold: Optional[float] = None
    tr: Optional[float] = None
    model_order: int = 1
    step_size: Optional[float] = None

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
        if self.step_size is not None and self.step_size <= 0:
            raise ValueError(
                "ConnectivityConfig: 'step_size' must be > 0 when set."
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
