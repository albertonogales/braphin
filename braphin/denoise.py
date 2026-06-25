import logging
from dataclasses import dataclass, field

import numpy as np

from .config import DenoiseConfig
from .exceptions import DenoisingError
from .preprocess import BRAPHINPreprocessBundle

logger = logging.getLogger(__name__)


@dataclass
class BRAPHINDenoiseBundle:
    """
    Data structure representing the output of the denoising phase.

    Main fields:
    - fmri_path: path to the original fMRI file
    - original_metadata: metadata inherited from the input
    - preprocess_metadata: metadata inherited from preprocessing
    - denoised_data: 4D volume after denoising
    - voxel_time_series: 2D denoised matrix (num_voxels, num_timepoints)
    - auxiliary_files: auxiliary files inherited from preprocessing
    - applied_steps: steps actually executed
    - pending_steps: steps requested in config but not executed
    - denoise_metadata: traceability information
    """

    fmri_path: str | None = None
    original_metadata: dict[str, object] | None = None
    preprocess_metadata: dict[str, object] | None = None
    denoised_data: np.ndarray | None = None
    voxel_time_series: np.ndarray | None = None
    auxiliary_files: dict[str, object] = field(default_factory=dict)
    applied_steps: list[str] = field(default_factory=list)
    pending_steps: list[str] = field(default_factory=list)
    denoise_metadata: dict[str, object] = field(default_factory=dict)


class DenoiseBRAPHINData:
    """
    Main denoising class for BRAPHIN.

    Implemented steps (in execution order, Lindquist et al. 2019):
    1. Outlier scrubbing (apply_scrubbing=True)
       Uses the outlier_mask from the preprocess bundle if available; otherwise
       recomputes DVARS from the denoised signal.
    2. Butterworth bandpass filtering (apply_bandpass=True; requires tr)
       When both bandpass AND confound regression are enabled, the confounds
       are bandpass-filtered with the same filter before OLS (Lindquist 2019).
    3. Confound regression via ordinary least squares (regress_confounds=True)
    """

    def __init__(
        self,
        preprocess_bundle: BRAPHINPreprocessBundle,
        config: DenoiseConfig | None = None,
    ):
        self.preprocess_bundle = preprocess_bundle
        self.config = config if config is not None else DenoiseConfig()

    def run(self) -> BRAPHINDenoiseBundle:
        self._validate_preprocess_bundle()

        voxel_time_series = np.array(
            self.preprocess_bundle.voxel_time_series,
            dtype=np.float32,
            copy=True,
        )

        applied_steps: list[str] = []
        pending_steps: list[str] = []
        confounds_name = None
        confounds_shape = None
        n_scrubbed = 0

        # -- 1. Scrubbing (must come first to avoid ringing) ------------------
        if self.config.apply_scrubbing:
            voxel_time_series, n_scrubbed = self._apply_scrubbing(voxel_time_series)
            applied_steps.append("scrubbing")

        # -- 2 & 3. Bandpass + confound regression ----------------------------
        # When both are enabled, confounds must be bandpass-filtered with the
        # same filter before OLS (Lindquist et al. 2019, NeuroImage).
        if self.config.regress_confounds:
            confounds_name, confounds_matrix = self._find_confounds_matrix(
                self.preprocess_bundle.auxiliary_files,
                voxel_time_series.shape[1],
            )
            if confounds_matrix is not None:
                if self.config.apply_bandpass:
                    # Filter signal and confounds with the same filter
                    voxel_time_series = self._apply_bandpass_filter(voxel_time_series)
                    applied_steps.append("bandpass_filtering")
                    confounds_matrix = self._apply_bandpass_filter_to_confounds(confounds_matrix)
                voxel_time_series = self._regress_confounds(voxel_time_series, confounds_matrix)
                applied_steps.append("confound_regression")
                confounds_shape = tuple(confounds_matrix.shape)
            else:
                pending_steps.append("confound_regression_requested_but_no_confounds_found")
                if self.config.apply_bandpass:
                    voxel_time_series = self._apply_bandpass_filter(voxel_time_series)
                    applied_steps.append("bandpass_filtering")
        elif self.config.apply_bandpass:
            # Bandpass only (no regression)
            voxel_time_series = self._apply_bandpass_filter(voxel_time_series)
            applied_steps.append("bandpass_filtering")

        denoised_data = self._reshape_to_4d(voxel_time_series)

        return BRAPHINDenoiseBundle(
            fmri_path=self.preprocess_bundle.fmri_path,
            original_metadata=self.preprocess_bundle.original_metadata,
            preprocess_metadata=self.preprocess_bundle.preprocess_metadata,
            denoised_data=denoised_data,
            voxel_time_series=voxel_time_series,
            auxiliary_files=dict(self.preprocess_bundle.auxiliary_files),
            applied_steps=applied_steps,
            pending_steps=pending_steps,
            denoise_metadata=self._build_denoise_metadata(
                voxel_time_series=voxel_time_series,
                confounds_name=confounds_name,
                confounds_shape=confounds_shape,
                n_scrubbed=n_scrubbed,
            ),
        )

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def _validate_preprocess_bundle(self) -> None:
        if self.preprocess_bundle is None:
            raise DenoisingError("No valid BRAPHINPreprocessBundle was provided.")
        if self.preprocess_bundle.preprocessed_data is None:
            raise DenoisingError("The preprocess bundle does not contain 4D preprocessed data.")
        if self.preprocess_bundle.voxel_time_series is None:
            raise DenoisingError("The preprocess bundle does not contain a voxel x time matrix.")
        if not isinstance(self.preprocess_bundle.voxel_time_series, np.ndarray):
            raise DenoisingError("The voxel x time representation is not a NumPy ndarray.")

    # -------------------------------------------------------------------------
    # Step 1 -- Confound regression
    # -------------------------------------------------------------------------

    def _find_confounds_matrix(
        self,
        auxiliary_files: dict[str, object],
        num_timepoints: int,
    ) -> tuple[str | None, np.ndarray | None]:
        """
        Search auxiliary_files for a confound matrix whose row count matches
        num_timepoints.
        Continue searching if this file is incompatible; do not abort on the
        first failure.

        Search order:
        1. Files whose name contains "confound" (standard BIDS pattern).
        2. If none found, fall back to files containing "motion", "regressors",
           "nuisance", or "timeseries" — a warning is emitted for these.
        """
        _STANDARD_PATTERN = "confound"
        _FALLBACK_PATTERNS = ("motion", "regressors", "nuisance", "timeseries")

        def _try_match(file_name, data, pattern, is_fallback):
            if pattern not in file_name.lower():
                return None, None
            if not isinstance(data, np.ndarray):
                return None, None
            c = np.asarray(data, dtype=np.float32)
            if c.ndim == 1 and c.shape[0] == num_timepoints:
                matrix = c.reshape(num_timepoints, 1)
            elif c.ndim == 2 and c.shape[0] == num_timepoints:
                matrix = c
            elif c.ndim == 2 and c.shape[1] == num_timepoints:
                matrix = c.T
            else:
                return None, None  # shape mismatch — skip
            if is_fallback:
                logger.warning(
                    "Confound file matched via fallback pattern '%s': %s",
                    pattern,
                    file_name,
                )
            return file_name, matrix

        # Pass 1: standard "confound" pattern
        for file_name, data in auxiliary_files.items():
            name, matrix = _try_match(file_name, data, _STANDARD_PATTERN, False)
            if name is not None:
                return name, matrix

        # Pass 2: fallback patterns
        for pattern in _FALLBACK_PATTERNS:
            for file_name, data in auxiliary_files.items():
                name, matrix = _try_match(file_name, data, pattern, True)
                if name is not None:
                    return name, matrix

        return None, None

    def _regress_confounds(
        self,
        voxel_time_series: np.ndarray,
        confounds_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Remove the linear effect of confounds via OLS residuals.
        voxel_time_series: (V, T) -- confounds_matrix: (T, K)
        """
        if voxel_time_series.ndim != 2:
            raise DenoisingError("The voxel x time signal must be a 2D matrix.")
        if confounds_matrix.ndim != 2:
            raise DenoisingError("The confounds matrix must be 2D.")

        T = voxel_time_series.shape[1]
        if confounds_matrix.shape[0] != T:
            raise DenoisingError(
                "The number of rows in the confounds matrix does not match the number of timepoints."
            )

        if np.any(np.isnan(confounds_matrix)):
            nan_cols = np.where(np.any(np.isnan(confounds_matrix), axis=0))[0]
            raise DenoisingError(
                f"Confound matrix contains NaN in columns {nan_cols.tolist()}. "
                "Check your confound file for missing values (e.g. missing first-volume FD)."
            )

        signal_tv = voxel_time_series.T  # (T, V)

        # Standardise confounds
        mu = np.mean(confounds_matrix, axis=0, keepdims=True)
        sd = np.std(confounds_matrix, axis=0, keepdims=True)
        sd[sd == 0] = 1.0
        C = (confounds_matrix - mu) / sd

        # Add intercept
        D = np.column_stack([np.ones((T, 1), dtype=np.float32), C])

        try:
            betas, _, _, _ = np.linalg.lstsq(D, signal_tv, rcond=None)
        except Exception as exc:
            raise DenoisingError("Failed to solve the confound regression.") from exc

        residuals = signal_tv - D @ betas
        return np.asarray(residuals.T, dtype=np.float32)

    # -------------------------------------------------------------------------
    # Step 2 -- Scrubbing
    # -------------------------------------------------------------------------

    def _apply_scrubbing(self, voxel_time_series: np.ndarray) -> tuple[np.ndarray, int]:
        """
        Interpolates (or removes) outlier volumes.

        The outlier mask is taken from the preprocess bundle's outlier_mask
        field if available (set by outlier_detection in PreprocessBRAPHINData).
        If not, DVARS is recomputed from the current signal.

        Returns (cleaned_vts, n_scrubbed).
        """
        T = voxel_time_series.shape[1]

        # Use precomputed mask if available
        outlier_mask = getattr(self.preprocess_bundle, "outlier_mask", None)

        if outlier_mask is None:
            # Recompute DVARS from the denoised signal.
            # Uses fixed IQR multiplier 1.5 for the denoising-stage DVARS fallback.
            # For a configurable threshold, supply a PreprocessBundle with
            # outlier_mask already computed.
            dvars = np.zeros(T, dtype=np.float64)
            diff = np.diff(voxel_time_series.astype(np.float64), axis=1)
            dvars[1:] = np.sqrt(np.mean(diff**2, axis=0))
            dvars_vals = dvars[dvars > 0]
            if len(dvars_vals) > 0:
                q75, q25 = np.percentile(dvars_vals, [75, 25])
                iqr = q75 - q25
                threshold = float(np.median(dvars_vals)) + 1.5 * iqr
            else:
                threshold = np.inf
            outlier_mask = dvars > threshold

        n_scrubbed = int(np.sum(outlier_mask))
        if n_scrubbed == 0:
            return voxel_time_series, 0

        cleaned = np.array(voxel_time_series, copy=True)
        for t_idx in np.where(outlier_mask)[0]:
            prev_ok = next(
                (t for t in range(int(t_idx) - 1, -1, -1) if not outlier_mask[t]),
                None,
            )
            next_ok = next(
                (t for t in range(int(t_idx) + 1, T) if not outlier_mask[t]),
                None,
            )
            if prev_ok is not None and next_ok is not None:
                alpha = (t_idx - prev_ok) / (next_ok - prev_ok)
                cleaned[:, t_idx] = (
                    (1.0 - alpha) * voxel_time_series[:, prev_ok]
                    + alpha * voxel_time_series[:, next_ok]
                ).astype(np.float32)
            elif prev_ok is not None:
                cleaned[:, t_idx] = voxel_time_series[:, prev_ok]
            elif next_ok is not None:
                cleaned[:, t_idx] = voxel_time_series[:, next_ok]

        return cleaned.astype(np.float32), n_scrubbed

    # -------------------------------------------------------------------------
    # Step 2 -- Temporal bandpass filter
    # -------------------------------------------------------------------------

    def _make_bandpass_sos(self, tr: float, low: float, high: float):
        """
        Build a 4th-order Butterworth bandpass SOS filter.
        Raises DenoisingError on invalid parameters.
        """
        from scipy.signal import butter

        fs = 1.0 / tr
        nyq = fs / 2.0

        if low <= 0.0:
            raise DenoisingError(f"bandpass_low must be > 0 Hz (got {low}).")
        if high >= nyq:
            raise DenoisingError(
                f"bandpass_high ({high} Hz) must be below the Nyquist frequency "
                f"({nyq:.4f} Hz) for TR={tr}s. Reduce bandpass_high or increase TR."
            )
        if low >= high:
            raise DenoisingError(
                f"bandpass_low ({low} Hz) must be less than bandpass_high ({high} Hz)."
            )

        return butter(4, [low, high], btype="bandpass", output="sos", fs=fs)

    def _apply_bandpass_filter_to_confounds(self, confounds_matrix: np.ndarray) -> np.ndarray:
        """
        Apply the same Butterworth bandpass filter used on the signal to each
        column of the confounds matrix (shape T x K, time axis=0).
        """
        from scipy.signal import sosfiltfilt

        if self.config.tr is None:
            raise DenoisingError(
                "Bandpass filtering of confounds requires 'tr' to be set in DenoiseConfig."
            )

        tr = float(self.config.tr)
        low = self.config.bandpass_low
        high = self.config.bandpass_high
        sos = self._make_bandpass_sos(tr, low, high)

        # confounds_matrix is (T, K); filter along axis=0 (time)
        filtered = sosfiltfilt(sos, confounds_matrix.astype(np.float64), axis=0)
        return np.asarray(filtered, dtype=np.float32)

    def _apply_bandpass_filter(self, voxel_time_series: np.ndarray) -> np.ndarray:
        """
        Applies a zero-phase 4th-order Butterworth bandpass filter to every
        voxel's time series.

        Requires config.tr (seconds).  Raises DenoisingError if None.

        Filter band: [config.bandpass_low, config.bandpass_high] Hz.
        Both bounds must be strictly below the Nyquist frequency (0.5 / tr).
        """
        from scipy.signal import sosfiltfilt

        if self.config.tr is None:
            raise DenoisingError(
                "Bandpass filtering requires 'tr' to be set in DenoiseConfig "
                "(e.g. DenoiseConfig(apply_bandpass=True, tr=2.0))."
            )

        tr = float(self.config.tr)
        low = self.config.bandpass_low
        high = self.config.bandpass_high

        T = voxel_time_series.shape[1]
        # Minimum length for sosfiltfilt with 4th-order filter:
        # padlen default ~= 3 * 2 * n_sections = 3 * 2 * 3 = 18
        # We need T > 2 * padlen
        min_t = 40
        if T < min_t:
            raise DenoisingError(
                f"Bandpass filtering requires at least {min_t} timepoints "
                f"(got {T}). Use a longer scan or disable apply_bandpass."
            )

        sos = self._make_bandpass_sos(tr, low, high)

        # Apply zero-phase filter along time axis (axis=1 of (V, T) matrix)
        filtered = sosfiltfilt(sos, voxel_time_series.astype(np.float64), axis=1)
        return np.asarray(filtered, dtype=np.float32)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _reshape_to_4d(self, voxel_time_series: np.ndarray) -> np.ndarray:
        assert self.preprocess_bundle.preprocessed_data is not None
        shape = self.preprocess_bundle.preprocessed_data.shape
        try:
            return voxel_time_series.reshape(shape).astype(np.float32)
        except Exception as exc:
            raise DenoisingError("Could not reconstruct the 4D volume after denoising.") from exc

    def _build_denoise_metadata(
        self,
        voxel_time_series: np.ndarray,
        confounds_name: str | None,
        confounds_shape: tuple[int, int] | None,
        n_scrubbed: int = 0,
    ) -> dict[str, object]:
        assert self.preprocess_bundle.preprocessed_data is not None
        return {
            "denoised_4d_shape": tuple(self.preprocess_bundle.preprocessed_data.shape),
            "denoised_voxel_time_series_shape": tuple(voxel_time_series.shape),
            "num_voxels": int(voxel_time_series.shape[0]),
            "num_timepoints": int(voxel_time_series.shape[1]),
            "confounds_file_used": confounds_name,
            "confounds_shape": confounds_shape,
            "regress_confounds_requested": self.config.regress_confounds,
            "scrubbing_requested": self.config.apply_scrubbing,
            "bandpass_requested": self.config.apply_bandpass,
            "n_volumes_scrubbed": int(n_scrubbed),
            "implemented_scope": [
                "confound_regression",
                "scrubbing",
                "bandpass_filtering",
            ],
        }

    def display_info(self, bundle: BRAPHINDenoiseBundle) -> None:
        logger.info("[BRAPHIN] Denoising completed")
        logger.info("Original fMRI: %s", bundle.fmri_path)
        if bundle.denoise_metadata:
            m = bundle.denoise_metadata
            logger.info("Denoised volume shape: %s", m["denoised_4d_shape"])
            logger.info("Confounds used:        %s", m["confounds_file_used"])
        if bundle.applied_steps:
            logger.info("Applied steps:")
            for step in bundle.applied_steps:
                logger.info("  [OK] %s", step)
        if bundle.pending_steps:
            logger.info("Pending steps:")
            for step in bundle.pending_steps:
                logger.info("  [PENDING] %s", step)
