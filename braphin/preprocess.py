import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from .config import PreprocessConfig
from .exceptions import PreprocessingError
from .importBRAPHINData import BRAPHINInputBundle

logger = logging.getLogger(__name__)


@dataclass
class BRAPHINPreprocessBundle:
    """
    Data structure representing the output of the preprocessing phase.

    Fields:
    - fmri_path: path to the original fMRI file
    - original_metadata: basic metadata obtained during loading
    - preprocessed_data: preprocessed 4-D array
    - voxel_time_series: 2-D representation with shape (num_voxels, num_timepoints)
    - auxiliary_files: auxiliary files inherited from the input bundle
    - applied_steps: steps that were actually executed
    - pending_steps: Always empty. Retained for backward compatibility; all
                     preprocessing steps are fully implemented.
    - preprocess_metadata: information useful for debugging and traceability
    - motion_params: array (T, 6) with estimated motion parameters,
                     or None if motion correction was not applied
    - outlier_mask: boolean array (T,) marking outlier volumes,
                    or None if outlier detection was not applied
    """
    fmri_path: Optional[str] = None
    original_metadata: Optional[Dict[str, object]] = None
    preprocessed_data: Optional[np.ndarray] = None
    voxel_time_series: Optional[np.ndarray] = None
    auxiliary_files: Dict[str, object] = field(default_factory=dict)
    applied_steps: List[str] = field(default_factory=list)
    pending_steps: List[str] = field(default_factory=list)
    preprocess_metadata: Dict[str, object] = field(default_factory=dict)
    motion_params: Optional[np.ndarray] = None
    outlier_mask: Optional[np.ndarray] = None


class PreprocessBRAPHINData:
    """
    Main preprocessing class for BRAPHIN.

    Implemented steps (in execution order):
    1. NaN/inf cleanup (always active)
    2. Slice-timing correction (apply_slice_timing=True; requires tr)
    3. Motion correction (apply_motion_correction=True)
    4. Outlier detection / scrubbing (apply_outlier_detection=True)
    5. Per-voxel temporal normalisation (apply_normalization=True)
    6. Gaussian spatial smoothing (apply_smoothing=True)
    """

    def __init__(
        self,
        input_bundle: BRAPHINInputBundle,
        config: Optional[PreprocessConfig] = None,
    ):
        self.input_bundle = input_bundle
        self.config = config if config is not None else PreprocessConfig()

    def run(self) -> BRAPHINPreprocessBundle:
        """
        Execute the preprocessing phase.

        Returns:
        - BRAPHINPreprocessBundle with preprocessed 4-D array, voxel-by-time matrix,
          metadata, applied steps, and motion/outlier parameters.
        """
        self._validate_input_bundle()

        fmri_data = self._extract_fmri_array()
        self._validate_fmri_array(fmri_data)

        applied_steps: List[str] = []
        pending_steps: List[str] = []
        motion_params: Optional[np.ndarray] = None
        outlier_mask: Optional[np.ndarray] = None

        # -- 1. Clean non-finite values (always) --------------------------------
        fmri_data, replaced_values = self._replace_non_finite_values(fmri_data)
        applied_steps.append("replace_non_finite_values")

        # -- 2. Slice-timing correction -----------------------------------------
        if self.config.apply_slice_timing:
            fmri_data = self._apply_slice_timing_correction(fmri_data)
            applied_steps.append("slice_timing_correction")

        # -- 3. Motion correction -----------------------------------------------
        if self.config.apply_motion_correction:
            fmri_data, motion_params = self._apply_motion_correction(fmri_data)
            applied_steps.append("motion_correction")

        # -- 4. Outlier detection / scrubbing ------------------------------------
        if self.config.apply_outlier_detection:
            fmri_data, outlier_mask = self._apply_outlier_detection(
                fmri_data, motion_params
            )
            applied_steps.append("outlier_detection")

        # -- 5. Per-voxel temporal normalisation --------------------------------
        if self.config.apply_normalization:
            fmri_data = self._normalize_data(fmri_data)
            applied_steps.append("per_voxel_temporal_normalisation")

        # -- 6. Spatial smoothing -----------------------------------------------
        if self.config.apply_smoothing:
            fmri_data = self._apply_spatial_smoothing(fmri_data)
            applied_steps.append("spatial_smoothing")

        voxel_time_series = self._reshape_to_voxel_time_series(fmri_data)

        preprocess_metadata = self._build_preprocess_metadata(
            fmri_data=fmri_data,
            voxel_time_series=voxel_time_series,
            replaced_values=replaced_values,
            motion_params=motion_params,
            outlier_mask=outlier_mask,
        )

        return BRAPHINPreprocessBundle(
            fmri_path=self.input_bundle.fmri_path,
            original_metadata=self.input_bundle.fmri_metadata,
            preprocessed_data=fmri_data,
            voxel_time_series=voxel_time_series,
            auxiliary_files=dict(self.input_bundle.auxiliary_files),
            applied_steps=applied_steps,
            pending_steps=pending_steps,
            preprocess_metadata=preprocess_metadata,
            motion_params=motion_params,
            outlier_mask=outlier_mask,
        )

    # --------------------------------------------------------------------------
    # Validation
    # --------------------------------------------------------------------------

    def _validate_input_bundle(self) -> None:
        if self.input_bundle is None:
            raise PreprocessingError("No valid BRAPHINInputBundle was provided.")
        if self.input_bundle.fmri_image is None:
            raise PreprocessingError(
                "The input bundle does not contain a loaded fMRI image."
            )
        if self.input_bundle.fmri_metadata is None:
            raise PreprocessingError(
                "The input bundle does not contain fMRI metadata."
            )

    def _extract_fmri_array(self) -> np.ndarray:
        try:
            return self.input_bundle.fmri_image.get_fdata(dtype=np.float32)
        except Exception as exc:
            raise PreprocessingError(
                "Could not extract the numerical array from the fMRI image."
            ) from exc

    def _validate_fmri_array(self, fmri_data: np.ndarray) -> None:
        if not isinstance(fmri_data, np.ndarray):
            raise PreprocessingError("The extracted data is not a NumPy ndarray.")
        if fmri_data.ndim != 4:
            raise PreprocessingError(
                f"Expected a 4-D array for fMRI, but got shape {fmri_data.shape}."
            )
        if fmri_data.shape[-1] < 2:
            raise PreprocessingError(
                "The number of timepoints is insufficient to continue with the pipeline."
            )

    # --------------------------------------------------------------------------
    # Step 1 -- NaN / inf cleanup
    # --------------------------------------------------------------------------

    def _replace_non_finite_values(self, fmri_data: np.ndarray) -> Tuple[np.ndarray, int]:
        """Replace NaN, +inf, -inf with 0.0. Returns (cleaned_array, n_replaced)."""
        mask = ~np.isfinite(fmri_data)
        n = int(np.sum(mask))
        if n == 0:
            return fmri_data, 0
        cleaned = np.array(fmri_data, copy=True)
        cleaned[mask] = 0.0
        return cleaned, n

    # --------------------------------------------------------------------------
    # Step 2 -- Slice-timing correction
    # --------------------------------------------------------------------------

    def _apply_slice_timing_correction(self, fmri_data: np.ndarray) -> np.ndarray:
        """
        Corrects for the staggered acquisition of slices within each TR using
        linear interpolation.  Every slice's time series is shifted to the
        reference slice acquisition time.

        Requires:
        - config.tr  (float, seconds)  -- raises PreprocessingError if None.

        Config fields used:
        - slice_order: "sequential" (default) or "interleaved"
        - slice_timing_ref_slice: reference slice index (0 = first, -1 = middle)
        """
        if self.config.tr is None:
            raise PreprocessingError(
                "Slice-timing correction requires 'tr' to be set in PreprocessConfig "
                "(e.g. PreprocessConfig(apply_slice_timing=True, tr=2.0))."
            )

        tr = float(self.config.tr)
        X, Y, Z, T = fmri_data.shape
        n_slices = Z

        # Build per-slice acquisition times (seconds within one TR)
        if self.config.slice_order == "sequential":
            slice_times = np.arange(n_slices, dtype=float) * (tr / n_slices)
        elif self.config.slice_order == "interleaved":
            # Odd indices first, then even (standard Siemens interleaved)
            order = list(range(0, n_slices, 2)) + list(range(1, n_slices, 2))
            rank = np.empty(n_slices, dtype=int)
            for pos, slc in enumerate(order):
                rank[slc] = pos
            slice_times = rank.astype(float) * (tr / n_slices)
        else:
            raise PreprocessingError(
                f"Unknown slice_order '{self.config.slice_order}'. "
                "Use 'sequential' or 'interleaved'."
            )

        # Reference time
        ref_idx = self.config.slice_timing_ref_slice
        if ref_idx == -1:
            ref_idx = n_slices // 2
        ref_time = slice_times[ref_idx]

        # Original time axis
        t_orig = np.arange(T, dtype=float) * tr  # [0, TR, 2TR, ...]

        corrected = np.empty_like(fmri_data)

        for z in range(n_slices):
            shift = ref_time - slice_times[z]     # positive -> shift forward in time
            t_query = np.clip(t_orig + shift, 0.0, (T - 1) * tr)

            # Fractional sample indices for linear interpolation
            frac = t_query / tr                   # range [0, T-1]
            lo = np.floor(frac).astype(int)
            hi = np.minimum(lo + 1, T - 1)
            alpha = frac - lo                     # interpolation weight toward hi

            # Vectorised across X x Y voxels in this slice
            slice_data = fmri_data[:, :, z, :]   # (X, Y, T)
            corrected[:, :, z, :] = (
                (1.0 - alpha) * slice_data[:, :, lo] +
                alpha * slice_data[:, :, hi]
            )

        return corrected.astype(np.float32)

    # --------------------------------------------------------------------------
    # Step 3 -- Motion correction (rigid-body realignment)
    # --------------------------------------------------------------------------

    def _apply_motion_correction(
        self, fmri_data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Realigns each volume to the first volume (volume 0) using 6-parameter
        rigid-body registration (3 translations + 3 rotations).

        Algorithm:
        - For each volume t >= 1, scipy.optimize.minimize (Powell's method) finds
          the parameters [tx, ty, tz, rx, ry, rz] that minimise the sum of
          squared voxel-wise differences between the transformed volume and the
          reference.
        - The affine transform is applied with scipy.ndimage.affine_transform
          (linear interpolation, rotation about the volume centre).

        Returns:
        - corrected: float32 ndarray (X, Y, Z, T)
        - motion_params: float64 ndarray (T, 6)
          columns: [tx_vox, ty_vox, tz_vox, rx_rad, ry_rad, rz_rad]

        Note on convention:
        - scipy.ndimage.affine_transform expects the matrix to map output->input
          coordinates (i.e. the inverse of the physical forward transform).
        - The optimiser therefore converges to the *inverse* rigid-body parameters.
        - As a result, motion_params reflect the transformation applied to bring
          each volume into reference space, not the physical head displacement.
          They should not be interpreted directly as head motion for
          quality-control purposes without sign-reversal.
        """
        from scipy.ndimage import affine_transform
        from scipy.optimize import minimize

        X, Y, Z, T = fmri_data.shape
        center = np.array([X / 2.0, Y / 2.0, Z / 2.0])
        reference = fmri_data[..., 0].astype(np.float64)

        motion_params = np.zeros((T, 6), dtype=np.float64)
        corrected = np.array(fmri_data, copy=True, dtype=np.float32)

        def _rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
            """Compose Rz @ Ry @ Rx rotation matrix."""
            cx, sx = np.cos(rx), np.sin(rx)
            cy, sy = np.cos(ry), np.sin(ry)
            cz, sz = np.cos(rz), np.sin(rz)
            Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
            Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
            return Rz @ Ry @ Rx

        def _cost(params: np.ndarray, moving: np.ndarray) -> float:
            tx, ty, tz, rx, ry, rz = params
            R = _rotation_matrix(rx, ry, rz)
            # Rotate around centre then translate
            offset = center - R @ center + np.array([tx, ty, tz])
            # NOTE: affine_transform maps output->input coords (inverse transform);
            # see docstring for the implication on motion_params interpretation.
            transformed = affine_transform(
                moving, R, offset=offset,
                order=1, mode="constant", cval=0.0,
            )
            diff = transformed - reference
            return float(np.sum(diff * diff))

        for t in range(1, T):
            moving = fmri_data[..., t].astype(np.float64)
            result = minimize(
                _cost,
                x0=np.zeros(6),
                args=(moving,),
                method="Powell",
                options={"maxiter": 200, "ftol": 1e-5, "xtol": 1e-5},
            )
            params = result.x
            motion_params[t] = params

            tx, ty, tz, rx, ry, rz = params
            R = _rotation_matrix(rx, ry, rz)
            offset = center - R @ center + np.array([tx, ty, tz])
            corrected[..., t] = affine_transform(
                moving, R, offset=offset,
                order=1, mode="constant", cval=0.0,
            ).astype(np.float32)

        return corrected, motion_params

    # --------------------------------------------------------------------------
    # Step 4 -- Outlier detection + scrubbing
    # --------------------------------------------------------------------------

    def _apply_outlier_detection(
        self,
        fmri_data: np.ndarray,
        motion_params: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detects outlier volumes using DVARS and optionally FD, then handles
        them according to config.scrubbing_strategy.

        DVARS_t = sqrt( mean( (Y_t - Y_{t-1})^2 ) )  for t >= 1; DVARS_0 = 0.

        Outlier threshold = median(DVARS) + config.outlier_threshold_dvars x IQR(DVARS).

        FD is computed when motion_params is available:
        FD_t = sum |delta_p_t|  (translations in voxels, rotations in rad x 50mm).

        Scrubbing strategies:
        - "interpolate": replace outlier volumes with linear interpolation from
          the nearest clean volumes on each side.
        - "mark": record outliers in the mask but do not modify the data.

        Returns:
        - cleaned: float32 ndarray (X, Y, Z, T)
        - outlier_mask: bool ndarray (T,)
        """
        X, Y, Z, T = fmri_data.shape
        flat = fmri_data.reshape(-1, T).astype(np.float64)  # (V, T)

        # DVARS
        dvars = np.zeros(T, dtype=np.float64)
        diff = np.diff(flat, axis=1)                        # (V, T-1)
        dvars[1:] = np.sqrt(np.mean(diff ** 2, axis=0))

        # FD (optional)
        fd = np.zeros(T, dtype=np.float64)
        if motion_params is not None:
            mp = motion_params.copy()
            mp[:, 3:] *= 50.0      # rotations (rad) -> mm (50 mm brain radius)
            fd[1:] = np.sum(np.abs(np.diff(mp, axis=0)), axis=1)

        # Outlier threshold via IQR
        dvars_vals = dvars[dvars > 0]
        if len(dvars_vals) > 0:
            q75, q25 = np.percentile(dvars_vals, [75, 25])
            iqr = q75 - q25
            threshold = float(np.median(dvars_vals)) + self.config.outlier_threshold_dvars * iqr
        else:
            threshold = np.inf

        outlier_mask = dvars > threshold

        cleaned = np.array(fmri_data, copy=True)

        if self.config.scrubbing_strategy == "interpolate" and np.any(outlier_mask):
            for t_idx in np.where(outlier_mask)[0]:
                prev_ok = next(
                    (t for t in range(t_idx - 1, -1, -1) if not outlier_mask[t]),
                    None,
                )
                next_ok = next(
                    (t for t in range(t_idx + 1, T) if not outlier_mask[t]),
                    None,
                )
                if prev_ok is not None and next_ok is not None:
                    alpha = (t_idx - prev_ok) / (next_ok - prev_ok)
                    cleaned[..., t_idx] = (
                        (1.0 - alpha) * fmri_data[..., prev_ok] +
                        alpha * fmri_data[..., next_ok]
                    ).astype(np.float32)
                elif prev_ok is not None:
                    cleaned[..., t_idx] = fmri_data[..., prev_ok]
                elif next_ok is not None:
                    cleaned[..., t_idx] = fmri_data[..., next_ok]
                # else: all volumes flagged -- nothing to do

        return cleaned.astype(np.float32), outlier_mask

    # --------------------------------------------------------------------------
    # Step 5 -- Per-voxel temporal normalisation
    # --------------------------------------------------------------------------

    def _normalize_data(self, fmri_data: np.ndarray) -> np.ndarray:
        """
        Per-voxel temporal z-score normalisation.

        For each voxel (x, y, z), subtracts its mean across time and divides by
        its standard deviation across time.  Voxels with std == 0 (constant
        signal) are left unchanged (effectively dividing by 1).

        This is the standard normalisation for fMRI time-series analysis and is
        preferable to global z-scoring, which conflates spatial and temporal
        variance.
        """
        mean = fmri_data.mean(axis=3, keepdims=True)
        std = fmri_data.std(axis=3, keepdims=True)
        # Avoid division by zero for constant-signal voxels
        std[std < 1e-10] = 1.0
        return ((fmri_data - mean) / std).astype(np.float32)

    # --------------------------------------------------------------------------
    # Step 6 -- Spatial smoothing
    # --------------------------------------------------------------------------

    def _apply_spatial_smoothing(self, fmri_data: np.ndarray) -> np.ndarray:
        """
        Applies an isotropic Gaussian smoothing kernel to each 3-D volume.

        The FWHM (config.smoothing_fwhm, in mm) is converted to sigma in voxels:
            sigma_vox[i] = (FWHM / (2 sqrt(2 ln 2))) / voxel_size_mm[i]

        Voxel sizes are read from the NIfTI header zooms.  If unavailable,
        2 mm isotropic is assumed.
        """
        from scipy.ndimage import gaussian_filter

        fwhm = float(self.config.smoothing_fwhm)
        if fwhm <= 0.0:
            raise PreprocessingError(
                f"smoothing_fwhm must be positive (got {fwhm})."
            )

        # Voxel sizes from NIfTI header
        zooms = None
        if self.input_bundle.fmri_metadata is not None:
            zooms = self.input_bundle.fmri_metadata.get("zooms")
        if zooms is not None and len(zooms) >= 3:
            voxel_sizes = np.array(zooms[:3], dtype=float)
        else:
            voxel_sizes = np.array([2.0, 2.0, 2.0])   # assume 2 mm isotropic

        # FWHM -> sigma: sigma = FWHM / (2 sqrt(2 ln 2)) ~ FWHM / 2.3548
        fwhm_to_sigma = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        sigma_vox = (fwhm * fwhm_to_sigma) / voxel_sizes   # per axis

        T = fmri_data.shape[-1]
        smoothed = np.empty_like(fmri_data)
        for t in range(T):
            smoothed[..., t] = gaussian_filter(
                fmri_data[..., t].astype(np.float64),
                sigma=sigma_vox,
            )

        return smoothed.astype(np.float32)

    # --------------------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------------------

    def _reshape_to_voxel_time_series(self, fmri_data: np.ndarray) -> np.ndarray:
        """Reshape (X, Y, Z, T) -> (X*Y*Z, T)."""
        return fmri_data.reshape(-1, fmri_data.shape[-1])

    def _build_preprocess_metadata(
        self,
        fmri_data: np.ndarray,
        voxel_time_series: np.ndarray,
        replaced_values: int,
        motion_params: Optional[np.ndarray],
        outlier_mask: Optional[np.ndarray],
    ) -> Dict[str, object]:
        original_shape = self.input_bundle.fmri_metadata.get("shape")
        n_outliers = int(np.sum(outlier_mask)) if outlier_mask is not None else 0

        return {
            "original_shape": original_shape,
            "processed_shape": tuple(fmri_data.shape),
            "ndim": fmri_data.ndim,
            "dtype": str(fmri_data.dtype),
            "num_voxels": int(np.prod(fmri_data.shape[:3])),
            "num_timepoints": int(fmri_data.shape[3]),
            "voxel_time_series_shape": tuple(voxel_time_series.shape),
            "non_finite_values_replaced": replaced_values,
            "normalization_applied": self.config.apply_normalization,
            "motion_correction_applied": self.config.apply_motion_correction,
            "slice_timing_applied": self.config.apply_slice_timing,
            "outlier_detection_applied": self.config.apply_outlier_detection,
            "smoothing_applied": self.config.apply_smoothing,
            "n_outlier_volumes": n_outliers,
            "motion_params_shape": tuple(motion_params.shape) if motion_params is not None else None,
            "implemented_scope": [
                "replace_non_finite_values",
                "slice_timing_correction",
                "motion_correction",
                "outlier_detection_scrubbing",
                "per_voxel_temporal_normalisation",
                "spatial_smoothing",
            ],
        }

    def display_info(self, bundle: BRAPHINPreprocessBundle) -> None:
        """Log a human-readable summary of the preprocessing result."""
        logger.info("[BRAPHIN] Preprocessing complete")
        logger.info("Original fMRI: %s", bundle.fmri_path)
        if bundle.preprocess_metadata:
            m = bundle.preprocess_metadata
            logger.info("Original shape:    %s", m["original_shape"])
            logger.info("Processed shape:   %s", m["processed_shape"])
            logger.info("Voxel x time:      %s", m["voxel_time_series_shape"])
            logger.info("Non-finite values replaced: %s", m["non_finite_values_replaced"])
            logger.info("Outlier volumes detected: %s", m["n_outlier_volumes"])
        if bundle.applied_steps:
            logger.info("Applied steps:")
            for step in bundle.applied_steps:
                logger.info("  [done] %s", step)
        if bundle.pending_steps:
            logger.info("Pending steps:")
            for step in bundle.pending_steps:
                logger.info("  [pending] %s", step)
