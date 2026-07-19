"""
Stage 5 of the BRAPHIN pipeline: functional connectivity modelling.

Computes the ROI × ROI connectivity matrix from the parcellated time series
produced by :class:`~braphin.transform.TransformBRAPHINData`.
"""

import logging
from dataclasses import dataclass, field

import numpy as np

from .config import ConnectivityConfig
from .exceptions import ConnectivityError
from .strategy import get_connectivity_strategy
from .tools import apply_connectivity_threshold
from .transform import BRAPHINTransformBundle

logger = logging.getLogger(__name__)


@dataclass
class BRAPHINConnectivityBundle:
    """Output bundle of the connectivity modelling stage."""

    fmri_path: str | None = None
    original_metadata: dict[str, object] | None = None
    preprocess_metadata: dict[str, object] | None = None
    denoise_metadata: dict[str, object] | None = None
    transform_metadata: dict[str, object] | None = None
    atlas_name: str | None = None
    roi_labels: list[str] = field(default_factory=list)
    roi_time_series: np.ndarray | None = None
    connectivity_matrix: np.ndarray | None = None
    dynamic_connectivity_matrices: np.ndarray | None = None
    window_centers_sec: list[float] | None = None
    applied_steps: list[str] = field(default_factory=list)
    pending_steps: list[str] = field(default_factory=list)
    connectivity_metadata: dict[str, object] = field(default_factory=dict)


def _compute_sliding_window_dfc(
    roi_time_series: np.ndarray,
    strategy,
    window_size: float,
    tr: float,
    step_size: float | None = None,
):
    """Compute sliding-window dynamic functional connectivity (dFC)."""
    N, T = roi_time_series.shape
    window_samples = int(round(window_size / tr))
    if step_size is None:
        step_size = window_size / 2.0
    step_samples = max(1, int(round(step_size / tr)))

    if window_samples < 2:
        raise ConnectivityError(
            f"Window size {window_size}s corresponds to {window_samples} sample(s) "
            f"at TR={tr}s. At least 2 samples per window are required."
        )
    if window_samples >= T:
        raise ConnectivityError(
            f"Window size {window_size}s ({window_samples} samples) >= total "
            f"time series length {T} samples (duration={T * tr:.1f}s). "
            "Reduce window_size or use a longer scan."
        )

    starts = list(range(0, T - window_samples + 1, step_samples))
    if not starts:
        raise ConnectivityError(
            "No windows fit in the time series with the given window_size and step_size."
        )

    dynamic_matrices = []
    window_centers: list[float] = []

    for start in starts:
        end = start + window_samples
        window_ts = roi_time_series[:, start:end]
        mat = strategy.compute(window_ts)
        dynamic_matrices.append(mat)
        window_centers.append((start + end) / 2.0 * tr)

    logger.info(
        "[BRAPHIN] Sliding-window dFC: %d windows, window=%.1fs, step=%.1fs",
        len(dynamic_matrices),
        window_size,
        step_size,
    )

    return np.stack(dynamic_matrices, axis=0), window_centers


class ModelBRAPHINConnectivityData:
    """Stage 5 of the BRAPHIN pipeline: functional connectivity modelling."""

    def __init__(
        self,
        transform_bundle: BRAPHINTransformBundle,
        config: ConnectivityConfig | None = None,
    ):
        self.transform_bundle = transform_bundle
        self.config = config if config is not None else ConnectivityConfig()

    def run(self) -> BRAPHINConnectivityBundle:
        """Execute the connectivity modelling stage."""
        self._validate_transform_bundle()

        roi_time_series = np.array(
            self.transform_bundle.roi_time_series,
            dtype=np.float32,
            copy=True,
        )

        # ── 1. Select connectivity strategy ──────────────────────────────────
        strategy = get_connectivity_strategy(
            self.config.method,
            tr=self.config.tr,
            model_order=self.config.model_order,
        )

        applied_steps = [self.config.method]
        dynamic_connectivity_matrices = None
        window_centers_sec = None

        # ── 2. Compute connectivity ───────────────────────────────────────────
        # When window_size is set: compute per-window matrices first (matching
        # EEG behaviour), then derive the static summary as their mean.
        # When window_size is None: compute one matrix on the whole signal.
        if self.config.window_size is not None:
            if self.config.tr is None or self.config.tr <= 0:
                raise ConnectivityError(
                    "ConnectivityConfig.tr must be set (> 0) for sliding-window "
                    "dynamic connectivity."
                )
            dynamic_connectivity_matrices, window_centers_sec = _compute_sliding_window_dfc(
                roi_time_series=roi_time_series,
                strategy=strategy,
                window_size=self.config.window_size,
                tr=self.config.tr,
                step_size=self.config.step_size,
            )
            applied_steps.append("windowed_dynamic_connectivity")
            # Static summary = mean over all windows
            connectivity_matrix = np.mean(dynamic_connectivity_matrices, axis=0).astype(np.float32)
            applied_steps.append("mean_over_windows")
        else:
            connectivity_matrix = strategy.compute(roi_time_series)

        # ── 3. Optional absolute threshold ───────────────────────────────────
        if self.config.threshold is not None:
            connectivity_matrix = apply_connectivity_threshold(
                connectivity_matrix,
                self.config.threshold,
            )
            applied_steps.append("threshold")

        connectivity_metadata = self._build_connectivity_metadata(
            roi_time_series=roi_time_series,
            connectivity_matrix=connectivity_matrix,
            dynamic_connectivity_matrices=dynamic_connectivity_matrices,
        )

        bundle = BRAPHINConnectivityBundle(
            fmri_path=self.transform_bundle.fmri_path,
            original_metadata=self.transform_bundle.original_metadata,
            preprocess_metadata=self.transform_bundle.preprocess_metadata,
            denoise_metadata=self.transform_bundle.denoise_metadata,
            transform_metadata=self.transform_bundle.transform_metadata,
            atlas_name=self.transform_bundle.atlas_name,
            roi_labels=list(self.transform_bundle.roi_labels),
            roi_time_series=roi_time_series,
            connectivity_matrix=connectivity_matrix,
            dynamic_connectivity_matrices=dynamic_connectivity_matrices,
            window_centers_sec=window_centers_sec,
            applied_steps=applied_steps,
            pending_steps=[],
            connectivity_metadata=connectivity_metadata,
        )

        return bundle

    # ── Validation ─────────────────────────────────────────────────────────────

    def _validate_transform_bundle(self) -> None:
        """Verify that the transform bundle contains the required data."""
        if self.transform_bundle is None:
            raise ConnectivityError("A valid BRAPHINTransformBundle must be provided.")

        if self.transform_bundle.roi_time_series is None:
            raise ConnectivityError("The transform bundle does not contain roi_time_series.")

        if not isinstance(self.transform_bundle.roi_time_series, np.ndarray):
            raise ConnectivityError("roi_time_series must be a NumPy ndarray.")

        if self.transform_bundle.roi_time_series.ndim != 2:
            raise ConnectivityError(
                f"Expected a 2-D ROI x time matrix, but received shape "
                f"{self.transform_bundle.roi_time_series.shape}."
            )

    # ── Metadata ───────────────────────────────────────────────────────────────

    def _build_connectivity_metadata(
        self,
        roi_time_series: np.ndarray,
        connectivity_matrix: np.ndarray,
        dynamic_connectivity_matrices: np.ndarray | None = None,
    ) -> dict[str, object]:
        """Build traceability metadata for the connectivity stage."""
        off_diagonal_mask = ~np.eye(connectivity_matrix.shape[0], dtype=bool)
        off_diagonal_values = connectivity_matrix[off_diagonal_mask]

        return {
            "method": self.config.method,
            "threshold": self.config.threshold,
            "window_size": self.config.window_size,
            "step_size": self.config.step_size,
            "n_windows": (
                int(dynamic_connectivity_matrices.shape[0])
                if dynamic_connectivity_matrices is not None
                else None
            ),
            "tr": self.config.tr,
            "num_rois": int(roi_time_series.shape[0]),
            "num_timepoints": int(roi_time_series.shape[1]),
            "roi_time_series_shape": tuple(roi_time_series.shape),
            "connectivity_matrix_shape": tuple(connectivity_matrix.shape),
            "matrix_dtype": str(connectivity_matrix.dtype),
            "matrix_is_symmetric": bool(
                np.allclose(connectivity_matrix, connectivity_matrix.T, atol=1e-5)
            ),
            "diagonal_all_ones": bool(np.allclose(np.diag(connectivity_matrix), 1.0, atol=1e-5)),
            "min_connectivity": (
                float(np.min(off_diagonal_values)) if off_diagonal_values.size > 0 else 0.0
            ),
            "max_connectivity": (
                float(np.max(off_diagonal_values)) if off_diagonal_values.size > 0 else 0.0
            ),
            "mean_connectivity": (
                float(np.mean(off_diagonal_values)) if off_diagonal_values.size > 0 else 0.0
            ),
            "implemented_scope": [
                "static_connectivity",
                self.config.method,
                "optional_thresholding",
            ],
        }

    # ── Display ────────────────────────────────────────────────────────────────

    def display_info(self, bundle: BRAPHINConnectivityBundle) -> None:
        """Log a summary of the connectivity computation result."""
        logger.info("[BRAPHIN] Connectivity computed")
        logger.info("  fMRI path:            %s", bundle.fmri_path)
        logger.info("  Method:               %s", bundle.connectivity_metadata.get("method"))
        logger.info(
            "  ROI x time shape:     %s", bundle.connectivity_metadata.get("roi_time_series_shape")
        )
        logger.info(
            "  Connectivity shape:   %s",
            bundle.connectivity_metadata.get("connectivity_matrix_shape"),
        )
        logger.info(
            "  Symmetric:            %s",
            bundle.connectivity_metadata.get("matrix_is_symmetric"),
        )
        logger.info(
            "  Diagonal = 1:         %s",
            bundle.connectivity_metadata.get("diagonal_all_ones"),
        )
        logger.info(
            "  Mean connectivity:    %.4f",
            bundle.connectivity_metadata.get("mean_connectivity", 0.0),
        )

        if bundle.applied_steps:
            logger.info("  Applied steps:")
            for step in bundle.applied_steps:
                logger.info("    - %s", step)

        if bundle.pending_steps:
            logger.info("  Pending steps (not yet implemented):")
            for step in bundle.pending_steps:
                logger.info("    - %s", step)

        if bundle.dynamic_connectivity_matrices is not None:
            logger.info(
                "  Dynamic FC windows:   %d  (shape %s)",
                bundle.dynamic_connectivity_matrices.shape[0],
                bundle.dynamic_connectivity_matrices.shape,
            )
