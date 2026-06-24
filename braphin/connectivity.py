"""
Stage 5 of the BRAPHIN pipeline: functional connectivity modelling.

Computes the ROI × ROI connectivity matrix from the parcellated time series
produced by :class:`~braphin.transform.TransformBRAPHINData`.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .config import ConnectivityConfig
from .exceptions import ConnectivityError
from .strategy import get_connectivity_strategy
from .tools import apply_connectivity_threshold
from .transform import BRAPHINTransformBundle

logger = logging.getLogger(__name__)


@dataclass
class BRAPHINConnectivityBundle:
    """
    Output bundle of the connectivity modelling stage.

    Fields
    ------
    fmri_path : str or None
        Path to the original fMRI file.
    original_metadata : dict or None
        Metadata inherited from the input stage.
    preprocess_metadata : dict or None
        Metadata inherited from the preprocessing stage.
    denoise_metadata : dict or None
        Metadata inherited from the denoising stage.
    transform_metadata : dict or None
        Metadata inherited from the atlas parcellation stage.
    atlas_name : str or None
        Logical name of the atlas used.
    roi_labels : list of str
        Labels for each ROI.
    roi_time_series : ndarray (N, T) or None
        Original ROI × time matrix passed to connectivity computation.
    connectivity_matrix : ndarray (N, N) or None
        Pairwise connectivity matrix.
    applied_steps : list of str
        Steps actually executed (e.g. ``"pearson_correlation"``,
        ``"threshold"``).
    pending_steps : list of str
        Steps requested but deferred (e.g. ``"windowed_dynamic_connectivity"``).
    connectivity_metadata : dict
        Traceability information (method, shape, statistics, …).
    """
    fmri_path: Optional[str] = None
    original_metadata: Optional[Dict[str, object]] = None
    preprocess_metadata: Optional[Dict[str, object]] = None
    denoise_metadata: Optional[Dict[str, object]] = None
    transform_metadata: Optional[Dict[str, object]] = None
    atlas_name: Optional[str] = None
    roi_labels: List[str] = field(default_factory=list)
    roi_time_series: Optional[np.ndarray] = None
    connectivity_matrix: Optional[np.ndarray] = None
    applied_steps: List[str] = field(default_factory=list)
    pending_steps: List[str] = field(default_factory=list)
    connectivity_metadata: Dict[str, object] = field(default_factory=dict)


class ModelBRAPHINConnectivityData:
    """
    Stage 5 of the BRAPHIN pipeline: functional connectivity modelling.

    Accepts an :class:`~braphin.transform.BRAPHINTransformBundle`, selects the
    requested connectivity strategy, computes the ROI × ROI matrix, optionally
    applies an absolute threshold, and returns an
    :class:`BRAPHINConnectivityBundle`.
    """

    def __init__(
        self,
        transform_bundle: BRAPHINTransformBundle,
        config: Optional[ConnectivityConfig] = None,
    ):
        self.transform_bundle = transform_bundle
        self.config = config if config is not None else ConnectivityConfig()

    def run(self) -> BRAPHINConnectivityBundle:
        """
        Execute the connectivity modelling stage.

        Returns
        -------
        BRAPHINConnectivityBundle
            Bundle containing the connectivity matrix, applied steps, and
            traceability metadata.
        """
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

        # ── 2. Compute ROI × ROI matrix ───────────────────────────────────────
        connectivity_matrix = strategy.compute(roi_time_series)

        applied_steps = [self.config.method]
        pending_steps: List[str] = []

        # ── 3. Optional absolute threshold ───────────────────────────────────
        if self.config.threshold is not None:
            connectivity_matrix = apply_connectivity_threshold(
                connectivity_matrix,
                self.config.threshold,
            )
            applied_steps.append("threshold")

        # ── 4. Dynamic connectivity (planned) ─────────────────────────────────
        # window_size is not None signals a request for windowed dynamic
        # connectivity, which is not yet implemented; flag it as pending.
        if self.config.window_size is not None:
            pending_steps.append("windowed_dynamic_connectivity")

        connectivity_metadata = self._build_connectivity_metadata(
            roi_time_series=roi_time_series,
            connectivity_matrix=connectivity_matrix,
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
            applied_steps=applied_steps,
            pending_steps=pending_steps,
            connectivity_metadata=connectivity_metadata,
        )

        return bundle

    # ── Validation ─────────────────────────────────────────────────────────────

    def _validate_transform_bundle(self) -> None:
        """Verify that the transform bundle contains the required data."""
        if self.transform_bundle is None:
            raise ConnectivityError(
                "A valid BRAPHINTransformBundle must be provided."
            )

        if self.transform_bundle.roi_time_series is None:
            raise ConnectivityError(
                "The transform bundle does not contain roi_time_series."
            )

        if not isinstance(self.transform_bundle.roi_time_series, np.ndarray):
            raise ConnectivityError(
                "roi_time_series must be a NumPy ndarray."
            )

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
    ) -> Dict[str, object]:
        """Build traceability metadata for the connectivity stage."""
        off_diagonal_mask = ~np.eye(connectivity_matrix.shape[0], dtype=bool)
        off_diagonal_values = connectivity_matrix[off_diagonal_mask]

        return {
            "method": self.config.method,
            "threshold": self.config.threshold,
            "window_size": self.config.window_size,
            "tr": self.config.tr,
            "num_rois": int(roi_time_series.shape[0]),
            "num_timepoints": int(roi_time_series.shape[1]),
            "roi_time_series_shape": tuple(roi_time_series.shape),
            "connectivity_matrix_shape": tuple(connectivity_matrix.shape),
            "matrix_dtype": str(connectivity_matrix.dtype),
            "matrix_is_symmetric": bool(
                np.allclose(connectivity_matrix, connectivity_matrix.T, atol=1e-5)
            ),
            "diagonal_all_ones": bool(
                np.allclose(np.diag(connectivity_matrix), 1.0, atol=1e-5)
            ),
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
        logger.info("  ROI x time shape:     %s", bundle.connectivity_metadata.get("roi_time_series_shape"))
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
