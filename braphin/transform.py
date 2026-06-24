import json
import logging
import os
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .atlas import get_atlas_definition, get_atlas_roi_name_map, resolve_supported_atlas_path
from .config import AtlasConfig
from .denoise import BRAPHINDenoiseBundle
from .exceptions import AtlasError, TransformationError
from .io.nifti import load_nifti_file

logger = logging.getLogger(__name__)


@dataclass
class BRAPHINTransformBundle:
    fmri_path: str | None = None
    original_metadata: dict[str, object] | None = None
    preprocess_metadata: dict[str, object] | None = None
    denoise_metadata: dict[str, object] | None = None
    atlas_name: str | None = None
    atlas_source: str | None = None
    atlas_labels: np.ndarray | None = None
    roi_time_series: np.ndarray | None = None
    roi_labels: list[str] = field(default_factory=list)

    centroid_coordinate_space: str | None = None

    roi_centroids_3d: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    centroid_json_path: str | None = None

    atlas_resampled: bool = False
    centroid_cache_used: bool = False

    auxiliary_files: dict[str, object] = field(default_factory=dict)
    applied_steps: list[str] = field(default_factory=list)
    transform_metadata: dict[str, object] = field(default_factory=dict)


class TransformBRAPHINData:
    """
    Stage 4 of the BRAPHIN pipeline: atlas parcellation and ROI time-series extraction.

    Accepts a denoised bundle and a 3-D atlas of integer labels, resamples the
    atlas to fMRI space if needed, extracts per-ROI mean time series, and
    returns an ``BRAPHINTransformBundle``.
    """

    def __init__(
        self,
        denoise_bundle: BRAPHINDenoiseBundle,
        atlas_data: object | None = None,
        config: AtlasConfig | None = None,
    ):
        self.denoise_bundle = denoise_bundle
        self.atlas_data = atlas_data
        self.config = config if config is not None else AtlasConfig()

    def run(self) -> BRAPHINTransformBundle:
        """
        Execute the atlas parcellation stage.

        Returns
        -------
        BRAPHINTransformBundle
            Bundle containing the validated 3-D atlas, the ROI × time matrix,
            ROI labels, world-space centroids, and traceability metadata.
        """
        self._validate_denoise_bundle()

        fmri_data = self.denoise_bundle.denoised_data
        spatial_shape = fmri_data.shape[:3]
        num_timepoints = fmri_data.shape[3]

        atlas_name = self.config.atlas_name.lower() if self.config.atlas_name else None
        fmri_affine = self._get_fmri_affine()

        atlas_source, atlas_labels, atlas_resampled, atlas_affine = self._resolve_atlas(
            spatial_shape=spatial_shape,
            fmri_affine=fmri_affine,
        )

        centroid_coordinate_space = "world" if atlas_affine is not None else "voxel"

        roi_ids = self._get_valid_roi_ids(atlas_labels)
        roi_labels = self._build_roi_labels(roi_ids)

        # Issue #14 — warn when a named atlas was resampled to fMRI space:
        # canonical centroids are from the reference atlas space (typically MNI152),
        # not the subject's native space.
        _centroid_space_warning: str | None = None
        if atlas_resampled and self.config.atlas_name is not None:
            warnings.warn(
                "Atlas was resampled to the fMRI voxel grid. The ROI centroid "
                "coordinates attached to the graph are from the original reference "
                "atlas space (typically MNI152), not the subject's native space. "
                "For native-space analyses, register your fMRI to MNI space first, "
                "or provide subject-specific ROI centroids.",
                UserWarning,
                stacklevel=2,
            )
            _centroid_space_warning = (
                "Centroids are in reference atlas space, not subject native space."
            )

        # Centroid resolution strategy:
        # - Named atlas  → canonical reference centroids (cached JSON, world space)
        # - Custom atlas → compute directly from supplied atlas labels

        if self.config.atlas_name is not None:
            # Named atlas: use canonical cached centroids (world space, JSON file).
            roi_centroids_3d, centroid_json_path = self._resolve_or_build_canonical_roi_centroids(
                roi_ids=roi_ids,
                roi_labels=roi_labels,
            )
            # Named atlases always have an affine → world space.
            centroid_coordinate_space = "world"
            centroid_cache_used = centroid_json_path is not None
        else:
            # Atlas passed directly (atlas_data) or detected in auxiliary files:
            # no stable name to cache on disk, so compute centroids from the resolved labels.
            roi_centroids_3d = self._compute_roi_centroids(
                atlas_labels=atlas_labels,
                roi_ids=roi_ids,
                roi_labels=roi_labels,
                atlas_affine=atlas_affine,
            )
            centroid_json_path = None
            centroid_cache_used = False
            # World space only when an affine is actually available.
            centroid_coordinate_space = "world" if atlas_affine is not None else "voxel"

        roi_time_series, roi_sizes = self._extract_roi_time_series(
            fmri_data=fmri_data,
            atlas_labels=atlas_labels,
            roi_ids=roi_ids,
        )

        applied_steps = [
            "validate_atlas_shape",
            "extract_roi_mean_time_series",
        ]

        if atlas_resampled:
            applied_steps.insert(0, "resample_atlas_to_fmri_space")

        transform_metadata = self._build_transform_metadata(
            spatial_shape=spatial_shape,
            num_timepoints=num_timepoints,
            atlas_labels=atlas_labels,
            roi_ids=roi_ids,
            roi_sizes=roi_sizes,
            atlas_source=atlas_source,
            atlas_name=atlas_name,
            atlas_resampled=atlas_resampled,
            centroid_json_path=centroid_json_path,
            centroid_cache_used=centroid_cache_used,
            centroid_coordinate_space=centroid_coordinate_space,
        )

        bundle = BRAPHINTransformBundle(
            fmri_path=self.denoise_bundle.fmri_path,
            original_metadata=self.denoise_bundle.original_metadata,
            preprocess_metadata=self.denoise_bundle.preprocess_metadata,
            denoise_metadata=self.denoise_bundle.denoise_metadata,
            atlas_name=atlas_name,
            atlas_source=atlas_source,
            atlas_labels=atlas_labels,
            roi_time_series=roi_time_series,
            roi_labels=roi_labels,
            roi_centroids_3d=roi_centroids_3d,
            centroid_json_path=centroid_json_path,
            atlas_resampled=atlas_resampled,
            centroid_cache_used=centroid_cache_used,
            centroid_coordinate_space=centroid_coordinate_space,
            auxiliary_files=dict(self.denoise_bundle.auxiliary_files),
            applied_steps=applied_steps,
            transform_metadata=transform_metadata,
        )

        return bundle

    def _validate_denoise_bundle(self) -> None:
        """Verify that the denoise bundle contains a valid 4-D volume."""
        if self.denoise_bundle is None:
            raise TransformationError("No valid BRAPHINDenoiseBundle was provided.")

        if self.denoise_bundle.denoised_data is None:
            raise TransformationError("The denoise bundle does not contain 4-D denoised data.")

        if not isinstance(self.denoise_bundle.denoised_data, np.ndarray):
            raise TransformationError("The denoised data is not a NumPy ndarray.")

        if self.denoise_bundle.denoised_data.ndim != 4:
            raise TransformationError(
                f"Expected a 4-D denoised volume, but received shape "
                f"{self.denoise_bundle.denoised_data.shape}."
            )

    def _resolve_atlas(
        self, spatial_shape: tuple[int, int, int], fmri_affine: np.ndarray
    ) -> tuple[str, np.ndarray, bool, np.ndarray | None]:
        """
        Determine which atlas to use and resample it to fMRI space if needed.

        Returns
        -------
        Tuple of (atlas_source, atlas_labels, atlas_resampled, atlas_affine).
        """
        self._validate_atlas_name_if_provided()

        # 1. Atlas passed directly to the constructor.
        if self.atlas_data is not None:
            return self._prepare_atlas_for_fmri_space(
                atlas_data=self.atlas_data,
                spatial_shape=spatial_shape,
                fmri_affine=fmri_affine,
                atlas_source="direct_input",
            )

        # 2. Atlas resolved from config (atlas_name / atlas_path).
        atlas_from_config = self._load_atlas_from_config_if_available()
        if atlas_from_config is not None:
            source_name = (
                f"config_atlas:{self.config.atlas_name.lower()}"
                if self.config.atlas_name is not None
                else f"config_path:{self.config.atlas_path}"
            )
            return self._prepare_atlas_for_fmri_space(
                atlas_data=atlas_from_config,
                spatial_shape=spatial_shape,
                fmri_affine=fmri_affine,
                atlas_source=source_name,
            )

        # 3. Atlas detected among auxiliary files.
        aux_name, aux_atlas = self._find_atlas_in_auxiliary_files(
            self.denoise_bundle.auxiliary_files
        )
        if aux_atlas is not None:
            self._validate_atlas_labels(aux_atlas, spatial_shape)
            return f"auxiliary_file:{aux_name}", aux_atlas, False, None

        raise TransformationError(
            "No atlas was provided. Pass atlas_data directly, set atlas_name or "
            "atlas_path in AtlasConfig, or include an atlas array in the auxiliary files."
        )

    def _validate_atlas_name_if_provided(self) -> None:
        """
        Validate the logical atlas name against the supported-atlas registry.

        Does not load the atlas; only checks that the name is recognised.
        """
        if self.config.atlas_name is None:
            return

        get_atlas_definition(self.config.atlas_name)

    def _find_atlas_in_auxiliary_files(
        self, auxiliary_files: dict[str, object]
    ) -> tuple[str | None, np.ndarray | None]:
        """
        Search for a 3-D atlas array among the auxiliary files.

        Looks for filenames containing 'atlas' and accepts 3-D NumPy arrays.
        """
        for file_name, data in auxiliary_files.items():
            if "atlas" not in file_name.lower():
                continue

            if isinstance(data, np.ndarray) and data.ndim == 3:
                return file_name, self._coerce_atlas_to_array(data)

        return None, None

    def _coerce_atlas_to_array(self, atlas_data: object) -> np.ndarray:
        """
        Convert various atlas representations to a 3-D integer ndarray.

        Supported inputs:
        - 3-D NumPy ndarray
        - NIfTI-like object with ``get_fdata()``
        """
        if isinstance(atlas_data, np.ndarray):
            atlas_array = atlas_data
        elif hasattr(atlas_data, "get_fdata"):
            atlas_array = atlas_data.get_fdata()
        else:
            raise AtlasError(
                "Unsupported atlas format. Provide a 3-D ndarray or a loaded NIfTI image."
            )

        if not isinstance(atlas_array, np.ndarray):
            atlas_array = np.asarray(atlas_array)

        if atlas_array.ndim != 3:
            raise AtlasError(
                f"Atlas must be 3-D, but received an array with shape {atlas_array.shape}."
            )

        # Round to nearest integer because the atlas contains region labels.
        return np.rint(atlas_array).astype(np.int32)

    def _validate_atlas_labels(
        self, atlas_labels: np.ndarray, spatial_shape: tuple[int, int, int]
    ) -> None:
        """Validate that the atlas spatial shape matches the fMRI volume."""
        if atlas_labels.shape != spatial_shape:
            raise AtlasError(
                f"Atlas shape {atlas_labels.shape} does not match fMRI spatial "
                f"shape {spatial_shape}."
            )

        positive_labels = atlas_labels[atlas_labels > 0]
        if positive_labels.size == 0:
            raise AtlasError("The atlas contains no valid ROI labels (all values are 0 or below).")

    def _get_valid_roi_ids(self, atlas_labels: np.ndarray) -> np.ndarray:
        """
        Return unique positive ROI label values from the atlas.

        Convention: label 0 is background; all values > 0 are ROIs.
        """
        roi_ids = np.unique(atlas_labels)
        roi_ids = roi_ids[roi_ids > 0]

        if roi_ids.size == 0:
            raise TransformationError("No valid ROI labels found in the atlas (all values are 0).")

        return roi_ids

    def _build_roi_labels(self, roi_ids: np.ndarray) -> list[str]:
        """
        Build human-readable labels for each ROI.

        Priority:
        1. Manual ``roi_labels`` provided in ``AtlasConfig``
        2. Anatomical names from the atlas registry (if atlas_name is set)
        3. Generic fallback: ``ROI_<id>``
        """
        if self.config.roi_labels is not None:
            if len(self.config.roi_labels) != len(roi_ids):
                raise TransformationError(
                    "The number of roi_labels does not match the number of ROIs in the atlas."
                )
            return list(self.config.roi_labels)

        if self.config.atlas_name is not None:
            roi_name_map = get_atlas_roi_name_map(self.config.atlas_name)
            if roi_name_map is not None:
                labels = []
                for roi_id in roi_ids:
                    roi_id_int = int(roi_id)
                    labels.append(roi_name_map.get(roi_id_int, f"ROI_{roi_id_int}"))
                return labels

        return [f"ROI_{int(roi_id)}" for roi_id in roi_ids]

    def _extract_roi_time_series(
        self,
        fmri_data: np.ndarray,
        atlas_labels: np.ndarray,
        roi_ids: np.ndarray,
    ) -> tuple[np.ndarray, dict[int, int]]:
        """
        Extract mean time series for each ROI.

        For each ROI label, selects all voxels belonging to that region and
        averages their signal across voxels at each timepoint.

        Returns
        -------
        roi_time_series : ndarray of shape (num_rois, T)
        roi_sizes : dict mapping ROI id → number of voxels
        """
        num_timepoints = fmri_data.shape[3]
        roi_series_list: list[np.ndarray] = []
        roi_sizes: dict[int, int] = {}

        for roi_id in roi_ids:
            roi_mask = atlas_labels == roi_id
            roi_voxels = fmri_data[roi_mask]

            if roi_voxels.size == 0:
                raise TransformationError(f"ROI {int(roi_id)} contains no voxels.")

            # Indexing a 4-D volume with a 3-D boolean mask produces shape
            # (num_voxels_roi, T).
            if roi_voxels.ndim != 2 or roi_voxels.shape[1] != num_timepoints:
                raise TransformationError(
                    f"ROI {int(roi_id)} did not produce a valid voxel x time matrix."
                )

            roi_mean_ts = np.mean(roi_voxels, axis=0, dtype=np.float32)
            roi_series_list.append(roi_mean_ts.astype(np.float32))
            roi_sizes[int(roi_id)] = int(roi_voxels.shape[0])

        roi_time_series = np.vstack(roi_series_list).astype(np.float32)
        return roi_time_series, roi_sizes

    def _build_transform_metadata(
        self,
        spatial_shape: tuple[int, int, int],
        num_timepoints: int,
        atlas_labels: np.ndarray,
        roi_ids: np.ndarray,
        roi_sizes: dict[int, int],
        atlas_source: str,
        atlas_name: str | None,
        atlas_resampled: bool,
        centroid_json_path: str | None,
        centroid_cache_used: bool,
        centroid_coordinate_space: str,
    ) -> dict[str, object]:
        """Build traceability metadata for the atlas parcellation stage."""
        metadata: dict[str, object] = {
            "fmri_spatial_shape": spatial_shape,
            "num_timepoints": int(num_timepoints),
            "atlas_shape": tuple(atlas_labels.shape),
            "atlas_source": atlas_source,
            "atlas_name": atlas_name,
            "num_rois": int(len(roi_ids)),
            "roi_ids": [int(x) for x in roi_ids.tolist()],
            "roi_sizes": roi_sizes,
            "roi_time_series_shape": (int(len(roi_ids)), int(num_timepoints)),
            "implemented_scope": ["validate_atlas_shape", "extract_mean_roi_time_series"],
            "atlas_resampled": bool(atlas_resampled),
            "centroid_coordinate_space": centroid_coordinate_space,
            "centroid_json_path": centroid_json_path,
            "centroid_cache_used": bool(centroid_cache_used),
            "fmri_affine_available": self.denoise_bundle.original_metadata is not None
            and "affine" in self.denoise_bundle.original_metadata,
        }

        if atlas_name is not None:
            atlas_definition = get_atlas_definition(atlas_name)
            metadata["atlas_family"] = atlas_definition.family
            metadata["atlas_description"] = atlas_definition.description
            metadata["atlas_expected_num_rois"] = atlas_definition.num_rois
            # Canonical centroids are always computed/cached for named atlases,
            # regardless of whether the atlas was resampled to fMRI space.
            metadata["centroid_json_path"] = str(self._get_centroid_json_path(roi_ids))
            metadata["has_saved_centroids"] = True

        return metadata

    def display_info(self, bundle: BRAPHINTransformBundle) -> None:
        """Log a summary of the atlas parcellation result."""
        logger.info("[BRAPHIN] ROI transformation completed")
        logger.info("  fMRI path:        %s", bundle.fmri_path)
        logger.info("  Atlas source:     %s", bundle.atlas_source)
        logger.info("  Atlas name:       %s", bundle.atlas_name)
        logger.info(
            "  Centroid space:   %s",
            bundle.transform_metadata.get("centroid_coordinate_space"),
        )

        if bundle.transform_metadata:
            logger.info("  Atlas shape:      %s", bundle.transform_metadata.get("atlas_shape"))
            logger.info("  Number of ROIs:   %s", bundle.transform_metadata.get("num_rois"))
            logger.info("  Atlas resampled:  %s", bundle.transform_metadata.get("atlas_resampled"))
            logger.info(
                "  Centroid cache:   %s", bundle.transform_metadata.get("centroid_cache_used")
            )
            logger.info(
                "  Centroid JSON:    %s", bundle.transform_metadata.get("centroid_json_path")
            )
            logger.info(
                "  ROI x time shape: %s",
                bundle.transform_metadata.get("roi_time_series_shape"),
            )

        if bundle.applied_steps:
            logger.info("  Applied steps:")
            for step in bundle.applied_steps:
                logger.info("    - %s", step)

    def _get_centroid_layout_dir(self) -> Path:
        """Return the directory where per-atlas centroid JSON files are stored."""
        layouts_dir = Path(__file__).resolve().parent / "atlas_centroids"
        layouts_dir.mkdir(parents=True, exist_ok=True)
        return layouts_dir

    def _build_centroid_json_name(self, roi_ids: np.ndarray) -> str:
        """
        Build the centroid JSON filename for the current atlas.

        Uses the atlas name when available; falls back to a generic name
        based on the number of ROIs.
        """
        if self.config.atlas_name:
            return f"{self.config.atlas_name.lower()}_centroids.json"

        return f"custom_{len(roi_ids)}rois_centroids.json"

    def _get_centroid_json_path(self, roi_ids: np.ndarray) -> Path:
        return self._get_centroid_layout_dir() / self._build_centroid_json_name(roi_ids)

    def _resolve_or_build_canonical_roi_centroids(
        self, roi_ids: np.ndarray, roi_labels: list[str]
    ) -> tuple[dict[str, tuple[float, float, float]], str | None]:
        """
        Return subject-independent canonical centroids for the named atlas.

        Centroids are computed from the original reference atlas associated
        with ``atlas_name``, not from the atlas resampled to fMRI space.
        Results are cached as JSON to avoid repeated computation.
        """
        if self.config.atlas_name is None:
            raise TransformationError("Cannot build canonical centroids when atlas_name is None.")

        json_path = self._get_centroid_json_path(roi_ids)

        # 1. Attempt to load from the cached JSON file.
        if json_path.exists():
            centroids = self._load_roi_centroids_from_json(
                json_path=json_path,
                expected_roi_labels=roi_labels,
                expected_coordinate_space="world",
            )
            if centroids:
                return centroids, str(json_path)

        # 2. Load the original reference atlas and compute centroids.
        atlas_ref = self._load_atlas_from_config_if_available()
        if atlas_ref is None:
            raise TransformationError(
                f"Could not load reference atlas for '{self.config.atlas_name}'."
            )

        atlas_ref_labels = self._coerce_atlas_to_array(atlas_ref)
        atlas_ref_affine = (
            np.asarray(atlas_ref.affine, dtype=float) if hasattr(atlas_ref, "affine") else None
        )

        centroids = self._compute_roi_centroids(
            atlas_labels=atlas_ref_labels,
            roi_ids=roi_ids,
            roi_labels=roi_labels,
            atlas_affine=atlas_ref_affine,
        )

        self._save_roi_centroids_to_json(
            json_path=json_path,
            roi_ids=roi_ids,
            roi_labels=roi_labels,
            centroids=centroids,
            atlas_shape=atlas_ref_labels.shape,
            atlas_name=self.config.atlas_name,
            coordinate_space="world",
        )

        return centroids, str(json_path)

    def _compute_roi_centroids(
        self,
        atlas_labels: np.ndarray,
        roi_ids: np.ndarray,
        roi_labels: list[str],
        atlas_affine: np.ndarray | None = None,
    ) -> dict[str, tuple[float, float, float]]:
        """
        Compute the centroid of each ROI.

        Returns world-space coordinates when an affine matrix is available;
        falls back to voxel-space coordinates otherwise.
        """
        centroids: dict[str, tuple[float, float, float]] = {}

        for roi_id, roi_label in zip(roi_ids, roi_labels, strict=True):
            coords = np.argwhere(atlas_labels == roi_id)
            if coords.size == 0:
                raise TransformationError(
                    f"Could not compute centroid for ROI {int(roi_id)}: no voxels found."
                )

            voxel_centroid = coords.mean(axis=0).astype(np.float64)

            if atlas_affine is not None:
                voxel_h = np.append(voxel_centroid, 1.0)
                world_centroid = np.asarray(atlas_affine, dtype=np.float64) @ voxel_h
                centroid = world_centroid[:3]
            else:
                centroid = voxel_centroid

            centroids[roi_label] = (
                float(centroid[0]),
                float(centroid[1]),
                float(centroid[2]),
            )

        return centroids

    def _save_roi_centroids_to_json(
        self,
        json_path: Path,
        roi_ids: np.ndarray,
        roi_labels: list[str],
        centroids: dict[str, tuple[float, float, float]],
        atlas_shape: tuple[int, int, int],
        atlas_name: str | None,
        coordinate_space: str,
    ) -> None:
        payload = {
            "atlas_name": atlas_name,
            "atlas_shape": list(atlas_shape),
            "coordinate_space": coordinate_space,
            "rois": [
                {
                    "roi_id": int(roi_id),
                    "roi_label": roi_label,
                    "centroid": list(centroids[roi_label]),
                }
                for roi_id, roi_label in zip(roi_ids, roi_labels, strict=True)
            ],
        }

        # Write atomically (temp file → rename) so concurrent workers cannot
        # produce a partially-written / interleaved file.
        dir_ = json_path.parent
        dir_.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            os.replace(tmp, json_path)  # atomic on POSIX / Windows
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _load_roi_centroids_from_json(
        self,
        json_path: Path,
        expected_roi_labels: list[str] | None = None,
        expected_coordinate_space: str | None = None,
    ) -> dict[str, tuple[float, float, float]]:
        with json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        payload_coordinate_space = payload.get("coordinate_space")
        if (
            expected_coordinate_space is not None
            and payload_coordinate_space != expected_coordinate_space
        ):
            return {}

        rois = payload.get("rois", [])
        payload_labels = [roi.get("roi_label") for roi in rois]

        if expected_roi_labels is not None and payload_labels != list(expected_roi_labels):
            return {}

        centroids: dict[str, tuple[float, float, float]] = {}
        for roi in rois:
            label = roi["roi_label"]
            centroid = roi["centroid"]
            centroids[label] = (
                float(centroid[0]),
                float(centroid[1]),
                float(centroid[2]),
            )

        return centroids

    def _get_fmri_affine(self) -> np.ndarray:
        """Retrieve the fMRI affine matrix from the inherited metadata."""
        metadata = self.denoise_bundle.original_metadata

        if metadata is None or "affine" not in metadata:
            raise TransformationError("fMRI affine matrix not found in the original metadata.")

        affine = np.asarray(metadata["affine"])

        if affine.shape != (4, 4):
            raise TransformationError(
                f"fMRI affine must be a 4×4 matrix, but received shape {affine.shape}."
            )

        return affine

    def _atlas_matches_fmri_space(
        self,
        atlas_img,
        spatial_shape: tuple[int, int, int],
        fmri_affine: np.ndarray,
        atol: float = 1e-5,
    ) -> bool:
        """Return True if the atlas and fMRI already share the same voxel grid."""
        if tuple(atlas_img.shape) != tuple(spatial_shape):
            return False

        atlas_affine = np.asarray(atlas_img.affine)
        return np.allclose(atlas_affine, fmri_affine, atol=atol)

    def _resample_atlas_to_fmri_space(
        self, atlas_img, spatial_shape: tuple[int, int, int], fmri_affine: np.ndarray
    ):
        """Resample the atlas to the fMRI voxel grid using nearest-neighbour interpolation."""
        try:
            from nibabel.processing import resample_from_to
        except ImportError as exc:
            raise TransformationError(
                "Could not import nibabel.processing to resample the atlas."
            ) from exc

        target = (spatial_shape, fmri_affine)

        try:
            resampled_img = resample_from_to(atlas_img, target, order=0)
        except Exception as exc:
            raise TransformationError("Failed to resample the atlas to fMRI space.") from exc

        return resampled_img

    def _prepare_atlas_for_fmri_space(
        self,
        atlas_data: object,
        spatial_shape: tuple[int, int, int],
        fmri_affine: np.ndarray,
        atlas_source: str,
    ) -> tuple[str, np.ndarray, bool, np.ndarray | None]:
        """
        Ensure the atlas is aligned to the fMRI voxel grid.

        Returns
        -------
        atlas_source : str
        atlas_labels : ndarray
        atlas_resampled : bool
        atlas_affine : ndarray or None
            Affine used for centroid computation; None for raw ndarray inputs.
        """
        # Case 1: NIfTI-like image — compare affines and resample if needed.
        if hasattr(atlas_data, "get_fdata") and hasattr(atlas_data, "affine"):
            atlas_img = atlas_data

            if self._atlas_matches_fmri_space(atlas_img, spatial_shape, fmri_affine):
                atlas_labels = self._coerce_atlas_to_array(atlas_img)
                self._validate_atlas_labels(atlas_labels, spatial_shape)
                return atlas_source, atlas_labels, False, np.asarray(atlas_img.affine, dtype=float)

            resampled_img = self._resample_atlas_to_fmri_space(
                atlas_img=atlas_img,
                spatial_shape=spatial_shape,
                fmri_affine=fmri_affine,
            )
            atlas_labels = self._coerce_atlas_to_array(resampled_img)
            self._validate_atlas_labels(atlas_labels, spatial_shape)
            return (
                f"{atlas_source}|resampled_to_fmri",
                atlas_labels,
                True,
                np.asarray(resampled_img.affine, dtype=float),
            )

        # Case 2: raw ndarray — cannot resample without an affine.
        atlas_labels = self._coerce_atlas_to_array(atlas_data)
        self._validate_atlas_labels(atlas_labels, spatial_shape)
        return atlas_source, atlas_labels, False, None

    def _load_atlas_from_config_if_available(self):
        """
        Load the atlas from the configuration when ``atlas_name`` or
        ``atlas_path`` is set; return ``None`` if neither is provided.
        """
        if self.config.atlas_name is None and self.config.atlas_path is None:
            return None

        if self.config.atlas_name is None and self.config.atlas_path is not None:
            atlas_path = Path(self.config.atlas_path)
            if not atlas_path.exists():
                raise TransformationError(
                    f"Atlas file specified in atlas_path not found: {atlas_path}"
                )
            return load_nifti_file(atlas_path)

        atlas_name = self.config.atlas_name.lower()
        resolved_path = resolve_supported_atlas_path(
            atlas_name=atlas_name,
            atlas_path=self.config.atlas_path,
        )
        return load_nifti_file(resolved_path)


def build_synthetic_atlas(spatial_shape: tuple[int, int, int], num_rois: int = 8) -> np.ndarray:
    """
    Build a simple synthetic atlas for testing.

    Divides all voxels of the volume into ``num_rois`` contiguous groups and
    assigns labels 1…num_rois. No background (0) is created; every voxel
    belongs to exactly one ROI.

    Parameters
    ----------
    spatial_shape : tuple of int
        3-D shape of the volume (X, Y, Z).
    num_rois : int
        Number of synthetic ROI regions.

    Returns
    -------
    ndarray of shape ``spatial_shape`` with integer ROI labels.
    """
    if len(spatial_shape) != 3:
        raise AtlasError(f"spatial_shape must have 3 dimensions, but received {spatial_shape}.")

    if num_rois < 1:
        raise AtlasError("num_rois must be at least 1.")

    num_voxels = int(np.prod(spatial_shape))
    if num_rois > num_voxels:
        raise AtlasError("num_rois cannot exceed the total number of voxels in the volume.")

    labels_flat = np.zeros(num_voxels, dtype=np.int32)
    voxel_indices = np.arange(num_voxels)
    voxel_groups = np.array_split(voxel_indices, num_rois)

    for roi_index, group in enumerate(voxel_groups, start=1):
        labels_flat[group] = roi_index

    atlas_labels = labels_flat.reshape(spatial_shape)
    return atlas_labels
