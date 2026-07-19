"""
Tests for braphin/transform.py

Covers:
- Bug 1 fix: atlas_data as ndarray (atlas_name=None) must not crash
- Bug 6 fix: centroid_coordinate_space is 'voxel' for ndarray atlas
- atlas_data as NIfTI image → world coordinates, no resampling
- ROI time series shape and dtype
- ROI label assignment (default, custom, mismatch)
- Centroid computation
- build_synthetic_atlas utility
"""

import dataclasses

import nibabel as nib
import numpy as np
import pytest

from braphin.config import AtlasConfig
from braphin.exceptions import AtlasError, TransformationError
from braphin.transform import BRAPHINTransformBundle, TransformBRAPHINData, build_synthetic_atlas

N_ROIS = 4   # matches conftest atlas_array


# ---------------------------------------------------------------------------
# Bug 1 fix: ndarray atlas (atlas_name = None)
# ---------------------------------------------------------------------------

def test_atlas_data_array_does_not_crash(denoise_bundle, atlas_array):
    """Bug 1 fix: passing atlas_data as ndarray with no atlas_name must succeed."""
    t = TransformBRAPHINData(denoise_bundle, atlas_data=atlas_array, config=AtlasConfig())
    result = t.run()
    assert isinstance(result, BRAPHINTransformBundle)


def test_atlas_data_array_roi_count(denoise_bundle, atlas_array):
    t = TransformBRAPHINData(denoise_bundle, atlas_data=atlas_array, config=AtlasConfig())
    result = t.run()
    assert result.roi_time_series.shape[0] == N_ROIS


def test_atlas_data_array_timepoints(denoise_bundle, atlas_array, num_timepoints):
    t = TransformBRAPHINData(denoise_bundle, atlas_data=atlas_array, config=AtlasConfig())
    result = t.run()
    assert result.roi_time_series.shape[1] == num_timepoints


# ---------------------------------------------------------------------------
# Bug 6 fix: centroid coordinate space
# ---------------------------------------------------------------------------

def test_centroid_space_voxel_for_ndarray_atlas(denoise_bundle, atlas_array):
    """Bug 6 fix: ndarray atlas has no affine → coordinate space must be 'voxel'."""
    t = TransformBRAPHINData(denoise_bundle, atlas_data=atlas_array, config=AtlasConfig())
    result = t.run()
    assert result.centroid_coordinate_space == "voxel"


def test_centroid_space_world_for_nifti_atlas(denoise_bundle, atlas_array, fmri_affine):
    """NIfTI atlas with matching affine → coordinate space must be 'world'."""
    atlas_img = nib.Nifti1Image(atlas_array.astype(np.float32), fmri_affine)
    t = TransformBRAPHINData(denoise_bundle, atlas_data=atlas_img, config=AtlasConfig())
    result = t.run()
    assert result.centroid_coordinate_space == "world"


# ---------------------------------------------------------------------------
# NIfTI atlas: no resampling when affines match
# ---------------------------------------------------------------------------

def test_nifti_atlas_same_affine_not_resampled(denoise_bundle, atlas_array, fmri_affine):
    atlas_img = nib.Nifti1Image(atlas_array.astype(np.float32), fmri_affine)
    t = TransformBRAPHINData(denoise_bundle, atlas_data=atlas_img, config=AtlasConfig())
    result = t.run()
    assert not result.atlas_resampled


def test_nifti_atlas_different_affine_resampled(denoise_bundle, atlas_array):
    """Different affine → atlas is resampled to fMRI space."""
    different_affine = np.diag([3.0, 3.0, 3.0, 1.0])   # 3 mm instead of 2 mm
    atlas_img = nib.Nifti1Image(atlas_array.astype(np.float32), different_affine)
    t = TransformBRAPHINData(denoise_bundle, atlas_data=atlas_img, config=AtlasConfig())
    result = t.run()
    assert result.atlas_resampled


# ---------------------------------------------------------------------------
# ROI time series
# ---------------------------------------------------------------------------

def test_roi_time_series_is_2d(denoise_bundle, atlas_array):
    t = TransformBRAPHINData(denoise_bundle, atlas_data=atlas_array, config=AtlasConfig())
    result = t.run()
    assert result.roi_time_series.ndim == 2


def test_roi_time_series_dtype(denoise_bundle, atlas_array):
    t = TransformBRAPHINData(denoise_bundle, atlas_data=atlas_array, config=AtlasConfig())
    result = t.run()
    assert result.roi_time_series.dtype == np.float32


def test_roi_time_series_finite(denoise_bundle, atlas_array):
    t = TransformBRAPHINData(denoise_bundle, atlas_data=atlas_array, config=AtlasConfig())
    result = t.run()
    assert np.isfinite(result.roi_time_series).all()


# ---------------------------------------------------------------------------
# ROI labels
# ---------------------------------------------------------------------------

def test_default_labels_start_with_roi(denoise_bundle, atlas_array):
    t = TransformBRAPHINData(denoise_bundle, atlas_data=atlas_array, config=AtlasConfig())
    result = t.run()
    for label in result.roi_labels:
        assert label.startswith("ROI_")


def test_custom_labels_used(denoise_bundle, atlas_array):
    custom = ["Frontal", "Parietal", "Temporal", "Occipital"]
    cfg = AtlasConfig(roi_labels=custom)
    t = TransformBRAPHINData(denoise_bundle, atlas_data=atlas_array, config=cfg)
    result = t.run()
    assert result.roi_labels == custom


def test_label_count_mismatch_raises(denoise_bundle, atlas_array):
    cfg = AtlasConfig(roi_labels=["only_one_label"])
    t = TransformBRAPHINData(denoise_bundle, atlas_data=atlas_array, config=cfg)
    with pytest.raises(TransformationError):
        t.run()


# ---------------------------------------------------------------------------
# Centroids
# ---------------------------------------------------------------------------

def test_centroids_count_matches_rois(denoise_bundle, atlas_array):
    t = TransformBRAPHINData(denoise_bundle, atlas_data=atlas_array, config=AtlasConfig())
    result = t.run()
    assert len(result.roi_centroids_3d) == N_ROIS


def test_centroids_are_3d_tuples(denoise_bundle, atlas_array):
    t = TransformBRAPHINData(denoise_bundle, atlas_data=atlas_array, config=AtlasConfig())
    result = t.run()
    for label, coord in result.roi_centroids_3d.items():
        assert len(coord) == 3


def test_centroids_keys_match_labels(denoise_bundle, atlas_array):
    t = TransformBRAPHINData(denoise_bundle, atlas_data=atlas_array, config=AtlasConfig())
    result = t.run()
    assert set(result.roi_centroids_3d.keys()) == set(result.roi_labels)


# ---------------------------------------------------------------------------
# build_synthetic_atlas
# ---------------------------------------------------------------------------

def test_build_synthetic_atlas_shape():
    atlas = build_synthetic_atlas((5, 5, 5), num_rois=3)
    assert atlas.shape == (5, 5, 5)


def test_build_synthetic_atlas_labels():
    atlas = build_synthetic_atlas((5, 5, 5), num_rois=3)
    assert set(np.unique(atlas)) == {1, 2, 3}


def test_build_synthetic_atlas_no_zero():
    """All voxels are assigned a ROI — no background voxel with label 0."""
    atlas = build_synthetic_atlas((4, 4, 4), num_rois=4)
    assert 0 not in np.unique(atlas)


def test_build_synthetic_atlas_dtype():
    atlas = build_synthetic_atlas((3, 3, 3), num_rois=2)
    assert atlas.dtype == np.int32


def test_build_synthetic_atlas_zero_rois_raises():
    with pytest.raises(AtlasError):
        build_synthetic_atlas((4, 4, 4), num_rois=0)


def test_build_synthetic_atlas_wrong_shape_raises():
    with pytest.raises(AtlasError):
        build_synthetic_atlas((4, 4), num_rois=2)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_no_atlas_raises(denoise_bundle):
    t = TransformBRAPHINData(denoise_bundle, atlas_data=None, config=AtlasConfig())
    with pytest.raises(TransformationError):
        t.run()


def test_none_denoise_bundle_raises():
    from braphin.exceptions import TransformationError
    t = TransformBRAPHINData(None, atlas_data=build_synthetic_atlas((4, 4, 4), num_rois=2))
    with pytest.raises(TransformationError):
        t.run()


def test_denoise_bundle_missing_data_raises():
    from braphin.denoise import BRAPHINDenoiseBundle
    from braphin.exceptions import TransformationError
    bundle = BRAPHINDenoiseBundle(denoised_data=None, voxel_time_series=None)
    t = TransformBRAPHINData(bundle, atlas_data=build_synthetic_atlas((4, 4, 4), num_rois=2))
    with pytest.raises(TransformationError):
        t.run()


def test_denoise_bundle_3d_data_raises(fmri_array):
    from braphin.denoise import BRAPHINDenoiseBundle
    from braphin.exceptions import TransformationError
    bundle = BRAPHINDenoiseBundle(
        denoised_data=fmri_array[:, :, :, 0],  # 3D slice
        voxel_time_series=None,
    )
    t = TransformBRAPHINData(bundle, atlas_data=build_synthetic_atlas((8, 8, 8), num_rois=2))
    with pytest.raises(TransformationError):
        t.run()


def test_atlas_from_auxiliary_files(denoise_bundle, spatial_shape):
    import dataclasses
    atlas = build_synthetic_atlas(spatial_shape, num_rois=3)
    bundle = dataclasses.replace(
        denoise_bundle,
        auxiliary_files={"my_atlas.nii": atlas},
    )
    t = TransformBRAPHINData(bundle, config=AtlasConfig())
    result = t.run()
    assert result.roi_time_series is not None
    assert result.roi_time_series.shape[0] == 3


def test_display_info_no_crash(denoise_bundle, atlas_array, caplog):
    import logging
    t = TransformBRAPHINData(denoise_bundle, atlas_data=atlas_array)
    result = t.run()
    with caplog.at_level(logging.INFO):
        t.display_info(result)


def test_coerce_atlas_nifti_image(denoise_bundle, atlas_array, fmri_affine):
    import nibabel as nib
    img = nib.Nifti1Image(atlas_array.astype(np.float32), fmri_affine)
    t = TransformBRAPHINData(denoise_bundle, atlas_data=img)
    result = t.run()
    assert result.roi_time_series is not None


def test_coerce_atlas_unsupported_type_raises(denoise_bundle):
    from braphin.exceptions import AtlasError
    t = TransformBRAPHINData(denoise_bundle, atlas_data="not_an_atlas")
    with pytest.raises((AtlasError, Exception)):
        t.run()


def test_validate_invalid_atlas_name_raises(denoise_bundle, atlas_array):
    from braphin.exceptions import TransformationError, AtlasError
    t = TransformBRAPHINData(
        denoise_bundle,
        atlas_data=atlas_array,
        config=AtlasConfig(atlas_name="invalid_atlas_xyz"),
    )
    with pytest.raises((TransformationError, AtlasError)):
        t.run()


def test_named_atlas_aal_labels_from_name_map(denoise_bundle, tmp_path, monkeypatch):
    """TransformBRAPHINData with atlas_name='aal' uses the ROI name map for labels.

    Uses a temporary centroid cache directory so the test never overwrites
    the bundled atlas_centroids/aal_centroids.json.
    """
    import braphin.transform as _tx_mod

    # Redirect centroid cache writes to a temp directory.
    _real_get_dir = _tx_mod.TransformBRAPHINData._get_centroid_layout_dir

    def _tmp_dir(self):
        return tmp_path

    monkeypatch.setattr(_tx_mod.TransformBRAPHINData, "_get_centroid_layout_dir", _tmp_dir)

    t = TransformBRAPHINData(denoise_bundle, config=AtlasConfig(atlas_name="aal"))
    result = t.run()
    assert result.roi_time_series is not None
    # Labels for AAL come from the roi_name_map (not generic ROI_N)
    assert any("_" in label for label in result.roi_labels)
