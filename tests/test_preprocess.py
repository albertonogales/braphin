"""
Tests for braphin/preprocess.py

Covers:
- Bundle shape and dtype
- NaN / inf replacement
- Global intensity normalisation
- applied_steps / pending_steps tracking
- Validation errors
"""

import dataclasses

import nibabel as nib
import numpy as np
import pytest

from braphin.config import PreprocessConfig
from braphin.exceptions import PreprocessingError
from braphin.importBRAPHINData import BRAPHINInputBundle
from braphin.io.nifti import get_nifti_metadata
from braphin.preprocess import BRAPHINPreprocessBundle, PreprocessBRAPHINData


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bundle(arr, affine=None):
    if affine is None:
        affine = np.eye(4)
    img = nib.Nifti1Image(arr, affine)
    meta = get_nifti_metadata(img)
    return BRAPHINInputBundle(fmri_path="fake.nii.gz", fmri_image=img, fmri_metadata=meta)


def _run(arr, affine=None, **cfg_kwargs):
    bundle = _make_bundle(arr, affine)
    cfg = PreprocessConfig(
        apply_motion_correction=False,
        apply_slice_timing=False,
        apply_outlier_detection=False,
        apply_voxel_zscore=False,
        apply_smoothing=False,
        **cfg_kwargs,
    )
    return PreprocessBRAPHINData(bundle, cfg).run()


# ---------------------------------------------------------------------------
# Basic output shape and type
# ---------------------------------------------------------------------------

def test_preprocess_returns_bundle(preprocess_bundle):
    assert isinstance(preprocess_bundle, BRAPHINPreprocessBundle)


def test_preprocessed_data_shape(preprocess_bundle, spatial_shape, num_timepoints):
    assert preprocess_bundle.preprocessed_data.shape == (*spatial_shape, num_timepoints)


def test_preprocessed_data_dtype(preprocess_bundle):
    assert preprocess_bundle.preprocessed_data.dtype == np.float32


def test_voxel_time_series_shape(preprocess_bundle, spatial_shape, num_timepoints):
    n_voxels = spatial_shape[0] * spatial_shape[1] * spatial_shape[2]
    assert preprocess_bundle.voxel_time_series.shape == (n_voxels, num_timepoints)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def test_metadata_ndim(preprocess_bundle):
    assert preprocess_bundle.preprocess_metadata["ndim"] == 4


def test_metadata_num_timepoints(preprocess_bundle, num_timepoints):
    assert preprocess_bundle.preprocess_metadata["num_timepoints"] == num_timepoints


def test_metadata_num_voxels(preprocess_bundle, spatial_shape):
    expected = spatial_shape[0] * spatial_shape[1] * spatial_shape[2]
    assert preprocess_bundle.preprocess_metadata["num_voxels"] == expected


# ---------------------------------------------------------------------------
# NaN / inf replacement
# ---------------------------------------------------------------------------

def test_replaces_nan_values():
    arr = np.ones((4, 4, 4, 10), dtype=np.float32) * 500.0
    arr[0, 0, 0, 0] = np.nan
    arr[1, 1, 1, 5] = np.inf
    arr[2, 2, 2, 9] = -np.inf
    result = _run(arr)
    assert np.isfinite(result.preprocessed_data).all()


def test_replaced_count_in_metadata():
    arr = np.ones((4, 4, 4, 10), dtype=np.float32) * 500.0
    arr[0, 0, 0, 0] = np.nan
    arr[3, 3, 3, 9] = np.inf
    result = _run(arr)
    assert result.preprocess_metadata["non_finite_values_replaced"] == 2


def test_clean_data_reports_zero_replacements():
    arr = np.ones((4, 4, 4, 10), dtype=np.float32) * 500.0
    result = _run(arr)
    assert result.preprocess_metadata["non_finite_values_replaced"] == 0


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def test_normalization_changes_data(input_bundle):
    cfg_off = PreprocessConfig(
        apply_motion_correction=False, apply_slice_timing=False,
        apply_outlier_detection=False, apply_voxel_zscore=False, apply_smoothing=False,
    )
    cfg_on = PreprocessConfig(
        apply_motion_correction=False, apply_slice_timing=False,
        apply_outlier_detection=False, apply_voxel_zscore=True, apply_smoothing=False,
    )
    raw = PreprocessBRAPHINData(input_bundle, cfg_off).run().preprocessed_data
    normed = PreprocessBRAPHINData(input_bundle, cfg_on).run().preprocessed_data
    assert not np.allclose(raw, normed)


def test_normalization_step_recorded(input_bundle):
    cfg = PreprocessConfig(
        apply_motion_correction=False, apply_slice_timing=False,
        apply_outlier_detection=False, apply_voxel_zscore=True, apply_smoothing=False,
    )
    result = PreprocessBRAPHINData(input_bundle, cfg).run()
    assert "per_voxel_temporal_zscore" in result.applied_steps


def test_normalization_metadata_flag(input_bundle):
    cfg = PreprocessConfig(
        apply_motion_correction=False, apply_slice_timing=False,
        apply_outlier_detection=False, apply_voxel_zscore=True, apply_smoothing=False,
    )
    result = PreprocessBRAPHINData(input_bundle, cfg).run()
    assert result.preprocess_metadata["voxel_zscore_applied"] is True


def test_no_normalization_metadata_flag(preprocess_bundle):
    assert preprocess_bundle.preprocess_metadata["voxel_zscore_applied"] is False


# ---------------------------------------------------------------------------
# Steps tracking
# ---------------------------------------------------------------------------

def test_all_steps_in_applied_steps_when_flags_true(input_bundle):
    """All four previously-stubbed steps are now implemented and appear in applied_steps."""
    cfg = PreprocessConfig(
        apply_motion_correction=True,
        apply_slice_timing=True,
        apply_outlier_detection=True,
        apply_voxel_zscore=False,
        apply_smoothing=True,
        tr=2.0,
    )
    result = PreprocessBRAPHINData(input_bundle, cfg).run()
    assert "motion_correction" in result.applied_steps
    assert "slice_timing_correction" in result.applied_steps
    assert "outlier_detection" in result.applied_steps
    assert "spatial_smoothing" in result.applied_steps
    # Nothing should be pending any more
    assert result.pending_steps == []


def test_no_pending_steps_when_all_flags_false(preprocess_bundle):
    assert preprocess_bundle.pending_steps == []


def test_no_steps_applied_without_normalization(preprocess_bundle):
    # NaN replacement only appears in applied_steps if there were values to replace
    assert "global_intensity_normalization" not in preprocess_bundle.applied_steps


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

def test_none_bundle_raises():
    with pytest.raises(PreprocessingError):
        PreprocessBRAPHINData(BRAPHINInputBundle()).run()


def test_3d_array_raises():
    arr = np.ones((4, 4, 4), dtype=np.float32)
    img = nib.Nifti1Image(arr, np.eye(4))
    meta = get_nifti_metadata(img)
    # Patch ndim so it passes _validate_input_bundle but fails _validate_fmri_array
    # The easiest way: just make it 3D and see that _validate_fmri_array rejects it
    bundle = BRAPHINInputBundle(fmri_path="x.nii", fmri_image=img, fmri_metadata=meta)
    with pytest.raises(PreprocessingError):
        PreprocessBRAPHINData(bundle).run()
