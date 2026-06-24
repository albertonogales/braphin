"""
End-to-end integration tests for the full braphin pipeline.

Each test runs the complete chain:
  InputBRAPHINData → PreprocessBRAPHINData → DenoiseBRAPHINData → TransformBRAPHINData → ModelBRAPHINConnectivityData

Uses only synthetic data (no external files required).
"""

import dataclasses

import nibabel as nib
import numpy as np
import pytest

from braphin.config import (
    AtlasConfig,
    ConnectivityConfig,
    DenoiseConfig,
    PreprocessConfig,
)
from braphin.denoise import DenoiseBRAPHINData
from braphin.importBRAPHINData import InputBRAPHINData
from braphin.connectivity import ModelBRAPHINConnectivityData
from braphin.preprocess import PreprocessBRAPHINData
from braphin.transform import TransformBRAPHINData, build_synthetic_atlas

SPATIAL = (8, 8, 8)
T = 50
N_ROIS = 6


@pytest.fixture(scope="module")
def pipeline_fmri_path(tmp_path_factory):
    rng = np.random.default_rng(123)
    arr = (rng.random((*SPATIAL, T)) * 200.0 + 600.0).astype(np.float32)
    affine = np.diag([2.0, 2.0, 2.0, 1.0]).astype(np.float64)
    img = nib.Nifti1Image(arr, affine)
    img.header.set_zooms((2.0, 2.0, 2.0, 2.0))
    p = tmp_path_factory.mktemp("pipeline") / "fmri.nii.gz"
    nib.save(img, str(p))
    return p


# ---------------------------------------------------------------------------
# Full pipeline: no confounds, no normalization
# ---------------------------------------------------------------------------

def test_full_pipeline_completes(pipeline_fmri_path):
    input_bundle = InputBRAPHINData(pipeline_fmri_path).load()

    pp_bundle = PreprocessBRAPHINData(
        input_bundle,
        PreprocessConfig(
            apply_motion_correction=False, apply_slice_timing=False,
            apply_outlier_detection=False, apply_voxel_zscore=False, apply_smoothing=False,
        ),
    ).run()

    dn_bundle = DenoiseBRAPHINData(
        pp_bundle,
        DenoiseConfig(regress_confounds=False, apply_scrubbing=False, apply_bandpass=False),
    ).run()

    atlas = build_synthetic_atlas(SPATIAL, num_rois=N_ROIS)
    tx_bundle = TransformBRAPHINData(dn_bundle, atlas_data=atlas, config=AtlasConfig()).run()

    conn_bundle = ModelBRAPHINConnectivityData(
        tx_bundle, ConnectivityConfig(method="pearson_correlation"),
    ).run()

    m = conn_bundle.connectivity_matrix
    assert m.shape == (N_ROIS, N_ROIS)
    np.testing.assert_allclose(m, m.T, atol=1e-4)
    assert np.all(m >= -1.0 - 1e-4) and np.all(m <= 1.0 + 1e-4)


def test_pipeline_preprocessed_shape(pipeline_fmri_path):
    input_bundle = InputBRAPHINData(pipeline_fmri_path).load()
    pp_bundle = PreprocessBRAPHINData(
        input_bundle,
        PreprocessConfig(
            apply_motion_correction=False, apply_slice_timing=False,
            apply_outlier_detection=False, apply_voxel_zscore=False, apply_smoothing=False,
        ),
    ).run()
    assert pp_bundle.preprocessed_data.shape == (*SPATIAL, T)


def test_pipeline_roi_time_series_shape(pipeline_fmri_path):
    input_bundle = InputBRAPHINData(pipeline_fmri_path).load()
    pp_bundle = PreprocessBRAPHINData(
        input_bundle,
        PreprocessConfig(
            apply_motion_correction=False, apply_slice_timing=False,
            apply_outlier_detection=False, apply_voxel_zscore=False, apply_smoothing=False,
        ),
    ).run()
    dn_bundle = DenoiseBRAPHINData(
        pp_bundle,
        DenoiseConfig(regress_confounds=False, apply_scrubbing=False, apply_bandpass=False),
    ).run()
    atlas = build_synthetic_atlas(SPATIAL, num_rois=N_ROIS)
    tx_bundle = TransformBRAPHINData(dn_bundle, atlas_data=atlas, config=AtlasConfig()).run()
    assert tx_bundle.roi_time_series.shape == (N_ROIS, T)


# ---------------------------------------------------------------------------
# Full pipeline: with confound regression
# ---------------------------------------------------------------------------

def test_pipeline_with_confound_regression(pipeline_fmri_path):
    input_bundle = InputBRAPHINData(pipeline_fmri_path).load()

    pp_bundle = PreprocessBRAPHINData(
        input_bundle,
        PreprocessConfig(
            apply_motion_correction=False, apply_slice_timing=False,
            apply_outlier_detection=False, apply_voxel_zscore=False, apply_smoothing=False,
        ),
    ).run()

    rng = np.random.default_rng(0)
    confounds = rng.random((T, 6)).astype(np.float32)
    pp_bundle_conf = dataclasses.replace(
        pp_bundle,
        auxiliary_files={"confounds_timeseries.tsv": confounds},
    )

    dn_bundle = DenoiseBRAPHINData(
        pp_bundle_conf,
        DenoiseConfig(regress_confounds=True, apply_scrubbing=False, apply_bandpass=False),
    ).run()
    assert "confound_regression" in dn_bundle.applied_steps

    atlas = build_synthetic_atlas(SPATIAL, num_rois=4)
    tx_bundle = TransformBRAPHINData(dn_bundle, atlas_data=atlas, config=AtlasConfig()).run()
    conn_bundle = ModelBRAPHINConnectivityData(tx_bundle, ConnectivityConfig()).run()
    assert conn_bundle.connectivity_matrix.shape == (4, 4)
    assert not np.any(np.isnan(conn_bundle.connectivity_matrix))


# ---------------------------------------------------------------------------
# Full pipeline: with normalization
# ---------------------------------------------------------------------------

def test_pipeline_with_normalization(pipeline_fmri_path):
    input_bundle = InputBRAPHINData(pipeline_fmri_path).load()
    pp_bundle = PreprocessBRAPHINData(
        input_bundle,
        PreprocessConfig(
            apply_motion_correction=False, apply_slice_timing=False,
            apply_outlier_detection=False, apply_voxel_zscore=True, apply_smoothing=False,
        ),
    ).run()
    assert "per_voxel_temporal_zscore" in pp_bundle.applied_steps

    dn_bundle = DenoiseBRAPHINData(
        pp_bundle,
        DenoiseConfig(regress_confounds=False, apply_scrubbing=False, apply_bandpass=False),
    ).run()
    atlas = build_synthetic_atlas(SPATIAL, num_rois=N_ROIS)
    tx_bundle = TransformBRAPHINData(dn_bundle, atlas_data=atlas, config=AtlasConfig()).run()
    conn_bundle = ModelBRAPHINConnectivityData(tx_bundle, ConnectivityConfig()).run()
    assert conn_bundle.connectivity_matrix.shape == (N_ROIS, N_ROIS)


# ---------------------------------------------------------------------------
# Full pipeline: all three connectivity methods
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", [
    "pearson_correlation",
    "cross_correlation",
    "corr_cross_correlation",
])
def test_pipeline_all_connectivity_methods(pipeline_fmri_path, method):
    input_bundle = InputBRAPHINData(pipeline_fmri_path).load()
    pp_bundle = PreprocessBRAPHINData(
        input_bundle,
        PreprocessConfig(
            apply_motion_correction=False, apply_slice_timing=False,
            apply_outlier_detection=False, apply_voxel_zscore=False, apply_smoothing=False,
        ),
    ).run()
    dn_bundle = DenoiseBRAPHINData(
        pp_bundle,
        DenoiseConfig(regress_confounds=False, apply_scrubbing=False, apply_bandpass=False),
    ).run()
    atlas = build_synthetic_atlas(SPATIAL, num_rois=N_ROIS)
    tx_bundle = TransformBRAPHINData(dn_bundle, atlas_data=atlas, config=AtlasConfig()).run()
    conn_bundle = ModelBRAPHINConnectivityData(
        tx_bundle, ConnectivityConfig(method=method),
    ).run()
    assert conn_bundle.connectivity_matrix.shape == (N_ROIS, N_ROIS)
    assert not np.any(np.isnan(conn_bundle.connectivity_matrix))


# ---------------------------------------------------------------------------
# Metadata propagation
# ---------------------------------------------------------------------------

def test_fmri_path_propagates_through_pipeline(pipeline_fmri_path):
    input_bundle = InputBRAPHINData(pipeline_fmri_path).load()
    pp_bundle = PreprocessBRAPHINData(
        input_bundle,
        PreprocessConfig(
            apply_motion_correction=False, apply_slice_timing=False,
            apply_outlier_detection=False, apply_voxel_zscore=False, apply_smoothing=False,
        ),
    ).run()
    dn_bundle = DenoiseBRAPHINData(
        pp_bundle,
        DenoiseConfig(regress_confounds=False, apply_scrubbing=False, apply_bandpass=False),
    ).run()
    atlas = build_synthetic_atlas(SPATIAL, num_rois=N_ROIS)
    tx_bundle = TransformBRAPHINData(dn_bundle, atlas_data=atlas, config=AtlasConfig()).run()
    conn_bundle = ModelBRAPHINConnectivityData(tx_bundle, ConnectivityConfig()).run()

    assert pp_bundle.fmri_path == str(pipeline_fmri_path)
    assert dn_bundle.fmri_path == str(pipeline_fmri_path)
    assert tx_bundle.fmri_path == str(pipeline_fmri_path)
    assert conn_bundle.fmri_path == str(pipeline_fmri_path)
