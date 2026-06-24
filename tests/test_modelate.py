"""
Tests for braphin/connectivity.py

Covers:
- Bundle returned with correct shape and type
- Pearson matrix is symmetric
- Threshold is applied
- Cross-correlation and corrected cross-correlation run end-to-end
- Connectivity metadata fields
"""

import numpy as np
import pytest

from braphin.config import AtlasConfig, ConnectivityConfig
from braphin.connectivity import BRAPHINConnectivityBundle, ModelBRAPHINConnectivityData
from braphin.exceptions import ConnectivityError
from braphin.transform import TransformBRAPHINData

N_ROIS = 4   # matches conftest atlas_array


@pytest.fixture(scope="module")
def transform_bundle(denoise_bundle, atlas_array):
    t = TransformBRAPHINData(denoise_bundle, atlas_data=atlas_array, config=AtlasConfig())
    return t.run()


# ---------------------------------------------------------------------------
# Basic output
# ---------------------------------------------------------------------------

def test_modelate_returns_bundle(transform_bundle):
    cfg = ConnectivityConfig(method="pearson_correlation")
    result = ModelBRAPHINConnectivityData(transform_bundle, cfg).run()
    assert isinstance(result, BRAPHINConnectivityBundle)


def test_connectivity_matrix_shape(transform_bundle):
    cfg = ConnectivityConfig(method="pearson_correlation")
    result = ModelBRAPHINConnectivityData(transform_bundle, cfg).run()
    assert result.connectivity_matrix.shape == (N_ROIS, N_ROIS)


def test_connectivity_matrix_dtype(transform_bundle):
    cfg = ConnectivityConfig(method="pearson_correlation")
    result = ModelBRAPHINConnectivityData(transform_bundle, cfg).run()
    assert result.connectivity_matrix.dtype == np.float32


# ---------------------------------------------------------------------------
# Pearson properties
# ---------------------------------------------------------------------------

def test_pearson_symmetric(transform_bundle):
    cfg = ConnectivityConfig(method="pearson_correlation")
    result = ModelBRAPHINConnectivityData(transform_bundle, cfg).run()
    m = result.connectivity_matrix
    np.testing.assert_allclose(m, m.T, atol=1e-4)


def test_pearson_diagonal_ones(transform_bundle):
    cfg = ConnectivityConfig(method="pearson_correlation")
    result = ModelBRAPHINConnectivityData(transform_bundle, cfg).run()
    np.testing.assert_allclose(np.diag(result.connectivity_matrix), np.ones(N_ROIS), atol=1e-4)


def test_pearson_values_in_range(transform_bundle):
    cfg = ConnectivityConfig(method="pearson_correlation")
    result = ModelBRAPHINConnectivityData(transform_bundle, cfg).run()
    m = result.connectivity_matrix
    assert np.all(m >= -1.0 - 1e-4) and np.all(m <= 1.0 + 1e-4)


# ---------------------------------------------------------------------------
# Threshold
# ---------------------------------------------------------------------------

def test_threshold_applied_step_recorded(transform_bundle):
    cfg = ConnectivityConfig(method="pearson_correlation", threshold=0.5)
    result = ModelBRAPHINConnectivityData(transform_bundle, cfg).run()
    assert "threshold" in result.applied_steps


def test_threshold_zeroes_small_values(transform_bundle):
    cfg = ConnectivityConfig(method="pearson_correlation", threshold=0.5)
    result = ModelBRAPHINConnectivityData(transform_bundle, cfg).run()
    m = result.connectivity_matrix
    n = m.shape[0]
    for i in range(n):
        for j in range(n):
            if i != j:
                assert abs(m[i, j]) >= 0.5 or m[i, j] == pytest.approx(0.0, abs=1e-6)


def test_no_threshold_step_when_none(transform_bundle):
    cfg = ConnectivityConfig(method="pearson_correlation", threshold=None)
    result = ModelBRAPHINConnectivityData(transform_bundle, cfg).run()
    assert "threshold" not in result.applied_steps


# ---------------------------------------------------------------------------
# Other connectivity methods
# ---------------------------------------------------------------------------

def test_cross_correlation_runs(transform_bundle):
    cfg = ConnectivityConfig(method="cross_correlation")
    result = ModelBRAPHINConnectivityData(transform_bundle, cfg).run()
    assert result.connectivity_matrix.shape == (N_ROIS, N_ROIS)
    assert not np.any(np.isnan(result.connectivity_matrix))


def test_corrected_cross_correlation_runs(transform_bundle):
    cfg = ConnectivityConfig(method="corr_cross_correlation")
    result = ModelBRAPHINConnectivityData(transform_bundle, cfg).run()
    assert result.connectivity_matrix.shape == (N_ROIS, N_ROIS)
    assert not np.any(np.isnan(result.connectivity_matrix))


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def test_metadata_method_recorded(transform_bundle):
    cfg = ConnectivityConfig(method="pearson_correlation")
    result = ModelBRAPHINConnectivityData(transform_bundle, cfg).run()
    assert result.connectivity_metadata["method"] == "pearson_correlation"


def test_metadata_num_rois(transform_bundle):
    cfg = ConnectivityConfig(method="pearson_correlation")
    result = ModelBRAPHINConnectivityData(transform_bundle, cfg).run()
    assert result.connectivity_metadata["num_rois"] == N_ROIS


def test_metadata_symmetry_flag(transform_bundle):
    cfg = ConnectivityConfig(method="pearson_correlation")
    result = ModelBRAPHINConnectivityData(transform_bundle, cfg).run()
    assert result.connectivity_metadata["matrix_is_symmetric"] is True


def test_metadata_diagonal_flag(transform_bundle):
    cfg = ConnectivityConfig(method="pearson_correlation")
    result = ModelBRAPHINConnectivityData(transform_bundle, cfg).run()
    assert result.connectivity_metadata["diagonal_all_ones"] is True
