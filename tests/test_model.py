"""
Tests for braphin.model.ModelMRIData

Covers:
- __init__: all bundles default to None
- __init__: ch_names defaults to []
- __init__: connectivity defaults to "pearson_correlation"
- _validate_input_bundle: raises ValueError when data is None
- _validate_input_bundle: raises ValueError when data missing fmri_image
- _validate_input_bundle: raises ValueError when data missing fmri_metadata
- _normalize_connectivity_name: returns "pearson_correlation" for None
- _normalize_connectivity_name: normalises alias "pearson"
- _normalize_connectivity_name: strips whitespace and lowercases
- _normalize_connectivity_name: unknown method returned unchanged
- _build_connectivity_config: returns ConnectivityConfig with method set
- _build_connectivity_config: respects existing connectivity_config
- _build_connectivity_config: sets window_size and threshold
- connectivity_workflow: returns (G, ndarray)
- connectivity_workflow: G is NetworkX Graph
- connectivity_workflow: matrix shape (N_ROIS, N_ROIS)
- connectivity_workflow: diagonal zeroed
- connectivity_workflow: ch_names populated after workflow
- connectivity_workflow: all intermediate bundles set
- display_info: works after workflow
- display_info: raises ValueError before workflow
"""

import numpy as np
import networkx as nx
import pytest

from braphin.model import ModelMRIData
from braphin.config import AtlasConfig, ConnectivityConfig

N_ROIS = 4


# ---------------------------------------------------------------------------
# __init__ defaults
# ---------------------------------------------------------------------------

def test_init_bundles_all_none(input_bundle):
    m = ModelMRIData(data=input_bundle)
    assert m.input_bundle is None
    assert m.preprocess_bundle is None
    assert m.denoise_bundle is None
    assert m.transform_bundle is None
    assert m.connectivity_bundle is None


def test_init_ch_names_default_empty(input_bundle):
    m = ModelMRIData(data=input_bundle)
    assert m.ch_names == []


def test_init_connectivity_default(input_bundle):
    m = ModelMRIData(data=input_bundle)
    assert m.connectivity == "pearson_correlation"


# ---------------------------------------------------------------------------
# _validate_input_bundle
# ---------------------------------------------------------------------------

def test_validate_raises_when_data_none():
    m = ModelMRIData(data=None)
    with pytest.raises(ValueError, match="no input data"):
        m._validate_input_bundle()


def test_validate_raises_when_no_fmri_image():
    class FakeBundleNoImage:
        fmri_metadata = {}

    m = ModelMRIData(data=FakeBundleNoImage())
    with pytest.raises(ValueError, match="fmri_image"):
        m._validate_input_bundle()


def test_validate_raises_when_no_fmri_metadata():
    class FakeBundleNoMeta:
        fmri_image = object()

    m = ModelMRIData(data=FakeBundleNoMeta())
    with pytest.raises(ValueError, match="fmri_metadata"):
        m._validate_input_bundle()


def test_validate_passes_with_valid_bundle(input_bundle):
    m = ModelMRIData(data=input_bundle)
    # Should not raise
    m._validate_input_bundle()


# ---------------------------------------------------------------------------
# _normalize_connectivity_name
# ---------------------------------------------------------------------------

def test_normalize_none_returns_pearson(input_bundle):
    m = ModelMRIData(data=input_bundle)
    assert m._normalize_connectivity_name(None) == "pearson_correlation"


def test_normalize_pearson_alias(input_bundle):
    m = ModelMRIData(data=input_bundle)
    assert m._normalize_connectivity_name("pearson") == "pearson_correlation"


def test_normalize_strips_and_lowercases(input_bundle):
    m = ModelMRIData(data=input_bundle)
    result = m._normalize_connectivity_name("  Pearson  ")
    assert result == "pearson_correlation"


def test_normalize_unknown_method_unchanged(input_bundle):
    m = ModelMRIData(data=input_bundle)
    result = m._normalize_connectivity_name("my_custom_method")
    assert result == "my_custom_method"


# ---------------------------------------------------------------------------
# _build_connectivity_config
# ---------------------------------------------------------------------------

def test_build_config_returns_connectivity_config(input_bundle):
    m = ModelMRIData(data=input_bundle, connectivity="pearson_correlation")
    cfg = m._build_connectivity_config(window_size=None, threshold=None)
    assert isinstance(cfg, ConnectivityConfig)


def test_build_config_method_set(input_bundle):
    m = ModelMRIData(data=input_bundle, connectivity="pearson_correlation")
    cfg = m._build_connectivity_config(window_size=None, threshold=None)
    assert cfg.method == "pearson_correlation"


def test_build_config_respects_existing_connectivity_config(input_bundle):
    existing = ConnectivityConfig(method="cross_correlation")
    m = ModelMRIData(
        data=input_bundle,
        connectivity="pearson_correlation",
        connectivity_config=existing,
    )
    cfg = m._build_connectivity_config(window_size=None, threshold=None)
    # connectivity name is always overwritten from self.connectivity
    assert cfg is existing


def test_build_config_sets_window_size(input_bundle):
    m = ModelMRIData(data=input_bundle)
    cfg = m._build_connectivity_config(window_size=10.0, threshold=None)
    assert cfg.window_size == 10.0


def test_build_config_sets_threshold(input_bundle):
    m = ModelMRIData(data=input_bundle)
    cfg = m._build_connectivity_config(window_size=None, threshold=0.3)
    assert cfg.threshold == 0.3


# ---------------------------------------------------------------------------
# connectivity_workflow (session-scoped result for speed)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def workflow_result(input_bundle, atlas_array):
    m = ModelMRIData(
        data=input_bundle,
        connectivity="pearson_correlation",
        atlas_data=atlas_array,
    )
    G, matrix = m.connectivity_workflow(window_size=None, threshold=None)
    return m, G, matrix


def test_workflow_returns_two_items(workflow_result):
    _, G, matrix = workflow_result
    assert G is not None
    assert matrix is not None


def test_workflow_G_is_networkx_graph(workflow_result):
    _, G, _ = workflow_result
    assert isinstance(G, (nx.Graph, nx.DiGraph))


def test_workflow_matrix_is_ndarray(workflow_result):
    _, _, matrix = workflow_result
    assert isinstance(matrix, np.ndarray)


def test_workflow_matrix_shape(workflow_result):
    _, _, matrix = workflow_result
    assert matrix.shape == (N_ROIS, N_ROIS)


def test_workflow_matrix_diagonal_zeroed(workflow_result):
    _, _, matrix = workflow_result
    np.testing.assert_array_equal(np.diag(matrix), np.zeros(N_ROIS))


def test_workflow_ch_names_populated(workflow_result):
    m, _, _ = workflow_result
    assert len(m.ch_names) == N_ROIS


def test_workflow_input_bundle_set(workflow_result):
    m, _, _ = workflow_result
    assert m.input_bundle is not None


def test_workflow_preprocess_bundle_set(workflow_result):
    m, _, _ = workflow_result
    assert m.preprocess_bundle is not None


def test_workflow_denoise_bundle_set(workflow_result):
    m, _, _ = workflow_result
    assert m.denoise_bundle is not None


def test_workflow_transform_bundle_set(workflow_result):
    m, _, _ = workflow_result
    assert m.transform_bundle is not None


def test_workflow_connectivity_bundle_set(workflow_result):
    m, _, _ = workflow_result
    assert m.connectivity_bundle is not None


# ---------------------------------------------------------------------------
# display_info
# ---------------------------------------------------------------------------

def test_display_info_works_after_workflow(workflow_result, capsys):
    m, _, _ = workflow_result
    m.display_info()
    captured = capsys.readouterr()
    assert "fMRI" in captured.out or "connectivity" in captured.out.lower()


def test_display_info_raises_before_workflow(input_bundle):
    m = ModelMRIData(data=input_bundle)
    with pytest.raises(ValueError, match="connectivity_workflow"):
        m.display_info()


# ---------------------------------------------------------------------------
# connectivity_workflow with bands parameter
# ---------------------------------------------------------------------------

def test_workflow_with_bands_populates_band_connectivity(input_bundle, atlas_array):
    """Passing a valid fMRI band list should populate self.band_connectivity."""
    m = ModelMRIData(
        data=input_bundle,
        connectivity="pearson_correlation",
        atlas_data=atlas_array,
    )
    m.connectivity_workflow(window_size=None, threshold=None, bands=["slow4"])
    assert hasattr(m, "band_connectivity")
    assert "slow4" in m.band_connectivity


def test_workflow_with_bands_matrix_shape(input_bundle, atlas_array):
    """Band-connectivity matrices should be (N_ROIS × N_ROIS)."""
    m = ModelMRIData(
        data=input_bundle,
        connectivity="pearson_correlation",
        atlas_data=atlas_array,
    )
    m.connectivity_workflow(window_size=None, threshold=None, bands=["slow4"])
    mat = m.band_connectivity["slow4"]
    assert mat.shape == (N_ROIS, N_ROIS)
