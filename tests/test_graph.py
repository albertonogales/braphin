"""
Tests for braphin.graph.BRAPHINGraph

Covers:
- __init__: modality defaults to None
- load_data: fMRI path sets modality and metadata
- load_data: "mri" alias accepted
- load_data: unsupported modality raises ValueError
- modelate (fMRI): returns (G, ndarray)
- modelate (fMRI): matrix has shape (N_ROIS, N_ROIS)
- modelate (fMRI): ch_names populated after modelate
- modelate (fMRI): metadata["modelate_stage"] == "completed"
- modelate: unsupported modality raises ValueError
"""

import numpy as np
import networkx as nx
import pytest

from braphin.graph import BRAPHINGraph
from braphin import Graph  # re-export alias

N_ROIS = 4


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

def test_init_modality_is_none():
    g = BRAPHINGraph()
    assert g.modality is None


def test_graph_alias_is_braphing_raph():
    """Verify that the public `Graph` import is indeed BRAPHINGraph."""
    g = Graph()
    assert isinstance(g, BRAPHINGraph)


# ---------------------------------------------------------------------------
# load_data — fMRI file
# ---------------------------------------------------------------------------

def test_load_data_fmri_sets_modality(saved_fmri_path):
    g = BRAPHINGraph()
    g.load_data(str(saved_fmri_path), modality="fmri")
    assert g.modality == "fmri"


def test_load_data_mri_alias_sets_modality_fmri(saved_fmri_path):
    g = BRAPHINGraph()
    g.load_data(str(saved_fmri_path), modality="mri")
    assert g.modality == "fmri"


def test_load_data_fmri_ch_names_empty(saved_fmri_path):
    g = BRAPHINGraph()
    g.load_data(str(saved_fmri_path), modality="fmri")
    assert g.ch_names == []


def test_load_data_fmri_metadata_keys(saved_fmri_path):
    g = BRAPHINGraph()
    g.load_data(str(saved_fmri_path), modality="fmri")
    for key in ("fmri_path", "fmri_metadata", "auxiliary_files", "input_stage"):
        assert key in g.metadata


def test_load_data_fmri_metadata_input_stage(saved_fmri_path):
    g = BRAPHINGraph()
    g.load_data(str(saved_fmri_path), modality="fmri")
    assert g.metadata["input_stage"] == "import"


def test_load_data_fmri_data_is_input_bundle(saved_fmri_path):
    from braphin.importBRAPHINData import BRAPHINInputBundle
    g = BRAPHINGraph()
    g.load_data(str(saved_fmri_path), modality="fmri")
    assert isinstance(g.data, BRAPHINInputBundle)


# ---------------------------------------------------------------------------
# load_data — invalid modality
# ---------------------------------------------------------------------------

def test_load_data_unsupported_modality_raises(saved_fmri_path):
    g = BRAPHINGraph()
    with pytest.raises(ValueError, match="Unsupported modality"):
        g.load_data(str(saved_fmri_path), modality="pet")


# ---------------------------------------------------------------------------
# modelate — fMRI (inject data directly to avoid file dependency)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fmri_graph_result(input_bundle, atlas_array):
    """Run modelate once and cache results for all modelate tests."""
    g = BRAPHINGraph()
    g.modality = "fmri"
    g.data = input_bundle
    g.ch_names = []
    g.metadata = {}
    G, matrix = g.modelate(
        window_size=None,
        connectivity="pearson_correlation",
        atlas_data=atlas_array,
    )
    return g, G, matrix


def test_modelate_fmri_returns_tuple(fmri_graph_result):
    _, G, matrix = fmri_graph_result
    assert G is not None
    assert matrix is not None


def test_modelate_fmri_G_is_networkx_graph(fmri_graph_result):
    _, G, _ = fmri_graph_result
    assert isinstance(G, (nx.Graph, nx.DiGraph))


def test_modelate_fmri_matrix_is_ndarray(fmri_graph_result):
    _, _, matrix = fmri_graph_result
    assert isinstance(matrix, np.ndarray)


def test_modelate_fmri_matrix_shape(fmri_graph_result):
    _, _, matrix = fmri_graph_result
    assert matrix.shape == (N_ROIS, N_ROIS)


def test_modelate_fmri_ch_names_populated(fmri_graph_result):
    g, _, _ = fmri_graph_result
    assert len(g.ch_names) == N_ROIS


def test_modelate_fmri_metadata_stage_completed(fmri_graph_result):
    g, _, _ = fmri_graph_result
    assert g.metadata.get("modelate_stage") == "completed"


def test_modelate_fmri_matrix_diagonal_zeroed(fmri_graph_result):
    """After modelate the returned matrix should have zeros on the diagonal."""
    _, _, matrix = fmri_graph_result
    np.testing.assert_array_equal(np.diag(matrix), np.zeros(N_ROIS))


def test_modelate_fmri_matrix_no_nans(fmri_graph_result):
    _, _, matrix = fmri_graph_result
    assert not np.any(np.isnan(matrix))


# ---------------------------------------------------------------------------
# modelate — unsupported modality
# ---------------------------------------------------------------------------

def test_modelate_unsupported_modality_raises():
    g = BRAPHINGraph()
    g.modality = "pet"
    with pytest.raises(ValueError, match="Unsupported modality"):
        g.modelate(window_size=None, connectivity="pearson_correlation")
