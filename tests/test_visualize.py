"""
Tests for braphin.visualize.build_fmri_graph and layout helpers

Covers:
- Returns a NetworkX graph
- Node count equals matrix N
- Directed graph when matrix is asymmetric
- Undirected graph when matrix is symmetric
- ROI label nodes when roi_labels supplied
- Integer nodes when roi_labels=None
- G.graph["modality"] == "fmri"
- G.graph["projection"] is set
- Edge "thickness" attribute set for all edges
- Invalid (non-square) matrix raises ValueError
- projection="axial" works
- projection="coronal" works
- projection="sagittal" works
- Unsupported projection raises ValueError
- With roi_centroids_3d: nodes get "pos" and "depth" attributes
- With roi_centroids_3d: nodes get "pos3d" attribute
- Without centroids: nodes have no "pos" attribute
- _infer_lr_partner identifies _L <-> _R pairs
- _infer_lr_partner returns None for unpaired labels
"""

import numpy as np
import networkx as nx
import pytest

from braphin.visualize import build_fmri_graph, _infer_lr_partner

N = 4

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def sym_matrix():
    """Symmetric 4×4 connectivity matrix (Pearson-like, diagonal=0)."""
    rng = np.random.default_rng(1)
    m = rng.random((N, N)).astype(np.float32)
    m = (m + m.T) / 2.0
    np.fill_diagonal(m, 0.0)
    return m


@pytest.fixture
def asym_matrix():
    """Asymmetric 4×4 connectivity matrix."""
    rng = np.random.default_rng(2)
    m = rng.random((N, N)).astype(np.float32)
    np.fill_diagonal(m, 0.0)
    # Force asymmetry
    m[0, 1] = 0.9
    m[1, 0] = 0.1
    return m


@pytest.fixture
def roi_labels():
    return ["ROI_L", "ROI_R", "Back_L", "Back_R"]


@pytest.fixture
def roi_centroids():
    return {
        "ROI_L":  (-20.0,  30.0, 10.0),
        "ROI_R":  ( 20.0,  30.0, 10.0),
        "Back_L": (-15.0, -40.0,  5.0),
        "Back_R": ( 15.0, -40.0,  5.0),
    }


# ---------------------------------------------------------------------------
# Basic return type and structure
# ---------------------------------------------------------------------------

def test_returns_networkx_graph(sym_matrix):
    G = build_fmri_graph(sym_matrix)
    assert isinstance(G, (nx.Graph, nx.DiGraph))


def test_node_count_equals_matrix_n(sym_matrix):
    G = build_fmri_graph(sym_matrix)
    assert G.number_of_nodes() == N


def test_symmetric_matrix_produces_undirected_graph(sym_matrix):
    G = build_fmri_graph(sym_matrix)
    assert not G.is_directed()


def test_asymmetric_matrix_produces_directed_graph(asym_matrix):
    G = build_fmri_graph(asym_matrix)
    assert G.is_directed()


# ---------------------------------------------------------------------------
# Node labelling
# ---------------------------------------------------------------------------

def test_integer_nodes_when_no_labels(sym_matrix):
    G = build_fmri_graph(sym_matrix)
    for node in G.nodes():
        assert isinstance(node, (int, np.integer))


def test_string_nodes_when_labels_supplied(sym_matrix, roi_labels):
    G = build_fmri_graph(sym_matrix, roi_labels=roi_labels)
    for node in G.nodes():
        assert isinstance(node, str)


def test_label_nodes_match_roi_labels(sym_matrix, roi_labels):
    G = build_fmri_graph(sym_matrix, roi_labels=roi_labels)
    assert set(G.nodes()) == set(roi_labels)


# ---------------------------------------------------------------------------
# Graph-level metadata attributes
# ---------------------------------------------------------------------------

def test_graph_modality_is_fmri(sym_matrix):
    G = build_fmri_graph(sym_matrix)
    assert G.graph["modality"] == "fmri"


def test_graph_projection_is_set_coronal(sym_matrix):
    G = build_fmri_graph(sym_matrix, projection="coronal")
    assert G.graph["projection"] == "coronal"


def test_graph_projection_is_set_axial(sym_matrix):
    G = build_fmri_graph(sym_matrix, projection="axial")
    assert G.graph["projection"] == "axial"


def test_graph_projection_is_set_sagittal(sym_matrix):
    G = build_fmri_graph(sym_matrix, projection="sagittal")
    assert G.graph["projection"] == "sagittal"


# ---------------------------------------------------------------------------
# Edge attributes
# ---------------------------------------------------------------------------

def test_all_edges_have_thickness(sym_matrix):
    G = build_fmri_graph(sym_matrix)
    for u, v, data in G.edges(data=True):
        assert "thickness" in data
        assert data["thickness"] >= 0.5


# ---------------------------------------------------------------------------
# Invalid input
# ---------------------------------------------------------------------------

def test_non_square_matrix_raises_value_error():
    bad = np.random.default_rng(3).random((3, 4)).astype(np.float32)
    with pytest.raises(ValueError, match="square"):
        build_fmri_graph(bad)


def test_unsupported_projection_raises_value_error(sym_matrix, roi_labels, roi_centroids):
    with pytest.raises(ValueError, match="projection"):
        build_fmri_graph(
            sym_matrix,
            roi_labels=roi_labels,
            roi_centroids_3d=roi_centroids,
            projection="lateral",
        )


# ---------------------------------------------------------------------------
# Projections with centroids
# ---------------------------------------------------------------------------

def test_axial_projection_sets_pos(sym_matrix, roi_labels, roi_centroids):
    G = build_fmri_graph(
        sym_matrix,
        roi_labels=roi_labels,
        roi_centroids_3d=roi_centroids,
        projection="axial",
    )
    for node in G.nodes():
        assert "pos" in G.nodes[node]


def test_axial_projection_sets_depth(sym_matrix, roi_labels, roi_centroids):
    G = build_fmri_graph(
        sym_matrix,
        roi_labels=roi_labels,
        roi_centroids_3d=roi_centroids,
        projection="axial",
    )
    for node in G.nodes():
        assert "depth" in G.nodes[node]


def test_axial_projection_sets_pos3d(sym_matrix, roi_labels, roi_centroids):
    G = build_fmri_graph(
        sym_matrix,
        roi_labels=roi_labels,
        roi_centroids_3d=roi_centroids,
        projection="axial",
    )
    for node in G.nodes():
        assert "pos3d" in G.nodes[node]


def test_coronal_projection_sets_pos(sym_matrix, roi_labels, roi_centroids):
    G = build_fmri_graph(
        sym_matrix,
        roi_labels=roi_labels,
        roi_centroids_3d=roi_centroids,
        projection="coronal",
    )
    for node in G.nodes():
        assert "pos" in G.nodes[node]


def test_sagittal_projection_sets_pos(sym_matrix, roi_labels, roi_centroids):
    G = build_fmri_graph(
        sym_matrix,
        roi_labels=roi_labels,
        roi_centroids_3d=roi_centroids,
        projection="sagittal",
    )
    for node in G.nodes():
        assert "pos" in G.nodes[node]


def test_without_centroids_no_pos_attribute(sym_matrix):
    G = build_fmri_graph(sym_matrix)
    for node in G.nodes():
        assert "pos" not in G.nodes[node]


# ---------------------------------------------------------------------------
# _infer_lr_partner
# ---------------------------------------------------------------------------

def test_infer_lr_partner_l_to_r():
    assert _infer_lr_partner("Precentral_L") == "Precentral_R"


def test_infer_lr_partner_r_to_l():
    assert _infer_lr_partner("Precentral_R") == "Precentral_L"


def test_infer_lr_partner_left_to_right():
    assert _infer_lr_partner("Region_Left") == "Region_Right"


def test_infer_lr_partner_right_to_left():
    assert _infer_lr_partner("Region_Right") == "Region_Left"


def test_infer_lr_partner_unpaired_returns_none():
    assert _infer_lr_partner("ROI_1") is None


def test_infer_lr_partner_no_suffix_returns_none():
    assert _infer_lr_partner("Cerebellum") is None


# ---------------------------------------------------------------------------
# Automatic fallback layout (no _L/_R names — uses proximity matching)
# ---------------------------------------------------------------------------

@pytest.fixture
def anonymous_matrix():
    """Symmetric 4×4 matrix for nodes without L/R name suffixes."""
    rng = np.random.default_rng(10)
    m = rng.random((4, 4)).astype(np.float32)
    m = (m + m.T) / 2.0
    np.fill_diagonal(m, 0.0)
    return m


@pytest.fixture
def anonymous_labels():
    """ROI labels with no _L/_R suffix — triggers automatic fallback matcher."""
    return ["Node1", "Node2", "Node3", "Node4"]


@pytest.fixture
def anonymous_centroids():
    """Centroids without _L/_R names: negative x for 'left', positive for 'right'."""
    return {
        "Node1": (-25.0,  10.0, 20.0),
        "Node2": ( 25.0,  10.0, 20.0),
        "Node3": (-18.0, -20.0, 15.0),
        "Node4": ( 18.0, -20.0, 15.0),
    }


def test_axial_fallback_sets_pos(anonymous_matrix, anonymous_labels, anonymous_centroids):
    G = build_fmri_graph(
        anonymous_matrix,
        roi_labels=anonymous_labels,
        roi_centroids_3d=anonymous_centroids,
        projection="axial",
    )
    for node in G.nodes():
        assert "pos" in G.nodes[node]


def test_coronal_fallback_sets_pos(anonymous_matrix, anonymous_labels, anonymous_centroids):
    G = build_fmri_graph(
        anonymous_matrix,
        roi_labels=anonymous_labels,
        roi_centroids_3d=anonymous_centroids,
        projection="coronal",
    )
    for node in G.nodes():
        assert "pos" in G.nodes[node]


def test_axial_fallback_left_node_has_negative_x(anonymous_matrix, anonymous_labels, anonymous_centroids):
    """After symmetry adjustment, the left node should have a negative x position."""
    G = build_fmri_graph(
        anonymous_matrix,
        roi_labels=anonymous_labels,
        roi_centroids_3d=anonymous_centroids,
        projection="axial",
    )
    # Node1 is in the left hemisphere (original x < 0) → pos.x should be <= 0
    pos = G.nodes["Node1"]["pos"]
    assert pos[0] <= 0


def test_coronal_fallback_sets_depth(anonymous_matrix, anonymous_labels, anonymous_centroids):
    G = build_fmri_graph(
        anonymous_matrix,
        roi_labels=anonymous_labels,
        roi_centroids_3d=anonymous_centroids,
        projection="coronal",
    )
    for node in G.nodes():
        assert "depth" in G.nodes[node]
