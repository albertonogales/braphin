"""
Tests for braphin/graph_metrics.py

Covers:
- build_graph_from_matrix: node count, edge count, threshold, directed flag
- compute_graph_metrics: presence of all keys, value types and ranges
- compute_modularity: range, empty graph, disconnected graph
- compute_rich_club_coefficient: type, non-negative values
- compute_metrics_all: batch wrapper
- Node-level metrics: correct node set, numeric values
"""

import math

import networkx as nx
import numpy as np
import pytest

from braphin.graph_metrics import (
    build_graph_from_matrix,
    compute_graph_metrics,
    compute_metrics_all,
    compute_modularity,
    compute_rich_club_coefficient,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N = 8
RNG = np.random.default_rng(42)


@pytest.fixture
def random_matrix():
    """Symmetric connectivity matrix with positive weights."""
    m = RNG.random((N, N)).astype(np.float32)
    m = (m + m.T) / 2          # symmetrise
    np.fill_diagonal(m, 1.0)
    return m


@pytest.fixture
def simple_graph():
    """Small fully-connected undirected graph, easy to reason about."""
    G = nx.complete_graph(6)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
    return G


@pytest.fixture
def disconnected_graph():
    G = nx.Graph()
    G.add_nodes_from(range(6))
    G.add_edge(0, 1, weight=0.8)
    G.add_edge(2, 3, weight=0.7)
    G.add_edge(4, 5, weight=0.9)
    return G


@pytest.fixture
def empty_graph():
    return nx.Graph()


# ---------------------------------------------------------------------------
# build_graph_from_matrix
# ---------------------------------------------------------------------------

def test_build_graph_node_count(random_matrix):
    G = build_graph_from_matrix(random_matrix)
    assert G.number_of_nodes() == N


def test_build_graph_custom_labels(random_matrix):
    labels = [f"R{i}" for i in range(N)]
    G = build_graph_from_matrix(random_matrix, roi_labels=labels)
    assert set(G.nodes()) == set(labels)


def test_build_graph_no_self_loops(random_matrix):
    G = build_graph_from_matrix(random_matrix)
    assert nx.number_of_selfloops(G) == 0


def test_build_graph_threshold(random_matrix):
    G_all = build_graph_from_matrix(random_matrix, threshold=0.0)
    G_thresh = build_graph_from_matrix(random_matrix, threshold=0.9)
    assert G_thresh.number_of_edges() <= G_all.number_of_edges()


def test_build_graph_directed(random_matrix):
    G = build_graph_from_matrix(random_matrix, directed=True)
    assert G.is_directed()


def test_build_graph_undirected(random_matrix):
    G = build_graph_from_matrix(random_matrix, directed=False)
    assert not G.is_directed()


def test_build_graph_default_labels(random_matrix):
    G = build_graph_from_matrix(random_matrix)
    assert "ROI_0" in G.nodes()
    assert f"ROI_{N - 1}" in G.nodes()


# ---------------------------------------------------------------------------
# compute_graph_metrics — key presence and types
# ---------------------------------------------------------------------------

EXPECTED_SCALAR_KEYS = [
    "density",
    "transitivity",
    "average_clustering",
    "global_efficiency",
    "local_efficiency",
    "average_path_length",
    "degree_assortativity",
    "small_world_sigma",
    "modularity",
]

EXPECTED_DICT_KEYS = [
    "degree_centrality",
    "degree",
    "node_strength",
    "betweenness_centrality",
    "eigenvector_centrality",
    "closeness_centrality",
    "rich_club_coefficient",
]


def test_metrics_has_all_scalar_keys(simple_graph):
    m = compute_graph_metrics(simple_graph)
    for key in EXPECTED_SCALAR_KEYS:
        assert key in m, f"Missing key: {key}"


def test_metrics_has_all_dict_keys(simple_graph):
    m = compute_graph_metrics(simple_graph)
    for key in EXPECTED_DICT_KEYS:
        assert key in m, f"Missing key: {key}"


def test_metrics_scalars_are_numeric(simple_graph):
    m = compute_graph_metrics(simple_graph)
    for key in EXPECTED_SCALAR_KEYS:
        val = m[key]
        assert isinstance(val, (int, float)), f"{key} should be numeric, got {type(val)}"


def test_metrics_dict_values_are_dicts(simple_graph):
    m = compute_graph_metrics(simple_graph)
    for key in EXPECTED_DICT_KEYS:
        assert isinstance(m[key], dict), f"{key} should be a dict"


def test_metrics_node_dicts_have_all_nodes(simple_graph):
    m = compute_graph_metrics(simple_graph)
    nodes = set(simple_graph.nodes())
    for key in ("degree_centrality", "betweenness_centrality",
                "eigenvector_centrality", "closeness_centrality",
                "node_strength", "degree"):
        assert set(m[key].keys()) == nodes, f"{key} missing nodes"


# ---------------------------------------------------------------------------
# compute_graph_metrics — value ranges
# ---------------------------------------------------------------------------

def test_density_range(simple_graph):
    m = compute_graph_metrics(simple_graph)
    assert 0.0 <= m["density"] <= 1.0


def test_transitivity_range(simple_graph):
    m = compute_graph_metrics(simple_graph)
    assert 0.0 <= m["transitivity"] <= 1.0


def test_average_clustering_range(simple_graph):
    m = compute_graph_metrics(simple_graph)
    assert 0.0 <= m["average_clustering"] <= 1.0


def test_global_efficiency_range(simple_graph):
    m = compute_graph_metrics(simple_graph)
    assert 0.0 <= m["global_efficiency"] <= 1.0


def test_local_efficiency_range(simple_graph):
    m = compute_graph_metrics(simple_graph)
    assert 0.0 <= m["local_efficiency"] <= 1.0


def test_average_path_length_positive(simple_graph):
    m = compute_graph_metrics(simple_graph)
    assert m["average_path_length"] >= 1.0


def test_modularity_range(simple_graph):
    m = compute_graph_metrics(simple_graph)
    assert 0.0 <= m["modularity"] <= 1.0


def test_degree_centrality_range(simple_graph):
    m = compute_graph_metrics(simple_graph)
    for v in m["degree_centrality"].values():
        assert 0.0 <= v <= 1.0


def test_betweenness_centrality_range(simple_graph):
    m = compute_graph_metrics(simple_graph)
    for v in m["betweenness_centrality"].values():
        assert 0.0 <= v <= 1.0


def test_eigenvector_centrality_range(simple_graph):
    m = compute_graph_metrics(simple_graph)
    for v in m["eigenvector_centrality"].values():
        assert 0.0 <= v <= 1.0 or math.isnan(v)


def test_node_strength_non_negative(simple_graph):
    m = compute_graph_metrics(simple_graph)
    for v in m["node_strength"].values():
        assert v >= 0.0


# ---------------------------------------------------------------------------
# compute_modularity (standalone)
# ---------------------------------------------------------------------------

def test_compute_modularity_complete_graph(simple_graph):
    q = compute_modularity(simple_graph)
    assert isinstance(q, float)
    assert 0.0 <= q <= 1.0


def test_compute_modularity_empty_graph(empty_graph):
    q = compute_modularity(empty_graph)
    assert q == 0.0


def test_compute_modularity_disconnected_graph(disconnected_graph):
    q = compute_modularity(disconnected_graph)
    assert isinstance(q, float)
    # Disconnected graph should have positive modularity
    assert q >= 0.0


# ---------------------------------------------------------------------------
# compute_rich_club_coefficient (standalone)
# ---------------------------------------------------------------------------

def test_rich_club_is_dict(simple_graph):
    rcc = compute_rich_club_coefficient(simple_graph)
    assert isinstance(rcc, dict)


def test_rich_club_non_negative(simple_graph):
    rcc = compute_rich_club_coefficient(simple_graph)
    for v in rcc.values():
        assert v >= 0.0


def test_rich_club_at_most_one(simple_graph):
    rcc = compute_rich_club_coefficient(simple_graph)
    for v in rcc.values():
        assert v <= 1.0 + 1e-9


def test_rich_club_empty_graph(empty_graph):
    rcc = compute_rich_club_coefficient(empty_graph)
    assert rcc == {}


# ---------------------------------------------------------------------------
# Disconnected graph — path length and small-world graceful fallback
# ---------------------------------------------------------------------------

def test_disconnected_path_length_is_nan_or_positive(disconnected_graph):
    m = compute_graph_metrics(disconnected_graph)
    val = m["average_path_length"]
    assert math.isnan(val) or val >= 1.0


def test_disconnected_small_world_is_nan_or_positive(disconnected_graph):
    m = compute_graph_metrics(disconnected_graph)
    val = m["small_world_sigma"]
    assert math.isnan(val) or val > 0.0


# ---------------------------------------------------------------------------
# Directed graph input — metrics computed on undirected copy
# ---------------------------------------------------------------------------

def test_metrics_accept_digraph():
    G = nx.DiGraph()
    for u, v in [(0, 1), (1, 2), (2, 0)]:
        G.add_edge(u, v, weight=0.5)
    m = compute_graph_metrics(G)
    assert "density" in m
    assert isinstance(m["density"], float)


# ---------------------------------------------------------------------------
# compute_metrics_all batch wrapper
# ---------------------------------------------------------------------------

def test_metrics_all_keys(simple_graph):
    graphs = {0: simple_graph, 1: simple_graph}
    all_m = compute_metrics_all(graphs)
    assert set(all_m.keys()) == {0, 1}
    for m in all_m.values():
        assert "density" in m
        assert "modularity" in m


def test_metrics_all_empty_dict():
    assert compute_metrics_all({}) == {}
