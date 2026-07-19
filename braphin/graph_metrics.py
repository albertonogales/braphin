"""
Graph-theoretic metrics for brain connectivity networks.

All metrics operate on a NetworkX Graph (or DiGraph) built from a connectivity
matrix.  For directed measures the graph is used as-is; for all undirected
measures a symmetric, self-loop-free copy is produced internally.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _undirected_clean(G: nx.Graph) -> nx.Graph:
    """Return an undirected, self-loop-free, unweighted copy of *G*."""
    G_und = G.to_undirected() if G.is_directed() else G
    G_clean = nx.Graph(G_und)
    G_clean.remove_edges_from(nx.selfloop_edges(G_clean))
    return G_clean


def _largest_cc(G: nx.Graph) -> nx.Graph:
    """Return the subgraph induced by the largest connected component."""
    components = list(nx.connected_components(G))
    if not components:
        return G.__class__()
    largest = max(components, key=len)
    return G.subgraph(largest).copy()


# ─────────────────────────────────────────────────────────────────────────────
# Modularity
# ─────────────────────────────────────────────────────────────────────────────


def compute_modularity(G: nx.Graph) -> float:
    """
    Estimate the network modularity Q via greedy community detection.

    Parameters
    ----------
    G : NetworkX Graph

    Returns
    -------
    float
        Modularity Q in [0, 1]. Returns 0.0 if the graph has no edges.
    """
    G_clean = _undirected_clean(G)
    if G_clean.number_of_edges() == 0:
        return 0.0
    communities = nx.community.greedy_modularity_communities(G_clean, weight="weight")
    return float(nx.community.modularity(G_clean, communities, weight="weight"))


# ─────────────────────────────────────────────────────────────────────────────
# Rich-club coefficient
# ─────────────────────────────────────────────────────────────────────────────


def compute_rich_club_coefficient(G: nx.Graph) -> dict[int, float]:
    """
    Compute the rich-club coefficient Φ(k) for all degree values k present in the graph.

    Parameters
    ----------
    G : NetworkX Graph

    Returns
    -------
    dict {int: float}
        Mapping from degree threshold k to Φ(k).
        Empty dict if the graph has fewer than 2 nodes.
    """
    G_clean = _undirected_clean(G)
    if G_clean.number_of_nodes() < 2:
        return {}
    try:
        rcc = nx.rich_club_coefficient(G_clean, normalized=False)
        return {int(k): float(v) for k, v in rcc.items()}
    except Exception as exc:
        logger.warning("Could not compute rich-club coefficient: %s", exc)
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────


def compute_graph_metrics(G: nx.Graph) -> dict[str, object]:
    """
    Compute a standard set of graph-theoretic metrics on a brain connectivity graph.

    Parameters
    ----------
    G : NetworkX Graph or DiGraph
        Nodes = ROIs, edges = connectivity weights.

    Returns
    -------
    metrics : dict
        Network-level scalars and node-level dicts; disconnected graphs fall back
        to the largest connected component for path-length measures.
    """
    G_clean = _undirected_clean(G)
    n = G_clean.number_of_nodes()
    metrics: dict[str, object] = {}

    # ── Network-level scalars ────────────────────────────────────────────────
    metrics["density"] = nx.density(G_clean)
    metrics["transitivity"] = nx.transitivity(G_clean)
    metrics["average_clustering"] = nx.average_clustering(G_clean, weight="weight")
    metrics["global_efficiency"] = nx.global_efficiency(G_clean)
    metrics["local_efficiency"] = nx.local_efficiency(G_clean)

    # Average path length (largest connected component fallback)
    if n > 1 and G_clean.number_of_edges() > 0:
        if nx.is_connected(G_clean):
            metrics["average_path_length"] = nx.average_shortest_path_length(G_clean, weight=None)
        else:
            lcc = _largest_cc(G_clean)
            if len(lcc) > 1:
                metrics["average_path_length"] = nx.average_shortest_path_length(lcc, weight=None)
                logger.warning(
                    "Graph is disconnected; average_path_length computed on "
                    "the largest connected component (%d / %d nodes).",
                    len(lcc),
                    n,
                )
            else:
                metrics["average_path_length"] = float("nan")
    else:
        metrics["average_path_length"] = float("nan")

    # Degree assortativity
    try:
        metrics["degree_assortativity"] = nx.degree_assortativity_coefficient(G_clean)
    except Exception:
        metrics["degree_assortativity"] = float("nan")

    # Small-world sigma: σ = (C / C_rand) / (L / L_rand)
    # Analytical random-graph approximations: C_rand ≈ <k>/n, L_rand ≈ ln(n)/ln(<k>)
    L = metrics["average_path_length"]
    if (
        n > 1
        and G_clean.number_of_edges() > 0
        and not np.isnan(float(cast(Any, L)))
        and float(cast(Any, L)) > 0
    ):
        k_mean = float(np.mean([d for _, d in G_clean.degree()]))
        C = float(cast(Any, metrics["average_clustering"]))
        if k_mean > 1:
            C_rand = k_mean / n
            L_rand = np.log(n) / np.log(k_mean)
            metrics["small_world_sigma"] = (
                (C / C_rand) / (float(cast(Any, L)) / L_rand)
                if C_rand > 0 and L_rand > 0
                else float("nan")
            )
        else:
            metrics["small_world_sigma"] = float("nan")
    else:
        metrics["small_world_sigma"] = float("nan")

    # Modularity
    metrics["modularity"] = compute_modularity(G_clean)

    # Rich-club coefficient
    metrics["rich_club_coefficient"] = compute_rich_club_coefficient(G_clean)

    # ── Node-level dicts ─────────────────────────────────────────────────────
    metrics["degree_centrality"] = nx.degree_centrality(G_clean)
    metrics["degree"] = dict(G_clean.degree())

    # Node strength (weighted degree)
    metrics["node_strength"] = {
        node: float(sum(data.get("weight", 1.0) for _, data in G_clean[node].items()))
        for node in G_clean.nodes()
    }

    metrics["betweenness_centrality"] = nx.betweenness_centrality(G_clean, weight="weight")

    try:
        metrics["eigenvector_centrality"] = nx.eigenvector_centrality(
            G_clean, weight="weight", max_iter=1000
        )
    except nx.PowerIterationFailedConvergence:
        logger.warning("Eigenvector centrality did not converge; returning NaN for all nodes.")
        metrics["eigenvector_centrality"] = {node: float("nan") for node in G_clean.nodes()}

    metrics["closeness_centrality"] = nx.closeness_centrality(G_clean)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Batch wrapper
# ─────────────────────────────────────────────────────────────────────────────


def compute_metrics_all(graphs: dict) -> dict[int, dict[str, object]]:
    """
    Compute graph-theoretic metrics for every graph in a dict of NetworkX graphs.

    Parameters
    ----------
    graphs : dict {key: NetworkX Graph}
        Dictionary of graphs, as returned by ``Graph.modelate()`` or built from
        a connectivity matrix with :func:`build_graph_from_matrix`.

    Returns
    -------
    dict {key: metrics_dict}
        Each value is the output of :func:`compute_graph_metrics`.
    """
    return {k: compute_graph_metrics(G) for k, G in graphs.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Utility: build a NetworkX graph from a connectivity matrix
# ─────────────────────────────────────────────────────────────────────────────


def build_graph_from_matrix(
    connectivity_matrix: np.ndarray,
    roi_labels: list[str] | None = None,
    threshold: float = 0.0,
    directed: bool = False,
) -> nx.Graph:
    """
    Build a NetworkX graph from a connectivity matrix.

    Parameters
    ----------
    connectivity_matrix : ndarray (N, N)
        Square connectivity matrix (symmetric for undirected graphs).
    roi_labels : list of str, optional
        Node labels.  Defaults to ``["ROI_0", "ROI_1", ...]``.
    threshold : float
        Edges with |weight| < threshold are excluded.  Default 0 (keep all).
    directed : bool
        If True, return a DiGraph.  Default False.

    Returns
    -------
    NetworkX Graph or DiGraph
    """
    N = connectivity_matrix.shape[0]
    if roi_labels is None:
        roi_labels = [f"ROI_{i}" for i in range(N)]

    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(roi_labels)

    for i in range(N):
        for j in range(N if directed else i):
            if i == j:
                continue
            w = float(connectivity_matrix[i, j])
            if abs(w) >= threshold:
                G.add_edge(roi_labels[i], roi_labels[j], weight=w)

    return G
