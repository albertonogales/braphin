"""
BRAPHIN visualisation helpers — reuse EEGraph's draw_graph unchanged.

Both functions are exact copies of the methods in ``eegraph.graph.Graph``
(``visualize_html`` / ``visualize_png``).  The underlying ``draw_graph``
function is imported directly from ``eegraph.tools`` — no re-implementation.

Usage
-----
    from braphin import build_graph_from_matrix, visualize_html, visualize_png

    G = build_graph_from_matrix(matrix, atlas_name="schaefer_200", threshold=0.3)
    visualize_html(G, "subject_01")          # → subject_01_plot.html
    visualize_png(G,  "subject_01")          # → subject_01.png
"""

from __future__ import annotations

import re

import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment

# Import draw_graph exactly as implemented in EEGraph — no code duplication.
from eegraph.tools import draw_graph


def visualize_html(graph, name: str, auto_open: bool = True) -> None:
    """
    Save an interactive Plotly graph of the connectivity network as HTML.

    Reuses ``eegraph.tools.draw_graph`` verbatim — identical to
    ``eegraph.graph.Graph.visualize_html``.

    Parameters
    ----------
    graph : NetworkX Graph
        Built by ``braphin.build_graph_from_matrix`` with *atlas_name* set so
        that node positions are attached.
    name : str
        Output file stem.  The file is written to ``<name>_plot.html``.
    auto_open : bool
        Open the HTML file in the default browser after writing.
    """
    fig = draw_graph(graph)
    fig.update_layout(title="", plot_bgcolor="white")
    fig.write_html(
        str(name) + "_plot.html",
        auto_open=auto_open,
        default_height="100%",
        default_width="100%",
    )


def visualize_png(graph, name: str) -> None:
    """
    Save the connectivity network graph as a PNG image.

    Reuses ``eegraph.tools.draw_graph`` verbatim — identical to
    ``eegraph.graph.Graph.visualize_png``.

    Parameters
    ----------
    graph : NetworkX Graph
        Built by ``braphin.build_graph_from_matrix`` with *atlas_name* set.
    name : str
        Output file stem.  The file is written to ``<name>.png``.
    """
    fig = draw_graph(graph)
    fig.update_layout(title="", plot_bgcolor="white")
    fig.write_image(str(name) + ".png", format="png", height=1000, width=1800)


# ---------------------------------------------------------------------------
# 3D brain layout helpers (extracted from ModelMRIData)
# ---------------------------------------------------------------------------


def _infer_lr_partner(label):
    """
    Attempt to infer the left/right partner of an ROI from its name.

    Covered cases:
    - AAL: Precentral_L <-> Precentral_R
    - Left/Right variants
    - Returns None if no pattern is matched.
    """
    label = str(label)

    patterns = [
        (r"^(.*)_L$", r"\1_R"),
        (r"^(.*)_R$", r"\1_L"),
        (r"^(.*)_Left$", r"\1_Right"),
        (r"^(.*)_Right$", r"\1_Left"),
        (r"^(.*)Left$", r"\1Right"),
        (r"^(.*)Right$", r"\1Left"),
    ]

    for pattern, replacement in patterns:
        if re.match(pattern, label):
            return re.sub(pattern, replacement, label)

    return None


def _build_symmetric_axial_layout(roi_centroids_3d):
    """
    Build a symmetric axial projection for visualisation purposes only.

    Important:
    - Does NOT modify the real 3D centroids of the atlas.
    - Returns:
        * pos2d: symmetric (x, y) positions for drawing
        * depth: symmetric z depth for colouring
    - Works well with AAL by name (_L/_R).
    - For atlases without real anatomical names (e.g. ROI_1...ROI_n),
      performs automatic left/right matching using proximity in (y, z).
    """
    if not roi_centroids_3d:
        return {}, {}

    raw = {
        node: (
            float(coords[0]),
            float(coords[1]),
            float(coords[2]),
        )
        for node, coords in roi_centroids_3d.items()
    }

    # Base layout: nodes that cannot be paired stay as-is
    pos2d = {node: (coords[0], coords[1]) for node, coords in raw.items()}
    depth = {node: coords[2] for node, coords in raw.items()}

    processed = set()

    # --------------------------------------------------
    # 1) Explicit pairing by name (_L / _R)
    # --------------------------------------------------
    for node in raw:
        partner = _infer_lr_partner(node)

        if partner is None or partner not in raw:
            continue

        if node in processed or partner in processed:
            continue

        left, right = node, partner

        # Ensure "left" is the one with the smaller x
        if raw[left][0] > raw[right][0]:
            left, right = right, left

        x_left, y_left, z_left = raw[left]
        x_right, y_right, z_right = raw[right]

        mirrored_x = 0.5 * (abs(x_left) + abs(x_right))
        avg_y = 0.5 * (y_left + y_right)
        avg_z = 0.5 * (z_left + z_right)

        pos2d[left] = (-mirrored_x, avg_y)
        pos2d[right] = (mirrored_x, avg_y)

        depth[left] = avg_z
        depth[right] = avg_z

        processed.add(left)
        processed.add(right)

    # --------------------------------------------------
    # 2) Automatic fallback for atlases without L/R names
    # --------------------------------------------------
    remaining_left = [node for node, (x, _, _) in raw.items() if x < 0 and node not in processed]
    remaining_right = [node for node, (x, _, _) in raw.items() if x > 0 and node not in processed]

    if remaining_left and remaining_right:
        left_yz = np.array([[raw[node][1], raw[node][2]] for node in remaining_left], dtype=float)
        right_yz = np.array([[raw[node][1], raw[node][2]] for node in remaining_right], dtype=float)

        # Match each left ROI with the most similar right ROI in (y, z)
        cost = np.sum((left_yz[:, None, :] - right_yz[None, :, :]) ** 2, axis=2)
        row_ind, col_ind = linear_sum_assignment(cost)

        for i, j in zip(row_ind, col_ind, strict=False):
            left = remaining_left[i]
            right = remaining_right[j]

            x_left, y_left, z_left = raw[left]
            x_right, y_right, z_right = raw[right]

            mirrored_x = 0.5 * (abs(x_left) + abs(x_right))
            avg_y = 0.5 * (y_left + y_right)
            avg_z = 0.5 * (z_left + z_right)

            pos2d[left] = (-mirrored_x, avg_y)
            pos2d[right] = (mirrored_x, avg_y)

            depth[left] = avg_z
            depth[right] = avg_z

    return pos2d, depth


def _build_symmetric_coronal_layout(roi_centroids_3d):
    """
    Build a symmetric coronal projection for visualisation purposes only.

    Returns:
    - pos2d: symmetric (x, z) positions for drawing
    - depth: symmetric y depth for colouring
    """
    if not roi_centroids_3d:
        return {}, {}

    raw = {
        node: (
            float(coords[0]),
            float(coords[1]),
            float(coords[2]),
        )
        for node, coords in roi_centroids_3d.items()
    }

    # Base layout: nodes that cannot be paired stay as-is
    pos2d = {node: (coords[0], coords[2]) for node, coords in raw.items()}
    depth = {node: coords[1] for node, coords in raw.items()}

    processed = set()

    # --------------------------------------------------
    # 1) Explicit pairing by name (_L / _R)
    # --------------------------------------------------
    for node in raw:
        partner = _infer_lr_partner(node)

        if partner is None or partner not in raw:
            continue

        if node in processed or partner in processed:
            continue

        left, right = node, partner

        # Ensure "left" is the one with the smaller x
        if raw[left][0] > raw[right][0]:
            left, right = right, left

        x_left, y_left, z_left = raw[left]
        x_right, y_right, z_right = raw[right]

        mirrored_x = 0.5 * (abs(x_left) + abs(x_right))
        avg_y = 0.5 * (y_left + y_right)
        avg_z = 0.5 * (z_left + z_right)

        pos2d[left] = (-mirrored_x, avg_z)
        pos2d[right] = (mirrored_x, avg_z)

        depth[left] = avg_y
        depth[right] = avg_y

        processed.add(left)
        processed.add(right)

    # --------------------------------------------------
    # 2) Automatic fallback for atlases without L/R names
    # --------------------------------------------------
    remaining_left = [node for node, (x, _, _) in raw.items() if x < 0 and node not in processed]
    remaining_right = [node for node, (x, _, _) in raw.items() if x > 0 and node not in processed]

    if remaining_left and remaining_right:
        left_yz = np.array([[raw[node][1], raw[node][2]] for node in remaining_left], dtype=float)
        right_yz = np.array([[raw[node][1], raw[node][2]] for node in remaining_right], dtype=float)

        # Match each left ROI with the most similar right ROI in (y, z)
        cost = np.sum((left_yz[:, None, :] - right_yz[None, :, :]) ** 2, axis=2)
        row_ind, col_ind = linear_sum_assignment(cost)

        for i, j in zip(row_ind, col_ind, strict=False):
            left = remaining_left[i]
            right = remaining_right[j]

            x_left, y_left, z_left = raw[left]
            x_right, y_right, z_right = raw[right]

            mirrored_x = 0.5 * (abs(x_left) + abs(x_right))
            avg_y = 0.5 * (y_left + y_right)
            avg_z = 0.5 * (z_left + z_right)

            pos2d[left] = (-mirrored_x, avg_z)
            pos2d[right] = (mirrored_x, avg_z)

            depth[left] = avg_y
            depth[right] = avg_y

    return pos2d, depth


def build_fmri_graph(
    connectivity_matrix,
    roi_labels=None,
    roi_centroids_3d=None,
    centroid_coordinate_space="world",
    projection="coronal",
):
    """
    Convert a ROI x ROI matrix into a NetworkX graph.

    For fMRI:
    - stores the full 3D position,
    - projects to 2D with a fixed anatomical view,
    - and stores a separate depth value for colouring nodes.

    In the axial view, symmetrisation is applied for visualisation only so
    that left/right nodes are mirrored across the midline.
    """
    matrix = np.array(connectivity_matrix, copy=True)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("The connectivity matrix must be square to build the graph.")

    is_symmetric = np.allclose(matrix, matrix.T, atol=1e-5)

    if is_symmetric:
        G = nx.from_numpy_array(matrix)
    else:
        G = nx.from_numpy_array(matrix, create_using=nx.DiGraph)

    if roi_labels is not None and len(roi_labels) == matrix.shape[0]:
        mapping = dict(enumerate(roi_labels))
        G = nx.relabel_nodes(G, mapping)

    G.graph["modality"] = "fmri"
    G.graph["coordinate_space"] = centroid_coordinate_space
    G.graph["projection"] = projection

    if roi_centroids_3d:
        pos2d = {}
        pos3d = {}
        depth = {}

        # Always store the real 3D position
        for node in G.nodes():
            centroid = roi_centroids_3d.get(node)
            if centroid is None:
                continue

            x, y, z = centroid
            pos3d[node] = (float(x), float(y), float(z))

        # --------------------------------------------------
        # 2D projection
        # --------------------------------------------------
        if projection == "axial":
            # Top view: X-Y plane
            # Symmetry is applied here for visualisation only.
            axial_centroids = {node: pos3d[node] for node in G.nodes() if node in pos3d}

            pos2d, depth = _build_symmetric_axial_layout(axial_centroids)

            G.graph["depth_axis"] = "z"
            G.graph["depth_legend"] = "Z depth (inferior ↔ superior)"

        elif projection == "coronal":
            # Front view: X-Z plane with left/right visual symmetry only.
            coronal_centroids = {node: pos3d[node] for node in G.nodes() if node in pos3d}

            pos2d, depth = _build_symmetric_coronal_layout(coronal_centroids)

            G.graph["depth_axis"] = "y"
            G.graph["depth_legend"] = "Y depth (posterior ↔ anterior)"

        elif projection == "sagittal":
            for node in G.nodes():
                if node not in pos3d:
                    continue

                x, y, z = pos3d[node]
                pos2d[node] = (y, z)
                depth[node] = x

            G.graph["depth_axis"] = "x"
            G.graph["depth_legend"] = "X depth (left ↔ right)"

        else:
            raise ValueError(f"Unsupported MRI projection: {projection}")

        if pos3d:
            nx.set_node_attributes(G, pos3d, "pos3d")
        if pos2d:
            nx.set_node_attributes(G, pos2d, "pos")
        if depth:
            nx.set_node_attributes(G, depth, "depth")

    for _u, _v, data in G.edges(data=True):
        weight = float(data.get("weight", 1.0))
        thickness = max(0.5, abs(weight) * 6)
        data["thickness"] = thickness

    return G
