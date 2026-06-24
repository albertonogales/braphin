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
    fig.update_layout(title='', plot_bgcolor='white')
    fig.write_html(
        str(name) + '_plot.html',
        auto_open=auto_open,
        default_height='100%',
        default_width='100%',
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
    fig.update_layout(title='', plot_bgcolor='white')
    fig.write_image(str(name) + '.png', format='png', height=1000, width=1800)
