"""
BRAPHIN

An fMRI functional connectivity pipeline extending EEGraph.
Transforms a raw 4-D BOLD NIfTI image into a NetworkX graph whose nodes are
brain regions of interest (ROIs) and whose edges are weighted by pairwise
connectivity.

Pipeline stages:
    InputBRAPHINData       -> BRAPHINInputBundle
    PreprocessBRAPHINData  -> BRAPHINPreprocessBundle
    DenoiseBRAPHINData     -> BRAPHINDenoiseBundle
    TransformBRAPHINData   -> BRAPHINTransformBundle
    ModelBRAPHINConnectivityData -> BRAPHINConnectivityBundle
"""

__version__ = "1.0.0"

from .config import (
    AtlasConfig,
    ConnectivityConfig,
    DenoiseConfig,
    InputConfig,
    BRAPHINConfig,
    PreprocessConfig,
)
from .connectivity import BRAPHINConnectivityBundle, ModelBRAPHINConnectivityData
from .denoise import DenoiseBRAPHINData, BRAPHINDenoiseBundle
from .importBRAPHINData import InputBRAPHINData, BRAPHINInputBundle
from .preprocess import BRAPHINPreprocessBundle, PreprocessBRAPHINData
from .strategy import (
    ConnectivityStrategy,
    PearsonConnectivityStrategy,
    get_connectivity_strategy,
)
from .tools import (
    CONNECTIVITY_MEASURES,
    apply_connectivity_threshold,
    compute_pearson_correlation,
    list_connectivity_measures,
    validate_connectivity_method,
)
from .graph_metrics import (
    build_graph_from_matrix,
    compute_graph_metrics,
    compute_metrics_all,
    compute_modularity,
    compute_rich_club_coefficient,
)
from .transform import BRAPHINTransformBundle, TransformBRAPHINData, build_synthetic_atlas
from .visualize import visualize_html, visualize_png

__all__ = [
    "InputConfig",
    "PreprocessConfig",
    "DenoiseConfig",
    "AtlasConfig",
    "ConnectivityConfig",
    "BRAPHINConfig",
    "InputBRAPHINData",
    "BRAPHINInputBundle",
    "PreprocessBRAPHINData",
    "BRAPHINPreprocessBundle",
    "DenoiseBRAPHINData",
    "BRAPHINDenoiseBundle",
    "TransformBRAPHINData",
    "BRAPHINTransformBundle",
    "build_synthetic_atlas",
    "CONNECTIVITY_MEASURES",
    "list_connectivity_measures",
    "validate_connectivity_method",
    "compute_pearson_correlation",
    "apply_connectivity_threshold",
    "ConnectivityStrategy",
    "PearsonConnectivityStrategy",
    "get_connectivity_strategy",
    "ModelBRAPHINConnectivityData",
    "BRAPHINConnectivityBundle",
    "compute_graph_metrics",
    "compute_metrics_all",
    "compute_modularity",
    "compute_rich_club_coefficient",
    "build_graph_from_matrix",
    "visualize_html",
    "visualize_png",
]
