"""
BRAPHIN

A multimodal brain connectivity library extending EEGraph.

Supported modalities
--------------------
fMRI (NIfTI)
    Five-stage pipeline: Input → Preprocess → Denoise → Transform → Connectivity.
    Connectivity measures: pearson_correlation, partial_correlation, cross_correlation,
    corr_cross_correlation, coherence, imag_coherence, lagged_coherence, aec, aec_orth,
    mutual_information, sync_likelihood, granger_causality, transfer_entropy, pdc, psi.

EEG
    Full EEGraph pipeline accessed via the unified Graph class.
    Connectivity measures: cross_correlation, pearson_correlation, squared_coherence,
    imag_coherence, corr_cross_correlation, wpli, plv, pli, pli_bands, dtf,
    power_spectrum, spectral_entropy, shannon_entropy.

EEG utilities
-------------
    load_deap_dat  — load a DEAP .dat file and return EEG data as an MNE object.

Unified entry point (both modalities)
--------------------------------------
    from braphin import Graph

    # EEG
    g = Graph()
    g.load_data("subject.edf", modality="eeg")
    G, matrix = g.modelate(window_size=None, connectivity="plv", bands=["alpha"])

    # fMRI
    g = Graph()
    g.load_data("subject.nii.gz", modality="fmri")
    G, matrix = g.modelate(window_size=None, connectivity="pearson_correlation")

fMRI pipeline stages
--------------------
    InputfMRIData          -> BRAPHINInputBundle
    PreprocessBRAPHINData  -> BRAPHINPreprocessBundle
    DenoiseBRAPHINData     -> BRAPHINDenoiseBundle
    TransformBRAPHINData   -> BRAPHINTransformBundle
    ModelBRAPHINConnectivityData -> BRAPHINConnectivityBundle

EEG pipeline stages
-------------------
    InputEEGData           -> EEGInputBundle
    (windowing)            -> per-window connectivity matrices
    ModelData              -> mean connectivity matrix + NetworkX graph
"""

__version__ = "1.0.0"

# EEG utilities
from eegraph.io import load_deap_dat
from eegraph.tools import connectivity_measures as EEG_CONNECTIVITY_MEASURES  # {name: class_name}

from .bands import (
    FMRI_BANDS,
    bandpass_roi_time_series,
    compute_all_bands_connectivity,
    compute_band_connectivity,
)
from .config import (
    AtlasConfig,
    BRAPHINConfig,
    ConnectivityConfig,
    DenoiseConfig,
    InputConfig,
    PreprocessConfig,
)
from .connectivity import BRAPHINConnectivityBundle, ModelBRAPHINConnectivityData
from .denoise import BRAPHINDenoiseBundle, DenoiseBRAPHINData

# Unified Graph class — BRAPHINGraph subclasses eegraph.Graph.
# Dependency is strictly one-way: braphin → eegraph.
# modality="eeg"  → EEGraph pipeline (PLV, PLI, wPLI, DTF, …)
# modality="fmri" → BRAPHIN pipeline (Pearson, Granger, AEC, …)
from .graph import BRAPHINGraph
from .graph_metrics import (
    build_graph_from_matrix,
    compute_graph_metrics,
    compute_metrics_all,
    compute_modularity,
    compute_rich_club_coefficient,
)
from .importBRAPHINData import (
    BRAPHINInputBundle,
    InputBRAPHINData,  # backward-compat alias
    InputfMRIData,
)
from .importEEGData import EEGInputBundle, InputEEGData
from .model import ModelMRIData
from .preprocess import BRAPHINPreprocessBundle, PreprocessBRAPHINData, get_motion_confounds
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
from .transform import BRAPHINTransformBundle, TransformBRAPHINData, build_synthetic_atlas
from .visualize import build_fmri_graph, visualize_html, visualize_png

Graph = BRAPHINGraph  # convenience alias

# ── Standardised connectivity measure registries ──────────────────────────────
# fMRI measures  (braphin pipeline)
FMRI_CONNECTIVITY_MEASURES = CONNECTIVITY_MEASURES  # {name: description}


def list_fmri_connectivity_measures():
    """Return the list of supported fMRI connectivity method names."""
    return list(FMRI_CONNECTIVITY_MEASURES.keys())


def list_eeg_connectivity_measures():
    """Return the list of supported EEG connectivity method names."""
    return list(EEG_CONNECTIVITY_MEASURES.keys())


__all__ = [
    "InputConfig",
    "PreprocessConfig",
    "DenoiseConfig",
    "AtlasConfig",
    "ConnectivityConfig",
    "BRAPHINConfig",
    "InputfMRIData",
    "InputBRAPHINData",  # backward-compat alias
    "InputEEGData",
    "EEGInputBundle",
    "BRAPHINInputBundle",
    "PreprocessBRAPHINData",
    "get_motion_confounds",
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
    "build_fmri_graph",
    "FMRI_BANDS",
    "bandpass_roi_time_series",
    "compute_band_connectivity",
    "compute_all_bands_connectivity",
    "ModelMRIData",
    # EEG utilities
    "load_deap_dat",
    # Unified entry point (BRAPHINGraph exported as Graph for convenience)
    "Graph",
    "BRAPHINGraph",
    # Standardised connectivity measure registries
    "EEG_CONNECTIVITY_MEASURES",
    "FMRI_CONNECTIVITY_MEASURES",
    "list_eeg_connectivity_measures",
    "list_fmri_connectivity_measures",
]
