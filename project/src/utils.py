"""Shared utilities, constants, and helpers."""

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

DATA_PKL = "/Users/albertonogales/Dropbox/UAH/Papers IERU/En proceso/16 Brno Epilepsy/dataset.pkl"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

ID_COLS = ["id_subject", "id_channel", "channel_name"]

FEATURE_FAMILIES = {
    "HFO": ["HFO_R", "HFO_FR"],
    "Spikes": ["spike_count_b", "spike_count_j"],
    "Entropy": [],   # filled dynamically
    "Complexity": [],
}

ENTROPY_PREFIXES = ("pse_", "shan_entropy_", "shannon_entropy_", "sample_entropy_", "ren_")
COMPLEXITY_PREFIXES = ("lc_", "hjorth_mobility_raw")


def rpath(fname: str) -> str:
    return os.path.join(RESULTS_DIR, fname)


def dpath(fname: str) -> str:
    return os.path.join(DATA_DIR, fname)


def load_raw() -> pd.DataFrame:
    import pickle
    with open(DATA_PKL, "rb") as f:
        df = pickle.load(f)
    # Derive is_SOZ from seizure_info
    df["is_SOZ"] = df["seizure_info"] == "seizure onset zone"
    # Normalise intervention: treat 0 and 'n/a' as 'untreated'
    df["intervention"] = df["intervention"].replace({0: "untreated", "n/a": "untreated"})
    return df


def get_feature_cols(df: pd.DataFrame):
    exclude = set(ID_COLS + [
        "id_institution", "institution_name", "intervention", "seizure_info",
        "imaging_pathology", "is_SOZ",
        "latest_ilae", "latest_engel", "good_outcome",
        "synthetic_engel", "synthetic_ilae", "synthetic_good_outcome",
        "adequacy_score", "soz_probability", "dup_type",
    ])
    return [c for c in df.columns if c not in exclude]


def get_family_cols(df: pd.DataFrame):
    all_feats = get_feature_cols(df)
    entropy = [c for c in all_feats if c.startswith(ENTROPY_PREFIXES[:5])]
    complexity = [c for c in all_feats if c.startswith(("lc_", "hjorth_"))]
    hfo = [c for c in all_feats if c in ["HFO_R", "HFO_FR"]]
    spikes = [c for c in all_feats if c in ["spike_count_b", "spike_count_j"]]
    return {"HFO": hfo, "Spikes": spikes, "Entropy": entropy, "Complexity": complexity, "Full": all_feats}
