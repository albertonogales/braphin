"""
experiments/graph_metrics_analysis.py
======================================

Graph-theoretical analysis of BRAPHIN-generated connectivity graphs
on the external Neurocon test set (Table 3 in the paper).

Computes 12 graph-theoretical metrics for each subject, compares
Parkinson's disease patients vs. healthy controls using the
Mann-Whitney U test, and applies Benjamini-Hochberg FDR correction
across the 12 simultaneous comparisons.

Input
-----
Same .npy connectivity matrices as gnn_parkinson.py, restricted to the
Neurocon subjects listed in data/splits/neurocon_*.txt.

Usage
-----
    python experiments/graph_metrics_analysis.py \
        --data_dir data/matrices --splits_dir data/splits --threshold 0.5
"""

import argparse
import os

import numpy as np
from scipy.stats import mannwhitneyu

from braphin import build_graph_from_matrix, compute_graph_metrics


# ---------------------------------------------------------------------------
# FDR correction (Benjamini-Hochberg)
# ---------------------------------------------------------------------------

def bh_correction(p_values: list[float], alpha: float = 0.05) -> list[float]:
    """Return BH-corrected p-values (same length as input)."""
    n = len(p_values)
    order = np.argsort(p_values)
    ranks = np.empty(n)
    ranks[order] = np.arange(1, n + 1)
    corrected = np.minimum(1.0, p_values * n / ranks)
    # Enforce monotonicity
    for i in range(n - 2, -1, -1):
        corrected[order[i]] = min(corrected[order[i]], corrected[order[i + 1]])
    return corrected.tolist()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_matrix(path: str) -> np.ndarray:
    if path.endswith(".npz"):
        content = np.load(path)
        key = next(
            (k for k in ("connectivity_matrix", "conn", "fc", "arr_0") if k in content),
            next(k for k in content if content[k].ndim == 2),
        )
        return content[key].astype(np.float32)
    return np.load(path).astype(np.float32)


def load_neurocon_matrices(data_dir: str, splits_dir: str, threshold: float):
    """
    Load connectivity matrices for Neurocon subjects only.

    Supports both BRAPHIN directory layout (parkinson_control / parkinson_patient
    with .npz files in subject subdirectories, identified by "neurocon" prefix)
    and the flat .npy layout (controls / patients directories with split lists).

    Returns:
        controls : list of NetworkX graphs (label=0)
        patients : list of NetworkX graphs (label=1)
    """
    braphin_mode = os.path.isdir(os.path.join(data_dir, "parkinson_control"))
    controls: list = []
    patients: list = []

    if braphin_mode:
        for graphs, folder in [(controls, "parkinson_control"), (patients, "parkinson_patient")]:
            folder_path = os.path.join(data_dir, folder)
            for entry in sorted(os.listdir(folder_path)):
                if not entry.startswith("neurocon"):
                    continue
                npz = os.path.join(folder_path, entry, "connectivity_matrix_fmri.npz")
                if not os.path.isfile(npz):
                    continue
                matrix = _load_matrix(npz)
                if matrix.shape == (116, 116):
                    graphs.append(build_graph_from_matrix(matrix, threshold=threshold))
    else:
        neurocon_controls: set[str] = set()
        neurocon_patients: set[str] = set()
        ctrl_path = os.path.join(splits_dir, "neurocon_controls.txt")
        pat_path = os.path.join(splits_dir, "neurocon_patients.txt")
        if os.path.exists(ctrl_path):
            with open(ctrl_path) as f:
                neurocon_controls = {line.strip() for line in f if line.strip()}
        if os.path.exists(pat_path):
            with open(pat_path) as f:
                neurocon_patients = {line.strip() for line in f if line.strip()}

        for graphs, subdir, neurocon_set in [
            (controls, "controls", neurocon_controls),
            (patients, "patients", neurocon_patients),
        ]:
            folder = os.path.join(data_dir, subdir)
            for fname in sorted(os.listdir(folder)):
                if not (fname.endswith(".npy") or fname.endswith(".npz")):
                    continue
                if fname not in neurocon_set:
                    continue
                matrix = _load_matrix(os.path.join(folder, fname))
                graphs.append(build_graph_from_matrix(matrix, threshold=threshold))

    return controls, patients


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

METRIC_NAMES = [
    "num_edges",
    "density",
    "mean_degree",
    "mean_weighted_strength",
    "global_efficiency",
    "mean_path_length",
    "weighted_clustering_coefficient",
    "local_efficiency",
    "modularity",
    "num_communities",
    "rich_club_coefficient",
    "small_world_sigma",
]


def extract_metrics(graphs: list) -> dict[str, list[float]]:
    """Compute all 12 graph-theoretical metrics for each graph."""
    results: dict[str, list[float]] = {m: [] for m in METRIC_NAMES}
    for G in graphs:
        m = compute_graph_metrics(G)
        for name in METRIC_NAMES:
            results[name].append(float(m.get(name, float("nan"))))
    return results


def run_analysis(controls: list, patients: list) -> None:
    print(f"\nNeurocon test set: {len(controls)} controls, {len(patients)} patients")
    print("Computing graph-theoretical metrics...")

    ctrl_metrics = extract_metrics(controls)
    pat_metrics = extract_metrics(patients)

    # Mann-Whitney U test for each metric
    p_values = []
    for name in METRIC_NAMES:
        _, p = mannwhitneyu(ctrl_metrics[name], pat_metrics[name], alternative="two-sided")
        p_values.append(p)

    p_adj = bh_correction(p_values)

    # Print results table
    header = (
        f"{'Metric':<35} {'Controls':>10} {'Patients':>10} "
        f"{'Δ%':>8} {'p':>8} {'p_adj':>8} {'Sig*':>5}"
    )
    print(f"\n{header}")
    print("-" * len(header))

    for i, name in enumerate(METRIC_NAMES):
        ctrl_mean = np.nanmean(ctrl_metrics[name])
        pat_mean = np.nanmean(pat_metrics[name])
        pct_change = (pat_mean - ctrl_mean) / max(abs(ctrl_mean), 1e-9) * 100
        sig = "*" if p_adj[i] < 0.05 else ""
        print(
            f"{name:<35} {ctrl_mean:>10.3f} {pat_mean:>10.3f} "
            f"{pct_change:>+8.2f}% {p_values[i]:>8.4f} {p_adj[i]:>8.4f} {sig:>5}"
        )

    n_sig = sum(1 for p in p_adj if p < 0.05)
    print(f"\n{n_sig}/{len(METRIC_NAMES)} metrics significant after BH FDR correction (α=0.05)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Graph-theoretical analysis of Parkinson vs. Control (Table 3)"
    )
    p.add_argument("--data_dir", default="data",
                   help="Root data directory. BRAPHIN layout: contains parkinson_control/ and "
                        "parkinson_patient/. Flat layout: contains controls/ and patients/ subdirs.")
    p.add_argument("--splits_dir", default="data/splits",
                   help="Directory with neurocon_controls.txt and neurocon_patients.txt "
                        "(only used with flat .npy layout)")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Connectivity threshold for graph construction")
    return p.parse_args()


def main():
    args = parse_args()
    controls, patients = load_neurocon_matrices(args.data_dir, args.splits_dir, args.threshold)
    if not controls or not patients:
        print("No Neurocon subjects found. Check data_dir and splits_dir.")
        return
    run_analysis(controls, patients)


if __name__ == "__main__":
    main()
