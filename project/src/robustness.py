"""Stage 12 — Robustness Analysis."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from .utils import rpath, dpath, RANDOM_SEED
from .prediction import _patient_aggregate, lopo_evaluate
from sklearn.linear_model import LogisticRegression

try:
    from minisom import MiniSom
    HAS_SOM = True
except ImportError:
    HAS_SOM = False


def _variation_of_information(labels_a, labels_b):
    from sklearn.metrics import mutual_info_score
    n = len(labels_a)
    mi = mutual_info_score(labels_a, labels_b)
    ha = -sum((np.bincount(labels_a.astype(int)) / n) *
               np.log((np.bincount(labels_a.astype(int)) / n) + 1e-12))
    hb = -sum((np.bincount(labels_b.astype(int)) / n) *
               np.log((np.bincount(labels_b.astype(int)) / n) + 1e-12))
    return ha + hb - 2 * mi


def robustness_analysis(feat_cols: list, eiss_features: list):
    records = []
    target = "synthetic_good_outcome"

    # Load three datasets
    datasets = {}
    for tag in ["A", "B", "C"]:
        try:
            datasets[tag] = pd.read_pickle(dpath(f"dataset_A.pkl"))  # reuse A since synthetic outcomes are attached
        except Exception:
            pass

    # Load processed datasets
    proc_datasets = {}
    for tag, fname in [("median", "processed_A_median.pkl"), ("knn", "processed_B_knn.pkl")]:
        try:
            proc_datasets[tag] = pd.read_pickle(dpath(fname))
        except Exception as e:
            print(f"  [Stage 12] Could not load {fname}: {e}")

    # Evaluate prediction robustness across imputation strategies
    for strat, df_proc in proc_datasets.items():
        avail = [c for c in feat_cols if c in df_proc.columns]
        if not avail or target not in df_proc.columns:
            continue
        model = LogisticRegression(max_iter=2000, random_state=RANDOM_SEED, solver="saga")
        try:
            metrics = lopo_evaluate(df_proc, avail, target, True, "LR", model)
            records.append({
                "analysis": "imputation",
                "variant": strat,
                "metric": "AUROC",
                "value": metrics.get("AUROC", np.nan),
            })
        except Exception as e:
            print(f"  [Stage 12] Imputation robustness failed ({strat}): {e}")

    # SOM stability — 30 runs
    som_aris = []
    som_vis = []
    if HAS_SOM:
        try:
            df_proc = pd.read_pickle(dpath("processed_A_median.pkl"))
            avail = [f for f in eiss_features if f in df_proc.columns]
            if not avail:
                avail = feat_cols[:5]
            pat_df = _patient_aggregate(df_proc, avail, target)
            X = pat_df[avail].fillna(0).values
            X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
            rows, cols = 8, 8

            ref_labels = None
            for run in range(30):
                seed = RANDOM_SEED + run
                som = MiniSom(rows, cols, X_norm.shape[1],
                              sigma=1.0, learning_rate=0.5,
                              neighborhood_function="gaussian",
                              random_seed=seed)
                som.random_weights_init(X_norm)
                som.train_random(X_norm, 5000)
                bmu_indices = np.array([som.winner(x)[0] * cols + som.winner(x)[1] for x in X_norm])

                if ref_labels is None:
                    ref_labels = bmu_indices
                else:
                    ari = adjusted_rand_score(ref_labels, bmu_indices)
                    vi = _variation_of_information(ref_labels, bmu_indices)
                    som_aris.append(ari)
                    som_vis.append(vi)

            records.append({"analysis": "SOM_stability", "variant": "ARI_mean",
                             "metric": "ARI", "value": np.mean(som_aris)})
            records.append({"analysis": "SOM_stability", "variant": "VI_mean",
                             "metric": "VI", "value": np.mean(som_vis)})
            print(f"[Stage 12] SOM stability ARI={np.mean(som_aris):.4f}, VI={np.mean(som_vis):.4f}")
        except Exception as e:
            print(f"[Stage 12] SOM stability analysis failed: {e}")

    robustness_df = pd.DataFrame(records)
    robustness_df.to_csv(rpath("robustness_report.csv"), index=False)

    # Stability boxplot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if som_aris:
        axes[0].boxplot(som_aris)
        axes[0].set_title("SOM Stability — ARI (30 runs)")
        axes[0].set_ylabel("Adjusted Rand Index")
    if som_vis:
        axes[1].boxplot(som_vis)
        axes[1].set_title("SOM Stability — Variation of Information")
        axes[1].set_ylabel("VI")
    plt.tight_layout()
    fig.savefig(rpath("stability_boxplot.png"), dpi=150)
    plt.close(fig)

    print("[Stage 12] Robustness analysis complete.")
    return robustness_df
