"""Stage 11 — Self-Organising Map analysis."""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from .utils import rpath, RANDOM_SEED
from .prediction import _patient_aggregate

try:
    from minisom import MiniSom
    HAS_SOM = True
except ImportError:
    HAS_SOM = False


def _train_som(X_norm: np.ndarray, rows: int, cols: int, seed: int):
    som = MiniSom(rows, cols, X_norm.shape[1],
                  sigma=1.0, learning_rate=0.5,
                  neighborhood_function="gaussian",
                  random_seed=seed)
    som.random_weights_init(X_norm)
    som.train_random(X_norm, 5000)
    return som


def _umatrix(som, rows, cols):
    weights = som.get_weights()
    u = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            neighbors = []
            for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                ni, nj = i+di, j+dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    neighbors.append(np.linalg.norm(weights[i,j] - weights[ni,nj]))
            u[i,j] = np.mean(neighbors) if neighbors else 0.0
    return u


def run_som(df: pd.DataFrame, feat_cols: list, eiss_features: list):
    if not HAS_SOM:
        print("[Stage 11] minisom not installed — skipping SOM analysis.")
        return

    target = "synthetic_good_outcome"
    avail = [f for f in eiss_features if f in df.columns]
    if not avail:
        avail = feat_cols[:5]

    pat_df = _patient_aggregate(df, feat_cols, target)
    for col in ["synthetic_engel", "synthetic_ilae"]:
        if col in df.columns:
            extras = df.groupby("id_subject")[col].first().reset_index()
            pat_df = pat_df.merge(extras, on="id_subject", how="left")

    X = pat_df[avail].fillna(0).values
    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    comp_dir = os.path.join(rpath(""), "component_planes")
    os.makedirs(comp_dir, exist_ok=True)

    for som_size in [(8, 8), (10, 10)]:
        rows, cols = som_size
        tag = f"{rows}x{cols}"
        print(f"[Stage 11] Training SOM {tag} ...", end=" ", flush=True)
        som = _train_som(X_norm, rows, cols, RANDOM_SEED)
        print("done")

        # U-matrix
        u = _umatrix(som, rows, cols)
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(u, cmap="bone_r", interpolation="nearest")
        plt.colorbar(im, ax=ax)
        ax.set_title(f"U-Matrix SOM {tag}")
        fig.savefig(os.path.join(comp_dir, f"u_matrix_{tag}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        # Copy first one to results root
        if tag == "8x8":
            fig2, ax2 = plt.subplots(figsize=(7, 6))
            im2 = ax2.imshow(u, cmap="bone_r", interpolation="nearest")
            plt.colorbar(im2, ax=ax2)
            ax2.set_title(f"U-Matrix SOM {tag}")
            fig2.savefig(rpath("u_matrix.png"), dpi=150, bbox_inches="tight")
            plt.close(fig2)

        # Component planes
        weights = som.get_weights()
        for fi, fname in enumerate(avail):
            fig, ax = plt.subplots(figsize=(5, 4))
            plane = weights[:, :, fi]
            im = ax.imshow(plane, cmap="viridis", interpolation="nearest")
            plt.colorbar(im, ax=ax)
            ax.set_title(f"{fname} — SOM {tag}")
            safe = fname.replace("/", "-")
            fig.savefig(os.path.join(comp_dir, f"comp_{tag}_{safe}.png"), dpi=120, bbox_inches="tight")
            plt.close(fig)

        # Hit map
        hit_map = np.zeros((rows, cols))
        bmus = [som.winner(x) for x in X_norm]
        for bmu in bmus:
            hit_map[bmu[0], bmu[1]] += 1
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(hit_map, cmap="hot_r", interpolation="nearest")
        plt.colorbar(im, ax=ax)
        ax.set_title(f"Hit Map SOM {tag}")
        fig.savefig(rpath("hit_map.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Ward clustering of prototypes
        proto = weights.reshape(-1, len(avail))
        Z = linkage(proto, method="ward")
        n_clusters = min(4, rows * cols)
        cluster_labels = fcluster(Z, n_clusters, criterion="maxclust")
        proto_map = cluster_labels.reshape(rows, cols)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        target_cols = ["synthetic_good_outcome", "synthetic_engel", "synthetic_ilae"]
        for ax, tcol in zip(axes, target_cols):
            overlay = np.full((rows, cols), np.nan)
            for bmu, tval in zip(bmus, pat_df[tcol].values if tcol in pat_df.columns else [0]*len(bmus)):
                overlay[bmu[0], bmu[1]] = tval
            im = ax.imshow(overlay, cmap="RdYlGn", interpolation="nearest")
            plt.colorbar(im, ax=ax)
            ax.set_title(tcol.replace("synthetic_", ""))
        plt.suptitle(f"SOM {tag} — Outcome Regions", fontsize=13)
        plt.tight_layout()
        fig.savefig(rpath(f"som_cluster_{tag}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    print("[Stage 11] SOM analysis complete.")
