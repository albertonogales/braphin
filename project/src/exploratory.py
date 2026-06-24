"""Stage 5 — Exploratory Analysis: correlation, PCA, UMAP."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from .utils import rpath, get_feature_cols, RANDOM_SEED


def _color_palette(labels, cmap="tab10"):
    uniq = sorted(set(labels))
    colors = plt.cm.get_cmap(cmap, len(uniq))
    return {v: colors(i) for i, v in enumerate(uniq)}, uniq


def correlation_heatmap(df: pd.DataFrame, feat_cols: list):
    corr = df[feat_cols].corr()
    fig, ax = plt.subplots(figsize=(18, 15))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0,
                vmin=-1, vmax=1, square=True, linewidths=0.0,
                ax=ax, cbar_kws={"shrink": 0.5})
    ax.set_title("Feature Correlation Heatmap", fontsize=14)
    plt.tight_layout()
    fig.savefig(rpath("correlation_heatmap.png"), dpi=150)
    plt.close(fig)
    print("[Stage 5] Saved correlation_heatmap.png")


def pca_projection(df: pd.DataFrame, feat_cols: list):
    X = df[feat_cols].values
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    coords = pca.fit_transform(X)

    targets = {
        "synthetic_engel": "Synthetic Engel",
        "synthetic_ilae": "Synthetic ILAE",
        "synthetic_good_outcome": "Good Outcome",
    }
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (col, title) in zip(axes, targets.items()):
        labels = df[col].astype(str).tolist() if col in df.columns else ["?"] * len(df)
        cmap, uniq = _color_palette(labels)
        for lbl in uniq:
            idx = [i for i, l in enumerate(labels) if l == lbl]
            ax.scatter(coords[idx, 0], coords[idx, 1],
                       c=[cmap[lbl]], label=lbl, s=8, alpha=0.5)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.legend(markerscale=2, fontsize=7)
    plt.suptitle("PCA Projection", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(rpath("pca_projection.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[Stage 5] Saved pca_projection.png")


def umap_projection(df: pd.DataFrame, feat_cols: list):
    try:
        import umap
    except ImportError:
        print("[Stage 5] UMAP not installed — skipping umap_projection.png")
        return

    X = df[feat_cols].values
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean",
                        random_state=RANDOM_SEED)
    coords = reducer.fit_transform(X)

    targets = {
        "synthetic_engel": "Synthetic Engel",
        "synthetic_ilae": "Synthetic ILAE",
        "synthetic_good_outcome": "Good Outcome",
    }
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (col, title) in zip(axes, targets.items()):
        labels = df[col].astype(str).tolist() if col in df.columns else ["?"] * len(df)
        cmap, uniq = _color_palette(labels)
        for lbl in uniq:
            idx = [i for i, l in enumerate(labels) if l == lbl]
            ax.scatter(coords[idx, 0], coords[idx, 1],
                       c=[cmap[lbl]], label=lbl, s=8, alpha=0.5)
        ax.set_title(title, fontsize=11)
        ax.legend(markerscale=2, fontsize=7)
    plt.suptitle("UMAP Projection", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(rpath("umap_projection.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[Stage 5] Saved umap_projection.png")


def run_exploratory(df: pd.DataFrame, feat_cols: list):
    correlation_heatmap(df, feat_cols)
    pca_projection(df, feat_cols)
    umap_projection(df, feat_cols)
