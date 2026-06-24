"""Stage 8 — Explainability: SHAP analysis."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from .utils import rpath, RANDOM_SEED
from .prediction import _patient_aggregate

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


def run_explainability(df: pd.DataFrame, feat_cols: list, best_model_name: str, best_model):
    if not HAS_SHAP:
        print("[Stage 8] SHAP not installed — skipping explainability.")
        return pd.DataFrame()

    target = "synthetic_good_outcome"
    pat_df = _patient_aggregate(df, feat_cols, target)
    X = pat_df[feat_cols].values
    y = pat_df[target].values

    # Refit on full data for SHAP
    best_model.fit(X, y)

    if isinstance(best_model, (RandomForestClassifier,)):
        explainer = shap.TreeExplainer(best_model)
        shap_vals = explainer.shap_values(X)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]  # positive class
    else:
        explainer = shap.LinearExplainer(best_model, X)
        shap_vals = explainer.shap_values(X)

    # SHAP summary plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_vals, X, feature_names=feat_cols, show=False, plot_size=None)
    plt.tight_layout()
    fig.savefig(rpath("shap_summary.png"), dpi=150, bbox_inches="tight")
    plt.close("all")

    # SHAP bar plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_vals, X, feature_names=feat_cols, plot_type="bar", show=False, plot_size=None)
    plt.tight_layout()
    fig.savefig(rpath("shap_barplot.png"), dpi=150, bbox_inches="tight")
    plt.close("all")

    importance = np.abs(shap_vals).mean(axis=0)
    feat_imp = pd.DataFrame({
        "feature": feat_cols,
        "mean_abs_shap": importance,
    }).sort_values("mean_abs_shap", ascending=False)
    feat_imp.to_csv(rpath("feature_importance.csv"), index=False)
    print(f"[Stage 8] SHAP analysis complete. Top 5: {feat_imp['feature'].head(5).tolist()}")
    return feat_imp
