"""Stage 6 & 7 — Outcome Prediction and Feature Family Ablation."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              accuracy_score, precision_score, recall_score,
                              f1_score, balanced_accuracy_score,
                              cohen_kappa_score)
from sklearn.preprocessing import label_binarize
from .utils import rpath, get_family_cols, RANDOM_SEED

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


def _get_models(n_classes=2):
    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=RANDOM_SEED, solver="saga"),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, random_state=RANDOM_SEED,
            eval_metric="logloss", verbosity=0,
            use_label_encoder=False if n_classes == 2 else False,
        )
    if HAS_LGB:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=200, random_state=RANDOM_SEED, verbosity=-1
        )
    return models


def _binary_metrics(y_true, y_pred, y_prob):
    return {
        "AUROC": round(roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan, 4),
        "AUPRC": round(average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan, 4),
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }


def _multiclass_metrics(y_true, y_pred):
    kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic") if len(np.unique(y_true)) > 1 else np.nan
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Balanced_Accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
        "QWK": round(float(kappa), 4),
        "Macro_F1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
    }


def _patient_aggregate(df: pd.DataFrame, feat_cols: list, target: str):
    """Aggregate channel-level features to patient-level (median)."""
    agg_dict = {c: "median" for c in feat_cols}
    agg_dict[target] = "first"
    pat_df = df.groupby("id_subject").agg(agg_dict).reset_index()
    return pat_df


def lopo_evaluate(df: pd.DataFrame, feat_cols: list, target: str, binary: bool, model_name: str, model):
    pat_df = _patient_aggregate(df, feat_cols, target)
    X = pat_df[feat_cols].values
    y = pat_df[target].values
    groups = pat_df["id_subject"].values

    logo = LeaveOneGroupOut()
    y_true_all, y_pred_all, y_prob_all = [], [], []

    for train_idx, test_idx in logo.split(X, y, groups):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y[train_idx]
        try:
            model.fit(X_tr, y_tr)
            pred = model.predict(X_te)
            if binary:
                prob = model.predict_proba(X_te)[:, 1]
            else:
                prob = model.predict_proba(X_te)
        except Exception:
            pred = np.zeros(len(X_te))
            prob = np.zeros(len(X_te)) if binary else np.zeros((len(X_te), len(np.unique(y))))
        y_true_all.extend(y[test_idx])
        y_pred_all.extend(pred)
        y_prob_all.extend(prob if binary else list(prob))

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    if binary:
        y_prob_all = np.array(y_prob_all)
        return _binary_metrics(y_true_all, y_pred_all, y_prob_all)
    else:
        return _multiclass_metrics(y_true_all, y_pred_all)


def run_prediction(df: pd.DataFrame, feat_cols: list):
    targets = {
        "synthetic_good_outcome": True,   # binary
        "synthetic_engel": False,
        "synthetic_ilae": False,
    }

    records = []
    best_model_name = None
    best_auroc = -1.0
    best_model_obj = None

    for target, binary in targets.items():
        n_classes = df[target].nunique()
        models = _get_models(n_classes)
        for mname, model in models.items():
            print(f"  [Stage 6] {mname} | {target} ...", end=" ", flush=True)
            try:
                metrics = lopo_evaluate(df, feat_cols, target, binary, mname, model)
            except Exception as e:
                metrics = {"error": str(e)}
            row = {"target": target, "model": mname, **metrics}
            records.append(row)
            print(metrics)

            if binary and "AUROC" in metrics and not np.isnan(metrics["AUROC"]):
                if metrics["AUROC"] > best_auroc:
                    best_auroc = metrics["AUROC"]
                    best_model_name = mname
                    best_model_obj = model

    results_df = pd.DataFrame(records)
    results_df.to_csv(rpath("prediction_results.csv"), index=False)
    print(f"[Stage 6] Best model: {best_model_name} (AUROC={best_auroc:.4f})")
    return results_df, best_model_name, best_model_obj


def run_ablation(df: pd.DataFrame, all_feat_cols: list):
    families = get_family_cols(df)
    families["Full"] = all_feat_cols
    targets = {
        "synthetic_good_outcome": True,
        "synthetic_engel": False,
        "synthetic_ilae": False,
    }
    records = []
    models = _get_models()

    for family, cols in families.items():
        available = [c for c in cols if c in df.columns]
        if not available:
            continue
        for target, binary in targets.items():
            for mname, model in models.items():
                print(f"  [Stage 7] {family} | {mname} | {target} ...", end=" ", flush=True)
                try:
                    metrics = lopo_evaluate(df, available, target, binary, mname, model)
                except Exception as e:
                    metrics = {"error": str(e)}
                row = {"family": family, "n_features": len(available),
                       "target": target, "model": mname, **metrics}
                records.append(row)
                print(metrics.get("AUROC", metrics.get("Accuracy", "?")))

    ablation_df = pd.DataFrame(records)
    ablation_df.to_csv(rpath("feature_family_comparison.csv"), index=False)
    print("[Stage 7] Feature family ablation complete.")

    # Figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    binary_abl = ablation_df[ablation_df["target"] == "synthetic_good_outcome"]
    if not binary_abl.empty and "AUROC" in binary_abl.columns:
        pivot = binary_abl.pivot_table(index="family", columns="model", values="AUROC")
        fig, ax = plt.subplots(figsize=(10, 5))
        pivot.plot(kind="bar", ax=ax)
        ax.set_title("Feature Family Ablation — AUROC (good_outcome)")
        ax.set_ylabel("AUROC")
        ax.set_xlabel("Feature Family")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        fig.savefig(rpath("feature_family_comparison.png"), dpi=150)
        plt.close(fig)

    return ablation_df
