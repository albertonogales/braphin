"""Stage 9 & 10 — Minimal Biomarker Discovery and EISS Construction."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.feature_selection import RFECV
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score
from .utils import rpath, RANDOM_SEED
from .prediction import _patient_aggregate


MAX_FEATURES = 5


def _lopo_auroc(X, y, groups, model):
    logo = LeaveOneGroupOut()
    y_true_all, y_prob_all = [], []
    for train_idx, test_idx in logo.split(X, y, groups):
        try:
            model.fit(X[train_idx], y[train_idx])
            prob = model.predict_proba(X[test_idx])[:, 1]
        except Exception:
            prob = np.zeros(len(test_idx))
        y_true_all.extend(y[test_idx])
        y_prob_all.extend(prob)
    y_true = np.array(y_true_all)
    y_prob = np.array(y_prob_all)
    return roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan


def discover_minimal_biomarkers(df: pd.DataFrame, feat_cols: list, feat_imp: pd.DataFrame):
    target = "synthetic_good_outcome"
    pat_df = _patient_aggregate(df, feat_cols, target)
    X = pat_df[feat_cols].values
    y = pat_df[target].values
    groups = pat_df["id_subject"].values

    # Baseline full AUROC
    baseline_auroc = _lopo_auroc(X, y, groups, LogisticRegression(max_iter=2000, random_state=RANDOM_SEED, solver="saga"))
    threshold_auroc = 0.95 * baseline_auroc

    records = []

    # Method 1 — SHAP / feature-importance greedy forward selection
    if feat_imp is not None and not feat_imp.empty:
        ranked_feats = [f for f in feat_imp["feature"].tolist() if f in feat_cols]
    else:
        # Fall back to variance ranking when SHAP is unavailable
        variances = X.var(axis=0)
        ranked_feats = [feat_cols[i] for i in np.argsort(variances)[::-1]]

    shap_subset = []
    for f in ranked_feats:
        shap_subset.append(f)
        idx = [feat_cols.index(f2) for f2 in shap_subset if f2 in feat_cols]
        if len(idx) >= 1:
            auroc = _lopo_auroc(X[:, idx], y, groups,
                                LogisticRegression(max_iter=2000, random_state=RANDOM_SEED, solver="saga"))
            records.append({"method": "SHAP_ranking", "n_features": len(shap_subset),
                             "features": ",".join(shap_subset), "AUROC": round(float(auroc), 4)})
            if auroc >= threshold_auroc and len(shap_subset) >= 2:
                break
    shap_best_subset = shap_subset[:MAX_FEATURES]

    # Method 2 — L1 Logistic Regression with class_weight to handle imbalance
    for C_val in [1.0, 0.5, 0.1]:
        l1_model = LogisticRegression(penalty="l1", C=C_val, max_iter=5000,
                                       random_state=RANDOM_SEED, solver="saga",
                                       class_weight="balanced")
        l1_model.fit(X, y)
        l1_coefs = np.abs(l1_model.coef_[0])
        l1_nonzero = [feat_cols[i] for i in np.argsort(l1_coefs)[::-1] if l1_coefs[i] > 1e-6]
        if l1_nonzero:
            break
    l1_subset = l1_nonzero[:MAX_FEATURES] if l1_nonzero else ranked_feats[:MAX_FEATURES]
    if l1_subset:
        idx = [feat_cols.index(f) for f in l1_subset]
        auroc = _lopo_auroc(X[:, idx], y, groups,
                            LogisticRegression(max_iter=2000, random_state=RANDOM_SEED, solver="saga"))
        records.append({"method": "L1_LogReg", "n_features": len(l1_subset),
                         "features": ",".join(l1_subset), "AUROC": round(float(auroc), 4)})

    # Method 3 — RFE with LR (single CV split for speed)
    try:
        from sklearn.feature_selection import RFE
        rfe_model = LogisticRegression(max_iter=2000, random_state=RANDOM_SEED,
                                        solver="saga", class_weight="balanced")
        rfe = RFE(rfe_model, n_features_to_select=MAX_FEATURES, step=1)
        rfe.fit(X, y.astype(int))
        support_mask = np.asarray(rfe.support_, dtype=bool)
        rfe_subset = [feat_cols[i] for i in range(len(feat_cols)) if support_mask[i]]
        if rfe_subset:
            idx = [feat_cols.index(f) for f in rfe_subset]
            auroc = _lopo_auroc(X[:, idx], y, groups,
                                LogisticRegression(max_iter=2000, random_state=RANDOM_SEED, solver="saga"))
            records.append({"method": "RFE", "n_features": len(rfe_subset),
                             "features": ",".join(rfe_subset), "AUROC": round(float(auroc), 4)})
    except Exception as e:
        print(f"  [Stage 9] RFE failed: {e}")
        rfe_subset = shap_best_subset

    min_df = pd.DataFrame(records)
    min_df.to_csv(rpath("minimal_feature_sets.csv"), index=False)
    print(f"[Stage 9] Minimal feature sets:\n{min_df.to_string(index=False)}")

    # Pick best subset (highest AUROC, <= MAX_FEATURES)
    valid = min_df[min_df["n_features"] <= MAX_FEATURES].sort_values("AUROC", ascending=False)
    if not valid.empty:
        best_row = valid.iloc[0]
        best_features = best_row["features"].split(",")
    else:
        best_features = shap_best_subset[:MAX_FEATURES]

    print(f"[Stage 9] Best minimal feature subset: {best_features}")
    return best_features, min_df


def build_eiss(df: pd.DataFrame, feat_cols: list, eiss_features: list):
    target = "synthetic_good_outcome"
    pat_df = _patient_aggregate(df, feat_cols, target)

    avail = [f for f in eiss_features if f in pat_df.columns]
    X = pat_df[avail].values
    y = pat_df[target].values

    model = LogisticRegression(penalty="l1", C=1.0, max_iter=5000,
                                random_state=RANDOM_SEED, solver="saga",
                                class_weight="balanced")
    model.fit(X, y)

    coefs = model.coef_[0]
    intercept = model.intercept_[0]

    # Raw log-odds scores
    raw_scores = X @ coefs + intercept
    # Normalize to 0-100
    s_min, s_max = raw_scores.min(), raw_scores.max()
    if s_max > s_min:
        eiss_scores = 100 * (raw_scores - s_min) / (s_max - s_min)
    else:
        eiss_scores = np.full(len(raw_scores), 50.0)

    pat_df["EISS"] = eiss_scores
    score_out = pat_df[["id_subject", target, "EISS"] + avail].copy()
    score_out.to_csv(rpath("patient_biomarker_scores.csv"), index=False)

    # Formula
    terms = []
    for f, c in sorted(zip(avail, coefs), key=lambda x: abs(x[1]), reverse=True):
        sign = "+" if c >= 0 else "-"
        terms.append(f"  {sign} {abs(c):.4f} * {f}")
    formula_lines = ["EISS (raw log-odds) =", f"  {intercept:.4f} (intercept)"] + terms
    formula_lines += [
        "",
        f"Normalized: EISS = 100 * (raw - {s_min:.4f}) / ({s_max:.4f} - {s_min:.4f})",
        "",
        "Range: 0–100 (higher = better predicted intervention success)",
    ]
    formula_txt = "\n".join(formula_lines)
    with open(rpath("eiss_formula.txt"), "w") as fh:
        fh.write(formula_txt)

    print("[Stage 10] EISS formula:")
    print(formula_txt)
    return pat_df, score_out
