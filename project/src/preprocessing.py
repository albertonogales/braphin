"""Stage 3 & 4 — Missing Data Analysis and Feature Preparation."""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from .utils import rpath, dpath, get_feature_cols, get_family_cols, RANDOM_SEED


def missing_data_analysis(df: pd.DataFrame) -> list:
    feat_cols = get_feature_cols(df)
    miss = df[feat_cols].isnull().mean().sort_values(ascending=False)

    report = pd.DataFrame({
        "feature": miss.index,
        "missingness_pct": (miss.values * 100).round(2),
    })
    report.to_csv(rpath("missing_data_report.csv"), index=False)

    drop_cols = miss[miss > 0.9].index.tolist()
    print(f"[Stage 3] Features with >90% missing: {len(drop_cols)} — {drop_cols}")
    keep_cols = [c for c in feat_cols if c not in drop_cols]
    print(f"          Kept {len(keep_cols)} features")
    return keep_cols


def build_pipelines(df: pd.DataFrame, feat_cols: list):
    X = df[feat_cols].values

    pipe_A = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    pipe_B = Pipeline([
        ("imputer", KNNImputer(n_neighbors=5)),
        ("scaler", StandardScaler()),
    ])

    X_A = pipe_A.fit_transform(X)
    X_B = pipe_B.fit_transform(X)

    # Preserve all non-feature columns, replace feature columns with processed values
    df_A = df.copy()
    df_B = df.copy()
    for i, c in enumerate(feat_cols):
        df_A[c] = X_A[:, i]
        df_B[c] = X_B[:, i]

    df_A.to_pickle(dpath("processed_A_median.pkl"))
    df_B.to_pickle(dpath("processed_B_knn.pkl"))

    print(f"[Stage 4] Preprocessing pipelines applied. Shape: {X_A.shape}")
    return df_A, df_B, pipe_A, pipe_B
