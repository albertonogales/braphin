"""Stage 1 — Duplicate Channel Resolution."""

import numpy as np
import pandas as pd
from .utils import rpath, dpath, ID_COLS, RANDOM_SEED


def _hfo_cols(df):
    return [c for c in ["HFO_R", "HFO_FR"] if c in df.columns]


def resolve_duplicates(df: pd.DataFrame):
    """
    Classify duplicate (id_subject, channel_name) pairs into types A/B/C,
    produce three alternative datasets, and save a resolution report.
    """
    key = ["id_subject", "channel_name"]
    df = df.copy()
    df["_dup_key"] = df["id_subject"].astype(str) + "||" + df["channel_name"].astype(str)

    groups = df.groupby("_dup_key")
    dup_keys = [k for k, g in groups if len(g) > 1]
    singleton_keys = [k for k, g in groups if len(g) == 1]

    hfo_cols = _hfo_cols(df)
    feat_cols = [c for c in df.columns
                 if c not in ID_COLS + ["id_institution", "institution_name",
                                         "intervention", "seizure_info", "imaging_pathology",
                                         "is_SOZ", "latest_ilae", "latest_engel",
                                         "good_outcome", "_dup_key"]]

    records = []
    resolved_rows_A = []   # keep all
    resolved_rows_B = []   # keep lower-HFO
    resolved_rows_C = []   # aggregated

    for key_val, grp in groups:
        if len(grp) == 1:
            row = grp.iloc[0].copy()
            row["soz_probability"] = float(row["is_SOZ"])
            row["dup_type"] = "none"
            resolved_rows_A.append(row)
            resolved_rows_B.append(row)
            resolved_rows_C.append(row)
            continue

        # Compute variability metrics
        num = grp[feat_cols].select_dtypes(include=np.number)
        cv = num.std() / (num.mean().replace(0, np.nan))
        mean_cv = cv.mean(skipna=True)

        soz_vals = grp["is_SOZ"].astype(int)
        soz_var = soz_vals.std()
        soz_prob = soz_vals.mean()

        if hfo_cols:
            hfo_vals = grp[hfo_cols].mean(axis=1)
            hfo_range = hfo_vals.max() - hfo_vals.min()
        else:
            hfo_range = 0.0

        # Classify
        if hfo_range > 5.0:
            dup_type = "C"
        elif mean_cv < 0.05:
            dup_type = "B"
        else:
            dup_type = "A"

        records.append({
            "dup_key": key_val,
            "n_obs": len(grp),
            "dup_type": dup_type,
            "mean_cv": round(mean_cv, 4),
            "soz_variability": round(float(soz_var), 4),
            "soz_probability": round(float(soz_prob), 4),
            "hfo_range": round(float(hfo_range), 4),
        })

        # Build aggregated row
        agg_row = grp.iloc[0].copy()
        agg_row["soz_probability"] = soz_prob
        agg_row["dup_type"] = dup_type

        num_grp = grp[feat_cols].select_dtypes(include=np.number)
        for c in num_grp.columns:
            agg_row[c] = grp[c].median(skipna=True)

        # Dataset A — keep all
        for _, r in grp.iterrows():
            r2 = r.copy()
            r2["soz_probability"] = soz_prob
            r2["dup_type"] = dup_type
            resolved_rows_A.append(r2)

        # Dataset B — keep lower-HFO observation
        if hfo_cols:
            lower_idx = grp[hfo_cols[0]].idxmin()
        else:
            lower_idx = grp.index[0]
        r2 = grp.loc[lower_idx].copy()
        r2["soz_probability"] = soz_prob
        r2["dup_type"] = dup_type
        resolved_rows_B.append(r2)

        # Dataset C — aggregated
        resolved_rows_C.append(agg_row)

    report_df = pd.DataFrame(records)
    report_df.to_csv(rpath("duplicate_resolution_report.csv"), index=False)
    print(f"[Stage 1] Duplicates detected: {len(records)}")
    type_counts = report_df["dup_type"].value_counts().to_dict() if not report_df.empty else {}
    print(f"          Type A={type_counts.get('A',0)}, B={type_counts.get('B',0)}, C={type_counts.get('C',0)}")

    ds_A = pd.DataFrame(resolved_rows_A).drop(columns=["_dup_key"], errors="ignore").reset_index(drop=True)
    ds_B = pd.DataFrame(resolved_rows_B).drop(columns=["_dup_key"], errors="ignore").reset_index(drop=True)
    ds_C = pd.DataFrame(resolved_rows_C).drop(columns=["_dup_key"], errors="ignore").reset_index(drop=True)

    ds_A.to_pickle(dpath("dataset_A.pkl"))
    ds_B.to_pickle(dpath("dataset_B.pkl"))
    ds_C.to_pickle(dpath("dataset_C.pkl"))

    print(f"          Dataset A={len(ds_A)}, B={len(ds_B)}, C={len(ds_C)} rows")
    return ds_A, ds_B, ds_C
