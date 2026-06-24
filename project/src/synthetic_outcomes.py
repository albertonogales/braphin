"""Stage 2 — Synthetic Outcome Generation."""

import numpy as np
import pandas as pd
from .utils import rpath, dpath


def _adequacy(grp: pd.DataFrame) -> float:
    soz = grp["is_SOZ"].astype(bool)
    treated = grp["intervention"].isin(["resection", "thermocoagulation"])

    n_soz = soz.sum()
    n_non_soz = (~soz).sum()

    coverage = (soz & treated).sum() / n_soz if n_soz > 0 else 0.0
    overtreatment = ((~soz) & treated).sum() / n_non_soz if n_non_soz > 0 else 0.0

    adequacy = 0.8 * coverage - 0.2 * overtreatment
    return float(np.clip(adequacy, 0.0, 1.0))


def _engel(adequacy: float) -> int:
    if adequacy >= 0.90:
        return 1
    elif adequacy >= 0.75:
        return 2
    elif adequacy >= 0.50:
        return 3
    else:
        return 4


def _ilae(adequacy: float) -> int:
    if adequacy >= 0.90:
        return 1
    elif adequacy >= 0.80:
        return 2
    elif adequacy >= 0.70:
        return 3
    elif adequacy >= 0.60:
        return 4
    elif adequacy >= 0.50:
        return 5
    else:
        return 6


def generate_synthetic_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    patient_stats = []
    for subj, grp in df.groupby("id_subject"):
        adq = _adequacy(grp)
        patient_stats.append({
            "id_subject": subj,
            "adequacy_score": adq,
            "synthetic_engel": _engel(adq),
            "synthetic_ilae": _ilae(adq),
            "synthetic_good_outcome": int(adq >= 0.75),
            "n_channels": len(grp),
            "n_SOZ": grp["is_SOZ"].sum(),
            "n_treated": grp["intervention"].isin(["resection", "thermocoagulation"]).sum(),
        })

    outcome_df = pd.DataFrame(patient_stats)
    outcome_df.to_csv(rpath("synthetic_outcomes.csv"), index=False)

    # Merge back — every channel gets patient-level labels
    df = df.merge(
        outcome_df[["id_subject", "adequacy_score", "synthetic_engel",
                    "synthetic_ilae", "synthetic_good_outcome"]],
        on="id_subject", how="left",
    )

    n_good = outcome_df["synthetic_good_outcome"].sum()
    print(f"[Stage 2] Synthetic outcomes generated for {len(outcome_df)} patients.")
    print(f"          Good outcome: {n_good} ({100*n_good/len(outcome_df):.1f}%)")
    print(f"          Engel dist : {outcome_df['synthetic_engel'].value_counts().sort_index().to_dict()}")

    return df
