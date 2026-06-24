"""
Brno Epilepsy Dataset — Full Research Pipeline
Run with: python main.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Make sure src is importable
sys.path.insert(0, os.path.dirname(__file__))

from src.utils import load_raw, get_feature_cols, RESULTS_DIR, DATA_DIR
from src.duplicate_resolution import resolve_duplicates
from src.synthetic_outcomes import generate_synthetic_outcomes
from src.preprocessing import missing_data_analysis, build_pipelines
from src.exploratory import run_exploratory
from src.prediction import run_prediction, run_ablation
from src.explainability import run_explainability
from src.biomarker import discover_minimal_biomarkers, build_eiss
from src.som_analysis import run_som
from src.robustness import robustness_analysis

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "component_planes"), exist_ok=True)


def main():
    print("=" * 70)
    print("Brno Epilepsy Pipeline — Starting")
    print("=" * 70)

    # ------------------------------------------------------------------ #
    # Load raw data
    # ------------------------------------------------------------------ #
    print("\n[Load] Reading dataset...")
    df_raw = load_raw()
    print(f"       Shape: {df_raw.shape}  |  Subjects: {df_raw['id_subject'].nunique()}")

    # ------------------------------------------------------------------ #
    # Stage 1 — Duplicate resolution
    # ------------------------------------------------------------------ #
    print("\n--- Stage 1: Duplicate Resolution ---")
    ds_A, ds_B, ds_C = resolve_duplicates(df_raw)
    # Use Dataset A (keep all) for main pipeline; B and C used in robustness
    df = ds_A.copy()

    # ------------------------------------------------------------------ #
    # Stage 2 — Synthetic outcomes
    # ------------------------------------------------------------------ #
    print("\n--- Stage 2: Synthetic Outcome Generation ---")
    df = generate_synthetic_outcomes(df)

    # Propagate synthetic outcomes to ds_B and ds_C for robustness
    out_cols = ["id_subject", "adequacy_score", "synthetic_engel",
                "synthetic_ilae", "synthetic_good_outcome"]
    outcome_map = df[out_cols].drop_duplicates("id_subject")
    for ds in [ds_B, ds_C]:
        for c in out_cols[1:]:
            if c in ds.columns:
                ds.drop(columns=[c], inplace=True)
        ds_merged = ds.merge(outcome_map, on="id_subject", how="left")
        ds.update(ds_merged)

    # ------------------------------------------------------------------ #
    # Stage 3 — Missing data analysis
    # ------------------------------------------------------------------ #
    print("\n--- Stage 3: Missing Data Analysis ---")
    feat_cols = missing_data_analysis(df)

    # ------------------------------------------------------------------ #
    # Stage 4 — Preprocessing
    # ------------------------------------------------------------------ #
    print("\n--- Stage 4: Feature Preparation ---")
    df_med, df_knn, pipe_A, pipe_B = build_pipelines(df, feat_cols)
    # Use median-imputed for main analysis
    df_proc = df_med

    # ------------------------------------------------------------------ #
    # Stage 5 — Exploratory analysis
    # ------------------------------------------------------------------ #
    print("\n--- Stage 5: Exploratory Analysis ---")
    run_exploratory(df_proc, feat_cols)

    # ------------------------------------------------------------------ #
    # Stage 6 — Prediction
    # ------------------------------------------------------------------ #
    print("\n--- Stage 6: Outcome Prediction (LOPO) ---")
    pred_results, best_model_name, best_model = run_prediction(df_proc, feat_cols)

    # ------------------------------------------------------------------ #
    # Stage 7 — Ablation
    # ------------------------------------------------------------------ #
    print("\n--- Stage 7: Feature Family Ablation ---")
    ablation_results = run_ablation(df_proc, feat_cols)

    # ------------------------------------------------------------------ #
    # Stage 8 — Explainability
    # ------------------------------------------------------------------ #
    print("\n--- Stage 8: Explainability (SHAP) ---")
    feat_imp = run_explainability(df_proc, feat_cols, best_model_name, best_model)

    # ------------------------------------------------------------------ #
    # Stage 9 — Minimal biomarker discovery
    # ------------------------------------------------------------------ #
    print("\n--- Stage 9: Minimal Biomarker Discovery ---")
    eiss_features, min_sets = discover_minimal_biomarkers(df_proc, feat_cols, feat_imp)

    # ------------------------------------------------------------------ #
    # Stage 10 — EISS construction
    # ------------------------------------------------------------------ #
    print("\n--- Stage 10: EISS Construction ---")
    pat_df, eiss_scores = build_eiss(df_proc, feat_cols, eiss_features)

    # ------------------------------------------------------------------ #
    # Stage 11 — SOM analysis
    # ------------------------------------------------------------------ #
    print("\n--- Stage 11: SOM Analysis ---")
    run_som(df_proc, feat_cols, eiss_features)

    # ------------------------------------------------------------------ #
    # Stage 12 — Robustness
    # ------------------------------------------------------------------ #
    print("\n--- Stage 12: Robustness Analysis ---")
    robustness_analysis(feat_cols, eiss_features)

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("Pipeline complete. Results saved to:", RESULTS_DIR)
    print("=" * 70)

    # List output files
    for f in sorted(os.listdir(RESULTS_DIR)):
        fpath = os.path.join(RESULTS_DIR, f)
        if os.path.isfile(fpath):
            size_kb = os.path.getsize(fpath) / 1024
            print(f"  {f:50s}  {size_kb:6.1f} KB")


if __name__ == "__main__":
    main()
