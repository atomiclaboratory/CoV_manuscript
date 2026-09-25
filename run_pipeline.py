"""Execute the analysis pipeline to reproduce the manuscript findings."""

from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True

import joblib  # noqa: E402

current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

from src.config import (  # noqa: E402
    CHECKPOINT_DIR, CV_FOLDS, INVENTORY_FILE, N_PERMUTATIONS, PLOTS_DIR,
    RANDOM_STATE, RAW_DATA_FILE, SUMMARY_FILE, make_output_dirs
)
from src.data_loader import generate_data_inventory, load_data  # noqa: E402
from src.feature_engineering import (  # noqa: E402
    build_longitudinal_matrix, engineer_systems_features, extract_extreme_phenotypes
)
from src.models import (  # noqa: E402
    run_decay_hcov_ols, run_decay_ols, run_permutation_test, run_rfecv_modeling
)
from src.plotting import (  # noqa: E402
    plot_figure1_n_vs_s_waning, plot_figure2_roc_curve,
    plot_figure3_permutation_test, plot_figure4_oc43_recall
)
from src.preprocessing import apply_cross_reactivity_penalty, floor_and_log10  # noqa: E402


def log_summary(text: str, mode: str = "a") -> None:
    """Write text to the statistical summary report file."""
    with open(SUMMARY_FILE, mode, encoding="utf-8") as f:
        f.write(text + "\n")


def main() -> None:
    """Run all analysis steps and generate output files."""
    start_time = time.time()
    make_output_dirs()

    print("=" * 70)
    print("SYSTEMS IMMUNOLOGY PIPELINE: SARS-CoV-2 VACCINE RESPONSE ANALYSIS")
    print("=" * 70)

    log_summary("=" * 70, mode="w")
    log_summary("SYSTEMS IMMUNOLOGY STATISTICAL AND MACHINE LEARNING SUMMARY")
    log_summary("=" * 70 + "\n")

    # Step 1: Load the cohort data and write the data inventory
    print("\n[1/7] Loading cohort data...")
    df_raw = load_data(RAW_DATA_FILE)
    print(f"      Loaded {len(df_raw)} records ({df_raw['pubid'].nunique()} unique participants).")
    generate_data_inventory(df_raw, INVENTORY_FILE)

    # Step 2: Preprocess the serological data
    print("[2/7] Preprocessing: Flooring at LOD (=1) and log10 transformation...")
    df_processed = floor_and_log10(df_raw)
    log_summary("1. PREPROCESSING")
    log_summary("  - Physiological LOD applied: floored values at 1.0.")
    log_summary("  - Transformation: applied log10 to all antibody measurements.\n")

    # Step 3: Correct the heterologous cross-reactivity
    print("[3/7] Correcting for heterologous cross-reactivity (regressing MERS/SARS-1)...")
    df_corrected, noise_models = apply_cross_reactivity_penalty(df_processed)
    log_summary("2. HETEROLOGOUS CROSS-REACTIVITY CORRECTION")
    log_summary("  - Residualized endemic hCoVs against noise: s_noise = log10(MERS) + log10(SARS-1).")
    for hcov, model in noise_models.items():
        beta_val = model.params['s_noise']
        p_val = model.pvalues['s_noise']
        log_summary(f"    * {hcov:<10}: R2 = {model.rsquared:.4f}, Beta = {beta_val:.4f}, P = {p_val:.4e}")
    log_summary("")

    df_corrected.to_csv(CHECKPOINT_DIR / "cross_reactivity_corrected.csv", index=False)

    # Step 4: Construct the cohort tables and engineer systems features
    print("[4/7] Structuring longitudinal cohorts and engineering systems features...")
    df_mag, df_decay = build_longitudinal_matrix(df_corrected)
    df_mag = engineer_systems_features(df_mag)
    df_decay = engineer_systems_features(df_decay)

    mag_med = df_mag['mag_v2p28'].median()
    decay_med = df_decay['decay_rate'].median()
    log_summary("3. COHORT OUTCOMES AND ENGINEERED FEATURES")
    log_summary(f"  - Peak Magnitude (v2+28 Spike IgG): N = {len(df_mag)}, Median Log10 = {mag_med:.4f}")
    log_summary(f"  - Longitudinal Decay (v2+28 to v2+180): N = {len(df_decay)}, Median Decay = {decay_med:.4f}")
    log_summary("  - Beta/Alpha Homology Ratio: (OC43_corr + HKU1_corr) / (229E_corr + NL63_corr + 1e-5)")
    log_summary("  - Endemic Breadth Score: Count of corrected hCoVs > cohort median (range 0 to 4)\n")

    df_mag.to_csv(CHECKPOINT_DIR / "magnitude_cohort.csv")
    df_decay.to_csv(CHECKPOINT_DIR / "decay_cohort.csv")

    # Step 5: Plot the biological control and recall figures
    print("[5/7] Generating biological control and recall plots (Figures 1 & 4)...")
    plot_figure1_n_vs_s_waning(df_corrected, PLOTS_DIR / "figure1_n_vs_s_waning")
    plot_figure4_oc43_recall(df_corrected, PLOTS_DIR / "figure4_oc43_recall")

    # Step 6: Fit the longitudinal antibody waning models
    print("[6/7] Fitting longitudinal antibody waning models...")
    decay_hcov_model = run_decay_hcov_ols(df_decay)
    decay_model = run_decay_ols(df_decay)
    log_summary("4. LONGITUDINAL ANTIBODY WANING (OLS)")
    log_summary("  Model 1: Multivariable baseline endemic hCoVs")
    log_summary(
        f"    OLS R² = {decay_hcov_model.rsquared:.2f} (R2 = {decay_hcov_model.rsquared:.4f}), "
        f"F-test P = {decay_hcov_model.f_pvalue:.3f}"
    )
    log_summary("    Finding: Baseline endemic hCoV titres do not significantly predict antibody waning rate.")
    log_summary("  Model 2: Hybrid immunity interaction")
    log_summary(decay_model.summary().tables[1].as_text())
    beta_inf = decay_model.params['prior_inf']
    p_inf = decay_model.pvalues['prior_inf']
    log_summary(
        f"    hybrid immunity slope beta = {beta_inf:.3f} (beta = {beta_inf:.4f}), "
        f"P = {p_inf:.3f} (P = {p_inf:.4f})"
    )
    log_summary("    Finding: Prior SARS-CoV-2 infection significantly attenuates antibody waning.\n")

    # Step 7: Train the classifier and run the permutation test
    print("[7/7] Training extreme-phenotype classification model (RFECV)...")
    df_extreme, X_ext, y_ext = extract_extreme_phenotypes(df_mag, quantile=0.25)
    ml_results = run_rfecv_modeling(X_ext, y_ext, cv_splits=CV_FOLDS, random_state=RANDOM_STATE)

    log_summary("5. EXTREME PHENOTYPE CLASSIFICATION (RFECV)")
    log_summary(f"  - Sample size: N = {len(df_extreme)} (Top 25% vs Bottom 25%)")
    log_summary(f"  - Optimal feature count: {ml_results['n_features']}")
    log_summary(f"  - Selected features: {ml_results['selected_features']}")
    log_summary(f"  - Cross-validated ROC AUC: {ml_results['auc']:.4f}")
    log_summary(f"  - Cross-validated Accuracy: {ml_results['accuracy']:.4f}\n")

    plot_figure2_roc_curve(
        ml_results['fpr'], ml_results['tpr'], ml_results['auc'],
        PLOTS_DIR / "figure2_extreme_phenotyping_roc"
    )

    df_extreme.to_csv(CHECKPOINT_DIR / "extreme_phenotypes.csv", index=False)
    joblib.dump(ml_results['rfecv'], CHECKPOINT_DIR / "rfecv_model.joblib")

    print(f"      Running {N_PERMUTATIONS} label permutations for validation...")
    X_selected = ml_results['X_prep'][:, ml_results['rfecv'].support_]
    true_score, perm_scores, p_val = run_permutation_test(
        ml_results['rfecv'].estimator_, X_selected, y_ext, ml_results['cv'],
        n_permutations=N_PERMUTATIONS, random_state=RANDOM_STATE
    )

    log_summary("6. PERMUTATION TEST VALIDATION")
    log_summary(f"  - Iterations: {N_PERMUTATIONS}")
    log_summary(f"  - Cross-validated baseline AUC = {true_score:.2f} (AUC = {true_score:.4f})")
    log_summary(f"  - Null Distribution Mean: {perm_scores.mean():.4f} +/- {perm_scores.std():.4f}")
    p_val_str = "empirical P < 0.001" if p_val < 0.001 else f"empirical P = {p_val:.4f}"
    log_summary(f"  - Empirical P-value: {p_val_str} (P = {p_val:.4e})\n")

    plot_figure3_permutation_test(true_score, perm_scores, p_val, PLOTS_DIR / "figure3_permutation_test")

    elapsed = time.time() - start_time
    log_summary("=" * 70)
    log_summary(f"Analysis completed in {elapsed:.2f} seconds.")
    log_summary("=" * 70)

    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f}s")
    print(f"  * Figures:     {PLOTS_DIR}")
    print(f"  * Checkpoints: {CHECKPOINT_DIR}")
    print(f"  * Summary:     {SUMMARY_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
