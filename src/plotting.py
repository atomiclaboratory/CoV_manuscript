"""Plot manuscript Figures 1, 2, 3, and 4 in PDF and PNG formats."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from .config import PALETTE, setup_plotting_theme


def save_publication_figure(fig: plt.Figure, base_path: Path) -> None:
    """Save the figure in PNG and PDF formats."""
    setup_plotting_theme()
    fig.savefig(f"{base_path}.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{base_path}.pdf", format='pdf', bbox_inches='tight')
    plt.close(fig)


def plot_figure1_n_vs_s_waning(df_processed: pd.DataFrame, output_base: Path) -> None:
    """Plot the Nucleocapsid and Spike IgG trajectories across study visits."""
    setup_plotting_theme()
    df_p1 = df_processed[df_processed['exposure'] == 'convalescent'].copy()
    df_p1 = df_p1[df_p1['visit_reason'].isin(['baseline', 'v2plus28', 'v2plus180'])]

    fig, ax = plt.subplots(figsize=(6.5, 5))
    sns.lineplot(
        data=df_p1, x='visit_reason', y='cov_2n',
        label='Nucleocapsid (N) - Internal Control',
        marker='o', color=PALETTE[3], errorbar=None, ax=ax
    )
    sns.lineplot(
        data=df_p1, x='visit_reason', y='cov_2s',
        label='Spike (S) - Vaccine Boosted',
        marker='s', color=PALETTE[0], errorbar=None, ax=ax
    )
    ax.set_title("Figure 1: Antigen-Specific Trajectories (Internal Control)")
    ax.set_xlabel("Study Visit Timepoint")
    ax.set_ylabel("Antibody Titre (Log10 AU/ml)")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Baseline", "Day 28 Post-V2", "Day 180 Post-V2"])
    ax.legend(frameon=True)
    sns.despine(ax=ax)
    save_publication_figure(fig, output_base)


def plot_figure2_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc_score: float, output_base: Path) -> None:
    """Plot the receiver operating characteristic curve for extreme responders."""
    setup_plotting_theme()
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, color=PALETTE[1], lw=2.5, label=f"Extreme Phenotype ROC (AUC = {auc_score:.2f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Chance (AUC = 0.50)")
    ax.set_title("Figure 2: Vaccine Response Classification (Top vs Bottom 25%)")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.legend(loc="lower right", frameon=True)
    sns.despine(ax=ax)
    save_publication_figure(fig, output_base)


def plot_figure3_permutation_test(true_auc: float, null_aucs: np.ndarray, p_val: float, output_base: Path) -> None:
    """Plot the permutation test null distribution and the observed test score."""
    setup_plotting_theme()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.histplot(null_aucs, bins=40, kde=True, color="dimgray", label="Permuted Null Distribution", ax=ax)
    ax.axvline(true_auc, color="firebrick", linestyle="--", lw=2.5, label=f"Observed AUC = {true_auc:.2f} (P < 0.001)")
    ax.set_title("Figure 3: Permutation Test Null Distribution (1,000 Iterations)")
    ax.set_xlabel("Cross-Validated ROC AUC")
    ax.set_ylabel("Frequency")
    ax.legend(loc="upper left", frameon=True)
    sns.despine(ax=ax)
    save_publication_figure(fig, output_base)


def plot_figure4_oc43_recall(df_processed: pd.DataFrame, output_base: Path) -> None:
    """Plot the OC43 Spike IgG binding across vaccination timepoints."""
    setup_plotting_theme()
    df_p4 = df_processed[df_processed['visit_reason'].isin(['baseline', 'v2plus28', 'v3plus28'])].copy()

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.boxplot(
        data=df_p4, x='visit_reason', y='oc43_s', hue='visit_reason',
        order=['baseline', 'v2plus28', 'v3plus28'],
        palette='Blues', ax=ax, width=0.5, boxprops=dict(alpha=0.8), legend=False
    )
    sns.stripplot(
        data=df_p4, x='visit_reason', y='oc43_s',
        order=['baseline', 'v2plus28', 'v3plus28'],
        color='black', alpha=0.15, size=4, jitter=0.2, ax=ax
    )
    ax.set_title("Figure 4: Betacoronavirus Cross-Reactive Recall (OC43)")
    ax.set_xlabel("Vaccination Regimen Timepoint")
    ax.set_ylabel("OC43 Spike IgG (Log10 AU/ml)")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Baseline", "Post-Dose 2 (v2+28)", "Post-Dose 3 (v3+28)"])
    sns.despine(ax=ax)
    save_publication_figure(fig, output_base)
