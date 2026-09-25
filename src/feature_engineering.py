"""Engineer features and construct longitudinal cohort tables."""

from typing import Tuple
import pandas as pd


def build_longitudinal_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Construct wide tables that pair baseline visits with day 28 and day 180 post-dose 2."""
    df_base = df[df['visit_reason'] == 'baseline'].drop_duplicates(subset=['pubid']).set_index('pubid')
    df_v2p28 = df[df['visit_reason'] == 'v2plus28'].drop_duplicates(subset=['pubid']).set_index('pubid')
    df_v2p180 = df[df['visit_reason'] == 'v2plus180'].drop_duplicates(subset=['pubid']).set_index('pubid')

    cols_to_keep = [
        'age', 'sex', 'exposure', 'vaccine_interval_days',
        '229e_s_corrected', 'hku1_s_corrected', 'nl63_s_corrected', 'oc43_s_corrected',
        's_noise', 'cov_2s', 'cov_2n'
    ]
    df_wide = df_base[[c for c in cols_to_keep if c in df_base.columns]].copy()

    df_wide['sex_f'] = (df_wide['sex'] == 'f').astype(int)
    df_wide['prior_inf'] = (df_wide['exposure'] == 'convalescent').astype(int)

    # Calculate the longitudinal response outcomes
    df_wide['mag_v2p28'] = df_v2p28['cov_2s']
    df_wide['mag_v2p180'] = df_v2p180['cov_2s']
    df_wide['decay_rate'] = df_wide['mag_v2p28'] - df_wide['mag_v2p180']

    df_magnitude = df_wide.dropna(subset=['mag_v2p28']).copy()
    df_decay = df_wide.dropna(subset=['decay_rate']).copy()

    return df_magnitude, df_decay


def engineer_systems_features(df_wide: pd.DataFrame) -> pd.DataFrame:
    """Calculate the Beta to Alpha ratio and the endemic breadth score."""
    df_out = df_wide.copy()

    # Calculate the Beta to Alpha ratio: (OC43 + HKU1) / (229E + NL63)
    beta_sum = df_out['oc43_s_corrected'].fillna(0) + df_out['hku1_s_corrected'].fillna(0)
    alpha_sum = df_out['229e_s_corrected'].fillna(0) + df_out['nl63_s_corrected'].fillna(0)
    df_out['beta_alpha_ratio'] = beta_sum / (alpha_sum + 1e-5)

    # Calculate the breadth score as the count of corrected titres above the cohort median
    hcov_cols = ['229e_s_corrected', 'hku1_s_corrected', 'nl63_s_corrected', 'oc43_s_corrected']
    for col in hcov_cols:
        med = df_out[col].median()
        df_out[f"{col}_high"] = (df_out[col] > med).astype(int)

    df_out['hcov_breadth'] = df_out[[f"{col}_high" for col in hcov_cols]].sum(axis=1)

    return df_out


def extract_extreme_phenotypes(
    df_feat: pd.DataFrame, quantile: float = 0.25
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Extract the top and bottom responder quartiles."""
    q_low = df_feat['mag_v2p28'].quantile(quantile)
    q_high = df_feat['mag_v2p28'].quantile(1.0 - quantile)

    df_extreme = df_feat[(df_feat['mag_v2p28'] <= q_low) | (df_feat['mag_v2p28'] >= q_high)].copy()
    df_extreme['target_class'] = (df_extreme['mag_v2p28'] >= q_high).astype(int)

    feature_cols = [
        'age', 'sex_f', 'prior_inf',
        '229e_s_corrected', 'hku1_s_corrected', 'nl63_s_corrected', 'oc43_s_corrected',
        'beta_alpha_ratio', 'hcov_breadth'
    ]

    X = df_extreme[feature_cols]
    y = df_extreme['target_class']

    return df_extreme, X, y
