"""Preprocess antibody titres and correct for non-specific binding."""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
import statsmodels.api as sm

ANTIBODY_COLUMNS = [
    '229e_s', 'cov_1s', 'cov2_rbd', 'cov_2n', 'cov_2s',
    'hku1_s', 'mers_s', 'nl63_s', 'oc43_s'
]

ENDEMIC_HCOVS = ['229e_s', 'hku1_s', 'nl63_s', 'oc43_s']


def floor_and_log10(df: pd.DataFrame) -> pd.DataFrame:
    """Floor antibody measurements at 1.0 and apply the log10 transformation."""
    df_out = df.copy()
    for col in ANTIBODY_COLUMNS:
        if col in df_out.columns:
            df_out[col] = np.log10(np.maximum(df_out[col], 1.0))
    return df_out


def apply_cross_reactivity_penalty(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Regress cross-reactive background binding from endemic coronavirus titres.

    Fit an ordinary least squares regression model:
        log10(hCoV) ~ beta_0 + beta_1 * s_noise
    Calculate the residuals as corrected antibody values.
    """
    df_out = df.copy()
    df_out['s_noise'] = df_out['mers_s'].fillna(0) + df_out['cov_1s'].fillna(0)

    models = {}
    for hcov in ENDEMIC_HCOVS:
        corr_col = f"{hcov}_corrected"
        temp_df = df_out[[hcov, 's_noise']].dropna()
        if len(temp_df) < 10:
            continue

        X = sm.add_constant(temp_df['s_noise'])
        y = temp_df[hcov]
        model = sm.OLS(y, X).fit()
        models[hcov] = model

        residuals = model.resid
        df_out.loc[residuals.index, corr_col] = residuals

    return df_out, models
