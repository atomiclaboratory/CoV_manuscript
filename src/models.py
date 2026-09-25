"""Fit statistical models and execute machine learning pipelines."""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict, permutation_test_score
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm


def run_decay_hcov_ols(df_decay: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Fit a multivariable OLS regression of antibody decay against baseline coronavirus titres."""
    features = [
        'age', 'sex_f', 'prior_inf',
        '229e_s_corrected', 'hku1_s_corrected', 'nl63_s_corrected', 'oc43_s_corrected',
        's_noise'
    ]
    sub = df_decay.dropna(subset=['decay_rate'] + features).copy()
    X = sm.add_constant(sub[features])
    y = sub['decay_rate']
    return sm.OLS(y, X).fit()


def run_decay_ols(df_decay: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Fit an OLS regression of antibody decay against sex and prior infection status."""
    sub = df_decay.dropna(subset=['decay_rate', 'sex_f', 'prior_inf']).copy()
    X = sm.add_constant(sub[['sex_f', 'prior_inf']])
    X['sex_x_prior'] = X['sex_f'] * X['prior_inf']
    y = sub['decay_rate']
    return sm.OLS(y, X).fit()


def run_rfecv_modeling(X: pd.DataFrame, y: pd.Series, cv_splits: int = 5, random_state: int = 42) -> Dict:
    """Select features and evaluate the classifier using cross-validation."""
    imputer = KNNImputer(n_neighbors=5)
    scaler = StandardScaler()
    X_prep = scaler.fit_transform(imputer.fit_transform(X))

    clf = LogisticRegression(
        penalty='l1',
        solver='liblinear',
        random_state=random_state,
        class_weight='balanced'
    )
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    rfecv = RFECV(estimator=clf, step=1, cv=cv, scoring='roc_auc')
    rfecv.fit(X_prep, y)

    selected_features = list(X.columns[rfecv.support_])

    # Calculate cross-validated predictions with the selected features
    X_selected = X_prep[:, rfecv.support_]
    y_pred_proba = cross_val_predict(rfecv.estimator_, X_selected, y, cv=cv, method='predict_proba')[:, 1]

    auc = roc_auc_score(y, y_pred_proba)
    acc = accuracy_score(y, (y_pred_proba > 0.5).astype(int))
    fpr, tpr, thresholds = roc_curve(y, y_pred_proba)

    return {
        'rfecv': rfecv,
        'selected_features': selected_features,
        'n_features': rfecv.n_features_,
        'auc': auc,
        'accuracy': acc,
        'y_pred_proba': y_pred_proba,
        'fpr': fpr,
        'tpr': tpr,
        'X_prep': X_prep,
        'cv': cv
    }


def run_permutation_test(
    clf, X_selected, y, cv, n_permutations: int = 1000, random_state: int = 42
) -> Tuple[float, np.ndarray, float]:
    """Calculate the empirical significance using label permutation tests."""
    score, perm_scores, pvalue = permutation_test_score(
        clf, X_selected, y,
        scoring="roc_auc",
        cv=cv,
        n_permutations=n_permutations,
        n_jobs=-1,
        random_state=random_state
    )
    return score, perm_scores, pvalue
