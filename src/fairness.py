"""Fairness metrics, proxy checks, and iterative attribute analysis."""

import logging

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
)


def disparate_impact_ratio(y_true, y_pred, sensitive) -> dict:
    """Compute disparate impact ratio per group vs. overall."""
    groups = pd.Series(sensitive).unique()
    overall_rate = np.mean(y_pred == 1)
    result = {}
    for g in groups:
        mask = sensitive == g
        group_rate = np.mean(y_pred[mask] == 1)
        result[g] = group_rate / overall_rate if overall_rate > 0 else np.nan
    return result


def compute_fairness_metrics(y_true, y_pred, sensitive) -> dict:
    """Compute all fairness metrics for a single sensitive attribute."""
    return {
        "demographic_parity_diff": demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive),
        "equalized_odds_diff": equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive),
        "disparate_impact": disparate_impact_ratio(y_true, y_pred, sensitive),
    }


def bin_continuous(series: pd.Series, bins: int = 3, labels: list = None) -> pd.Series:
    """Bin a continuous variable into discrete groups for fairness analysis."""
    return pd.qcut(series, q=bins, labels=labels, duplicates="drop")


def fairness_report(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    sensitive_map: dict,
    threshold: float = 0.5,
    min_group_size: int = 30,
) -> pd.DataFrame:
    """Compute per-(attribute, group) fairness metrics.

    Parameters
    ----------
    model : sklearn Pipeline with predict_proba.
    X_test : DataFrame passed unchanged to predict_proba.
    y_test : Ground-truth binary labels.
    sensitive_map : attribute-name -> pd.Series aligned with X_test's rows.
        e.g. {"grade": X_test["grade"], "fico_bin": bin_continuous(X_test["fico"])}
    threshold : Classification threshold for y_pred.
    min_group_size : Groups smaller than this get NaN metrics and a warning.

    Returns DataFrame with columns:
        attribute, group, group_size, selection_rate, TPR, FPR,
        demographic_parity_diff, equalized_odds_diff, disparate_impact_ratio.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    y_true = np.asarray(y_test)

    overall_selection = float(y_pred.mean())
    overall_tpr = float(y_pred[y_true == 1].mean()) if (y_true == 1).any() else np.nan
    overall_fpr = float(y_pred[y_true == 0].mean()) if (y_true == 0).any() else np.nan

    rows: list = []
    for attr_name, attr_series in sensitive_map.items():
        attr_values = pd.Series(attr_series).reset_index(drop=True)
        groups = attr_values.dropna().unique()
        for g in groups:
            mask = (attr_values == g).to_numpy()
            size = int(mask.sum())
            if size < min_group_size:
                logging.warning(
                    "fairness_report: attribute=%s group=%r size=%d < %d - metrics set to NaN",
                    attr_name, g, size, min_group_size,
                )
                rows.append({
                    "attribute": attr_name, "group": str(g), "group_size": size,
                    "selection_rate": np.nan, "TPR": np.nan, "FPR": np.nan,
                    "demographic_parity_diff": np.nan,
                    "equalized_odds_diff": np.nan,
                    "disparate_impact_ratio": np.nan,
                })
                continue

            y_pred_g = y_pred[mask]
            y_true_g = y_true[mask]
            sel = float(y_pred_g.mean())
            tpr = float(y_pred_g[y_true_g == 1].mean()) if (y_true_g == 1).any() else np.nan
            fpr = float(y_pred_g[y_true_g == 0].mean()) if (y_true_g == 0).any() else np.nan

            dp_diff = sel - overall_selection
            eo_diff = max(
                abs((tpr if not np.isnan(tpr) else 0.0) - (overall_tpr if not np.isnan(overall_tpr) else 0.0)),
                abs((fpr if not np.isnan(fpr) else 0.0) - (overall_fpr if not np.isnan(overall_fpr) else 0.0)),
            )
            di = (sel / overall_selection) if overall_selection > 0 else np.nan

            rows.append({
                "attribute": attr_name, "group": str(g), "group_size": size,
                "selection_rate": sel, "TPR": tpr, "FPR": fpr,
                "demographic_parity_diff": dp_diff,
                "equalized_odds_diff": eo_diff,
                "disparate_impact_ratio": di,
            })

    return pd.DataFrame(rows)
