"""Evaluation metrics: AUROC, AUPRC, F1, Balanced Accuracy."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
)


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    """Compute all four paper metrics.

    Args:
        y_true: ground truth labels (0/1)
        y_pred: predicted labels (0/1)
        y_prob: predicted probabilities for class 1
    """
    return {
        "AUROC": roc_auc_score(y_true, y_prob),
        "AUPRC": average_precision_score(y_true, y_prob),
        "F1": f1_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def metrics_table(results: dict[str, dict]) -> pd.DataFrame:
    """Format a model->metrics mapping as a DataFrame.

    If metric values are dicts with mean/ci_low/ci_high keys, cells are
    formatted as "0.720 [0.710, 0.730]". Otherwise, point estimates pass through.
    """
    rows = {}
    for name, metrics in results.items():
        row = {}
        for k, v in metrics.items():
            if isinstance(v, dict) and {"mean", "ci_low", "ci_high"}.issubset(v.keys()):
                row[k] = f"{v['mean']:.3f} [{v['ci_low']:.3f}, {v['ci_high']:.3f}]"
            else:
                row[k] = v
        rows[name] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def bootstrap_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Bootstrap 95% CIs for the five metrics in compute_metrics.

    For each of n_bootstraps resamples (with replacement), recomputes every
    metric; returns {metric: {"mean", "ci_low", "ci_high"}} using 2.5/97.5 percentiles.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    rng = np.random.default_rng(seed)

    # Map output keys to compute_metrics keys
    metric_mapping = {
        "AUPRC": "AUPRC",
        "AUROC": "AUROC",
        "F1": "F1",
        "balanced_accuracy": "Balanced Accuracy",
        "MCC": "MCC",
    }
    samples: dict[str, list[float]] = {k: [] for k in metric_mapping.keys()}

    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        try:
            m = compute_metrics(y_true[idx], y_pred[idx], y_prob[idx])
        except ValueError:
            continue
        for out_key, comp_key in metric_mapping.items():
            samples[out_key].append(m[comp_key])

    result: dict[str, dict[str, float]] = {}
    for k, vals in samples.items():
        arr = np.asarray(vals)
        result[k] = {
            "mean": float(arr.mean()),
            "ci_low": float(np.percentile(arr, 2.5)),
            "ci_high": float(np.percentile(arr, 97.5)),
        }
    return result


def compute_performance_over_time(
    model,
    X_slices: dict[str, pd.DataFrame],
    y_slices: dict[str, pd.Series],
    threshold: float,
) -> pd.DataFrame:
    """Compute model metrics per time slice."""
    rows = []
    for period, X in X_slices.items():
        y = y_slices[period]
        y_prob = model.predict_proba(X)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
        metrics = compute_metrics(y, y_pred, y_prob)
        metrics["period"] = period
        metrics["n_samples"] = len(y)
        metrics["default_rate"] = round(float(y.mean()), 4)
        rows.append(metrics)
    return pd.DataFrame(rows)
