import numpy as np
import pytest

from src.evaluate import compute_metrics, bootstrap_metrics, metrics_table


def test_compute_metrics_includes_mcc():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=100)
    y_prob = rng.random(size=100)
    y_pred = (y_prob >= 0.5).astype(int)
    m = compute_metrics(y_true, y_pred, y_prob)
    assert "MCC" in m
    assert -1.0 <= m["MCC"] <= 1.0


def test_bootstrap_returns_ci_shape():
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, size=200)
    y_prob = rng.random(size=200)
    y_pred = (y_prob >= 0.5).astype(int)
    res = bootstrap_metrics(y_true, y_pred, y_prob, n_bootstraps=50, seed=42)
    for metric in ("AUPRC", "AUROC", "F1", "balanced_accuracy", "MCC"):
        assert metric in res
        entry = res[metric]
        assert set(entry.keys()) == {"mean", "ci_low", "ci_high"}
        assert entry["ci_low"] <= entry["mean"] <= entry["ci_high"]


def test_bootstrap_is_reproducible():
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, size=200)
    y_prob = rng.random(size=200)
    y_pred = (y_prob >= 0.5).astype(int)
    a = bootstrap_metrics(y_true, y_pred, y_prob, n_bootstraps=50, seed=42)
    b = bootstrap_metrics(y_true, y_pred, y_prob, n_bootstraps=50, seed=42)
    assert a["AUPRC"]["mean"] == b["AUPRC"]["mean"]
    assert a["AUPRC"]["ci_low"] == b["AUPRC"]["ci_low"]


def test_metrics_table_formats_ci_dicts():
    results = {
        "modelA": {
            "AUPRC": {"mean": 0.72, "ci_low": 0.71, "ci_high": 0.73},
            "AUROC": {"mean": 0.80, "ci_low": 0.79, "ci_high": 0.81},
            "F1":    {"mean": 0.40, "ci_low": 0.38, "ci_high": 0.42},
            "balanced_accuracy": {"mean": 0.65, "ci_low": 0.64, "ci_high": 0.66},
            "MCC":   {"mean": 0.22, "ci_low": 0.21, "ci_high": 0.23},
        }
    }
    tbl = metrics_table(results)
    cell = tbl.loc["modelA", "AUPRC"]
    assert "0.720" in str(cell) and "0.710" in str(cell) and "0.730" in str(cell)


def test_metrics_table_handles_point_estimates():
    results = {"modelA": {"AUPRC": 0.72, "AUROC": 0.80, "F1": 0.40,
                          "balanced_accuracy": 0.65, "MCC": 0.22}}
    tbl = metrics_table(results)
    cell = tbl.loc["modelA", "AUPRC"]
    assert cell == 0.72 or "0.72" in str(cell)
