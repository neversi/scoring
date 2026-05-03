"""Data drift detection: PSI, KS test, and simulation."""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two distributions.

    PSI < 0.1: stable. 0.1-0.25: moderate drift. > 0.25: significant drift.
    Uses reference quantiles as bin edges for consistent binning.
    """
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts = np.histogram(reference, bins=edges)[0]
    cur_counts = np.histogram(current, bins=edges)[0]

    # Laplace smoothing to avoid log(0)
    ref_pct = (ref_counts + 1) / (len(reference) + bins)
    cur_pct = (cur_counts + 1) / (len(current) + bins)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def compute_ks_test(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """Kolmogorov-Smirnov test for distribution shift. Returns (statistic, p_value)."""
    stat, pvalue = ks_2samp(reference, current)
    return float(stat), float(pvalue)


def compute_drift_report(
    X_reference: pd.DataFrame,
    X_current: pd.DataFrame,
    feature_names: list[str],
    psi_thresholds: tuple[float, float] = (0.1, 0.25),
    ks_alpha: float = 0.05,
) -> pd.DataFrame:
    """Compute PSI + KS for all features. Returns DataFrame with drift levels."""
    rows = []
    for feat in feature_names:
        ref = X_reference[feat].dropna().values
        cur = X_current[feat].dropna().values
        psi = compute_psi(ref, cur)
        ks_stat, ks_pval = compute_ks_test(ref, cur)

        if psi < psi_thresholds[0]:
            level = "no_drift"
        elif psi < psi_thresholds[1]:
            level = "moderate"
        else:
            level = "significant"

        rows.append({
            "feature": feat,
            "psi": round(psi, 4),
            "ks_statistic": round(ks_stat, 4),
            "ks_pvalue": round(ks_pval, 6),
            "ks_significant": ks_pval < ks_alpha,
            "drift_level": level,
        })
    return pd.DataFrame(rows)


def simulate_drift(X: pd.DataFrame, feature_shifts: dict[str, float]) -> pd.DataFrame:
    """Apply additive shifts to specified features. Returns modified copy."""
    X_shifted = X.copy()
    for feat, shift in feature_shifts.items():
        if feat in X_shifted.columns and shift != 0:
            X_shifted[feat] = X_shifted[feat] + shift
    return X_shifted


def drift_summary(drift_df: pd.DataFrame) -> dict:
    """Summarize a drift report into counts-per-level and a significance boolean.

    Expects a ``drift_level`` column with values in
    {"no_drift", "moderate", "significant"}. If absent, infers from ``psi``
    using the (0.1, 0.25) thresholds.
    """
    df = drift_df
    if "drift_level" not in df.columns and "psi" in df.columns:
        def _classify(psi):
            if psi < 0.1: return "no_drift"
            if psi < 0.25: return "moderate"
            return "significant"
        df = df.assign(drift_level=df["psi"].map(_classify))

    counts = df["drift_level"].value_counts().to_dict() if "drift_level" in df.columns else {}
    counts_full = {
        "no_drift":    int(counts.get("no_drift", 0)),
        "moderate":    int(counts.get("moderate", 0)),
        "significant": int(counts.get("significant", 0)),
    }
    return {
        "counts_per_level": counts_full,
        "any_significant": counts_full["significant"] > 0,
    }
