"""Learning curves and bias-variance diagnosis.

- learning_curve_data fits the pipeline factory at multiple training-size
  fractions per fold and collects train/val scores.
- plot_learning_curve draws mean+/-std curves with a bias/variance annotation.
- diagnose returns one of {'high_bias', 'high_variance', 'good_fit', 'inconclusive'}.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def learning_curve_data(
    pipeline_factory: Callable[[], "Pipeline"],
    X_raw: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    train_sizes: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 1.0),
    metric: str = "AUPRC",
) -> pd.DataFrame:
    """For each fraction x each fold: fit on that fraction of the fold's
    training portion (sampled from the end to preserve chronology),
    evaluate on used-for-training + full validation slice.
    Returns long-form DataFrame: train_size, fold, train_score, val_score, metric.
    """
    rows: list[dict] = []
    for fold_i, (train_idx, val_idx) in enumerate(splits):
        X_tr_all, y_tr_all = X_raw.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X_raw.iloc[val_idx], y.iloc[val_idx]
        n = len(X_tr_all)
        for frac in train_sizes:
            k = max(10, int(round(n * frac)))
            X_tr = X_tr_all.iloc[-k:]
            y_tr = y_tr_all.iloc[-k:]
            pipe = pipeline_factory()
            pipe.fit(X_tr, y_tr)
            tr_score = average_precision_score(y_tr, pipe.predict_proba(X_tr)[:, 1])
            va_score = average_precision_score(y_va, pipe.predict_proba(X_va)[:, 1])
            rows.append({
                "train_size": float(frac), "fold": fold_i,
                "train_score": float(tr_score), "val_score": float(va_score),
                "metric": metric,
            })
    return pd.DataFrame(rows)


def diagnose(lc_df: pd.DataFrame, gap_threshold: float = 0.05, floor_threshold: float = 0.3) -> str:
    """Classify bias-variance regime from the largest-train-size row."""
    if len(lc_df) == 0 or lc_df["train_size"].max() == lc_df["train_size"].min():
        return "inconclusive"
    final = lc_df[lc_df["train_size"] == lc_df["train_size"].max()]
    train_mean = final["train_score"].mean()
    val_mean = final["val_score"].mean()
    gap = train_mean - val_mean
    if gap > gap_threshold:
        return "high_variance"
    if train_mean < floor_threshold and val_mean < floor_threshold:
        return "high_bias"
    return "good_fit"


def plot_learning_curve(lc_df: pd.DataFrame, title: str = "", ax=None, out_path: Path | None = None):
    """Mean+/-std train and validation curves versus train_size."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure
    agg = lc_df.groupby("train_size").agg(
        train_mean=("train_score", "mean"),
        train_std=("train_score", "std"),
        val_mean=("val_score", "mean"),
        val_std=("val_score", "std"),
    ).reset_index()
    ax.plot(agg["train_size"], agg["train_mean"], "-o", color="#4c72b0", label="train")
    ax.fill_between(agg["train_size"],
                    agg["train_mean"] - agg["train_std"],
                    agg["train_mean"] + agg["train_std"], alpha=0.2, color="#4c72b0")
    ax.plot(agg["train_size"], agg["val_mean"], "-o", color="#dd8452", label="val")
    ax.fill_between(agg["train_size"],
                    agg["val_mean"] - agg["val_std"],
                    agg["val_mean"] + agg["val_std"], alpha=0.2, color="#dd8452")
    diag = diagnose(lc_df)
    ax.set_title(f"{title} -- {diag}".strip(" -"))
    ax.set_xlabel("train size (fraction)")
    ax.set_ylabel(f"{lc_df['metric'].iloc[0]} score")
    ax.legend(loc="lower right")
    fig.tight_layout()
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
    return fig, ax
