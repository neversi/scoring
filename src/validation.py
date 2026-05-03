"""Time-series cross-validation utilities.

Expanding-window splits carved on a date column (not row order), with
optional embargo gap between train and validation windows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta


def make_time_series_splits(
    dates: pd.Series,
    n_splits: int,
    embargo_months: int = 0,
    min_train_months: int = 24,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window time-series splits with optional embargo.

    Parameters
    ----------
    dates : pd.Series
        Date-valued series aligned with the rows of X and y.
    n_splits : int
        Number of (train_idx, val_idx) pairs to return.
    embargo_months : int
        Months to skip between the last train date and the first val date.
    min_train_months : int
        Minimum span (in months) of the first fold's training portion.

    Returns
    -------
    list of (train_idx, val_idx) numpy arrays. Row indices, not positional
    into any sorted view.
    """
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")

    dates = pd.to_datetime(dates)
    # unique month-start anchors over the full span
    months = pd.to_datetime(sorted({pd.Timestamp(d.year, d.month, 1) for d in dates}))

    total_months = len(months)
    val_months_remaining = total_months - min_train_months
    if val_months_remaining < n_splits:
        raise ValueError(
            f"Not enough months ({total_months}) for n_splits={n_splits} "
            f"with min_train_months={min_train_months}."
        )

    val_window_size = val_months_remaining // n_splits
    splits: list[tuple[np.ndarray, np.ndarray]] = []

    for i in range(n_splits):
        val_start_idx = min_train_months + i * val_window_size
        val_end_idx = val_start_idx + val_window_size if i < n_splits - 1 else total_months
        val_start = months[val_start_idx]
        val_end = months[val_end_idx - 1] + relativedelta(months=1)  # exclusive end

        train_end = val_start - relativedelta(months=embargo_months)

        train_mask = dates < train_end
        val_mask = (dates >= val_start) & (dates < val_end)

        train_idx = np.where(train_mask.to_numpy())[0]
        val_idx = np.where(val_mask.to_numpy())[0]

        if len(train_idx) == 0 or len(val_idx) == 0:
            raise ValueError(
                f"Fold {i} produced an empty train or val -- check "
                f"min_train_months and embargo_months relative to data span."
            )

        splits.append((train_idx, val_idx))

    return splits


from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


def _metric_values(y_true, y_pred, y_prob) -> dict[str, float]:
    return {
        "AUPRC": average_precision_score(y_true, y_prob),
        "AUROC": roc_auc_score(y_true, y_prob),
        "F1": f1_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }


def cross_validate_model(
    pipeline_factory: Callable[[], "Pipeline"],
    X_raw: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    model_name: str | None = None,
    reports_dir: Path | None = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Fit a fresh pipeline on each fold's train, score on each fold's val.

    Returns a DataFrame with one row per fold and a trailing ``mean±std``
    row whose cells are pre-formatted strings. If ``model_name`` and
    ``reports_dir`` are given, saves ``{reports_dir}/cv_results/{name}.csv``.
    """
    rows: list[dict] = []
    for i, (train_idx, val_idx) in enumerate(splits):
        X_tr, y_tr = X_raw.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X_raw.iloc[val_idx], y.iloc[val_idx]

        pipe = pipeline_factory()
        pipe.fit(X_tr, y_tr)

        y_prob = pipe.predict_proba(X_va)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
        metrics = _metric_values(y_va.to_numpy(), y_pred, y_prob)
        rows.append({"fold": i, **metrics})

    df = pd.DataFrame(rows)
    # mean±std summary row
    summary = {"fold": "mean±std"}
    for col in ("AUPRC", "AUROC", "F1", "balanced_accuracy"):
        m, s = df[col].mean(), df[col].std()
        summary[col] = f"{m:.4f} ± {s:.4f}"
    df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)

    if model_name and reports_dir is not None:
        out = Path(reports_dir) / "cv_results" / f"{model_name}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)

    return df


def plot_fold_timeline(
    dates: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    embargo_months: int = 0,
    out_path: Path | None = None,
):
    """Horizontal bar chart of each fold's train/val windows on the date axis.

    Embargo gaps are left visually empty so the methodology is legible in a
    presentation slide.
    """
    fig, ax = plt.subplots(figsize=(10, 0.6 * len(splits) + 1))
    dates = pd.to_datetime(dates)

    for i, (train_idx, val_idx) in enumerate(splits):
        tr_start, tr_end = dates.iloc[train_idx].min(), dates.iloc[train_idx].max()
        va_start, va_end = dates.iloc[val_idx].min(), dates.iloc[val_idx].max()
        ax.barh(i, (tr_end - tr_start).days, left=tr_start, color="#4c72b0", label="train" if i == 0 else None)
        ax.barh(i, (va_end - va_start).days, left=va_start, color="#dd8452", label="val" if i == 0 else None)

    ax.set_yticks(range(len(splits)))
    ax.set_yticklabels([f"Fold {i+1}" for i in range(len(splits))])
    ax.set_xlabel("issue_d")
    ax.set_title(f"Time-series CV folds (embargo = {embargo_months} months)")
    ax.legend(loc="lower right")
    fig.autofmt_xdate()
    fig.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
    return fig, ax
