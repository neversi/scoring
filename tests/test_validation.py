import numpy as np
import pandas as pd
import pytest

from src.validation import make_time_series_splits


def _synthetic_dates(n_months: int = 60, rows_per_month: int = 100) -> pd.Series:
    starts = pd.date_range("2010-01-01", periods=n_months, freq="MS")
    return pd.Series(np.repeat(starts, rows_per_month))


def test_returns_n_splits():
    dates = _synthetic_dates(60, 100)
    splits = make_time_series_splits(dates, n_splits=5, embargo_months=0, min_train_months=24)
    assert len(splits) == 5


def test_splits_are_chronological():
    dates = _synthetic_dates(60, 100)
    splits = make_time_series_splits(dates, n_splits=5, embargo_months=0, min_train_months=24)
    for train_idx, val_idx in splits:
        assert dates.iloc[train_idx].max() < dates.iloc[val_idx].min()


def test_embargo_gap_respected():
    dates = _synthetic_dates(60, 100)
    splits = make_time_series_splits(dates, n_splits=5, embargo_months=3, min_train_months=24)
    for train_idx, val_idx in splits:
        gap_days = (dates.iloc[val_idx].min() - dates.iloc[train_idx].max()).days
        assert gap_days >= 3 * 28  # 3 months, approximated to 28-day months (strict lower bound)


def test_min_train_months_respected():
    dates = _synthetic_dates(60, 100)
    splits = make_time_series_splits(dates, n_splits=5, embargo_months=0, min_train_months=24)
    first_train_idx, _ = splits[0]
    span = dates.iloc[first_train_idx].max() - dates.iloc[first_train_idx].min()
    assert span.days >= 24 * 28  # at least ~24 months


def test_indices_are_numpy_arrays():
    dates = _synthetic_dates(60, 100)
    splits = make_time_series_splits(dates, n_splits=3, embargo_months=0, min_train_months=12)
    for train_idx, val_idx in splits:
        assert isinstance(train_idx, np.ndarray)
        assert isinstance(val_idx, np.ndarray)


def test_no_overlap_between_train_and_val():
    dates = _synthetic_dates(60, 100)
    splits = make_time_series_splits(dates, n_splits=5, embargo_months=0, min_train_months=24)
    for train_idx, val_idx in splits:
        assert len(set(train_idx) & set(val_idx)) == 0


from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.validation import cross_validate_model


def _toy_pipeline_factory():
    return Pipeline([("clf", LogisticRegression(max_iter=200))])


def test_cross_validate_returns_expected_rows():
    dates = _synthetic_dates(60, 50)
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x": rng.normal(size=len(dates))})
    y = pd.Series((X["x"] > 0).astype(int))
    splits = make_time_series_splits(dates, n_splits=3, embargo_months=0, min_train_months=12)
    df = cross_validate_model(_toy_pipeline_factory, X, y, dates, splits)
    # 3 fold rows + 1 mean+/-std row
    assert len(df) == 4
    assert {"fold", "AUPRC", "AUROC", "F1", "balanced_accuracy"}.issubset(df.columns)
    assert df.iloc[-1]["fold"] == "mean±std"


def test_cross_validate_mean_row_formatted():
    dates = _synthetic_dates(60, 50)
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x": rng.normal(size=len(dates))})
    y = pd.Series((X["x"] > 0).astype(int))
    splits = make_time_series_splits(dates, n_splits=3, embargo_months=0, min_train_months=12)
    df = cross_validate_model(_toy_pipeline_factory, X, y, dates, splits)
    last = df.iloc[-1]
    # mean+/-std cells are strings with "±"
    assert isinstance(last["AUPRC"], str) and "±" in last["AUPRC"]
