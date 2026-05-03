import numpy as np
import pandas as pd
import pytest

from src.features import COLUMN_TYPES, build_preprocessor


def _toy_df():
    return pd.DataFrame({
        "fico": [700.0, np.nan, 680.0, 720.0],
        "annual_inc": [50000.0, 60000.0, np.nan, 80000.0],
        "open_acc": [5, np.nan, 3, 7],
        "purpose": ["car", "home", "car", "other"],
    })


def test_column_types_has_expected_keys():
    assert set(COLUMN_TYPES.keys()) == {"numeric", "count", "categorical"}
    for key, cols in COLUMN_TYPES.items():
        assert isinstance(cols, list)
        assert all(isinstance(c, str) for c in cols)


def test_preprocessor_imputes_and_encodes():
    pre = build_preprocessor(
        numeric_cols=["fico", "annual_inc"],
        count_cols=["open_acc"],
        cat_cols=["purpose"],
    )
    X = _toy_df()
    out = pre.fit_transform(X)
    assert out.shape[0] == 4
    # no NaNs remain
    assert not np.isnan(np.asarray(out, dtype=float)).any()


def test_preprocessor_does_not_leak_train_to_test():
    """Median computed on train must differ from median computed on train+test."""
    pre = build_preprocessor(
        numeric_cols=["fico"], count_cols=[], cat_cols=[],
    )
    train = pd.DataFrame({"fico": [600.0, 650.0, 700.0]})  # median 650
    test = pd.DataFrame({"fico": [np.nan]})
    pre.fit(train)
    out_train_only = pre.transform(test)[0, 0]
    pre.fit(pd.concat([train, pd.DataFrame({"fico": [800.0, 900.0]})]))
    out_train_plus = pre.transform(test)[0, 0]
    assert out_train_only != out_train_plus


def test_preprocessor_handles_unknown_categories():
    pre = build_preprocessor(
        numeric_cols=[], count_cols=[], cat_cols=["purpose"],
    )
    train = pd.DataFrame({"purpose": ["a", "b", "a"]})
    test = pd.DataFrame({"purpose": ["a", "c"]})  # "c" unseen
    pre.fit(train)
    out = pre.transform(test)
    # two train categories -> 2 columns; unknown encoded as all-zero row
    assert out.shape == (2, 2)
    assert out[1].sum() == 0.0
