"""Feature engineering and preprocessing utilities.

- ``average_fico`` is the pre-Pipeline feature derivation kept from the
  original notebook 01 flow.
- ``COLUMN_TYPES`` plus ``build_preprocessor`` replace the old leaky
  ``impute_missing``: the preprocessor fits per fold inside a
  ``sklearn.Pipeline``, so train/test separation is structural.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder


COLUMN_TYPES: dict[str, list[str]] = {
    "numeric": [
        "fico", "annual_inc", "dti", "revol_util", "int_rate",
        "loan_amnt", "tot_hi_cred_lim", "avg_cur_bal", "total_bc_limit",
        "installment", "revol_bal", "total_acc", "mort_acc", "bc_open_to_buy",
        "bc_util", "mo_sin_old_il_acct", "mo_sin_old_rev_tl_op",
        "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl", "mths_since_recent_bc",
        "mths_since_recent_inq", "num_accts_ever_120_pd", "num_actv_bc_tl",
        "num_actv_rev_tl", "num_bc_sats", "num_bc_tl", "num_il_tl",
        "num_op_rev_tl", "num_rev_accts", "num_rev_tl_bal_gt_0",
        "num_sats", "num_tl_30dpd", "num_tl_90g_dpd_24m",
        "num_tl_op_past_12m", "pct_tl_nvr_dlq", "percent_bc_gt_75",
        "tot_coll_amt", "tot_cur_bal", "total_bal_ex_mort",
        "total_il_high_credit_limit", "total_rev_hi_lim",
    ],
    "count": [
        "open_acc", "pub_rec", "delinq_2yrs", "acc_open_past_24mths",
        "inq_last_6mths", "pub_rec_bankruptcies", "tax_liens",
        "chargeoff_within_12_mths", "collections_12_mths_ex_med",
        "acc_now_delinq",
    ],
    "categorical": [
        "purpose", "home_ownership", "grade", "sub_grade", "term",
        "emp_length", "verification_status", "application_type",
        "initial_list_status",
    ],
}


def average_fico(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with a single ``fico`` column = midpoint of fico range."""
    out = df.copy()
    if "fico_range_low" in out.columns and "fico_range_high" in out.columns:
        out["fico"] = (out["fico_range_low"] + out["fico_range_high"]) / 2.0
        out = out.drop(columns=["fico_range_low", "fico_range_high"])
    return out


def encode_categoricals(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Normalize categorical string values (lowercase + strip).

    NOTE: This is string normalization only. Actual one-hot encoding is
    handled by ``build_preprocessor`` inside the model Pipeline, so train
    and test encodings stay aligned without leakage.
    """
    out = df.copy()
    cols = columns if columns is not None else out.select_dtypes(include=["object"]).columns
    for c in cols:
        if c in out.columns:
            out[c] = out[c].astype("string").str.strip().str.lower()
    return out


def _strip_percent(X):
    """Strip trailing '%' and convert to float (handles string columns like revol_util)."""
    if hasattr(X, "iloc"):
        # DataFrame or Series — convert each column
        out = X.copy()
        for col in out.columns:
            if out[col].dtype == object:
                out[col] = (
                    out[col].astype(str)
                    .str.replace("%", "", regex=False)
                    .replace("nan", np.nan)
                    .astype(float)
                )
        return out
    # numpy array path
    result = np.empty(X.shape, dtype=float)
    for j in range(X.shape[1]):
        col = X[:, j]
        if col.dtype.kind in ("U", "O"):
            result[:, j] = pd.to_numeric(
                pd.Series(col).astype(str).str.replace("%", "", regex=False).replace("nan", np.nan),
                errors="coerce",
            ).to_numpy()
        else:
            result[:, j] = col.astype(float)
    return result


# Columns that arrive as "52.8%" strings and must be stripped before imputation.
_PERCENT_COLS = ["revol_util"]


def build_preprocessor(
    numeric_cols: list[str],
    count_cols: list[str],
    cat_cols: list[str],
) -> ColumnTransformer:
    """Per-column-type imputation + one-hot encoding.

    Numeric -> median imputation (robust to skew in lending features).
    Count   -> constant 0 fill (absence of the count).
    Categorical -> OHE with handle_unknown='ignore' so unseen categories
                  in transform land as all-zero instead of raising.

    Columns in ``_PERCENT_COLS`` (e.g. ``revol_util``) arrive as strings
    like '52.8%' from the raw CSV; they are routed through a
    strip-then-impute sub-pipeline before being treated as numeric.
    """
    # Split numeric cols: those that need percent-stripping vs plain numeric
    pct_cols = [c for c in _PERCENT_COLS if c in numeric_cols]
    plain_num_cols = [c for c in numeric_cols if c not in pct_cols]

    strip_then_impute = Pipeline([
        ("strip", FunctionTransformer(_strip_percent, validate=False,
                                      feature_names_out="one-to-one")),
        ("impute", SimpleImputer(strategy="median")),
    ])

    transformers = []
    if plain_num_cols:
        transformers.append(("num", SimpleImputer(strategy="median"), plain_num_cols))
    if pct_cols:
        transformers.append(("pct", strip_then_impute, pct_cols))
    transformers += [
        ("count", SimpleImputer(strategy="constant", fill_value=0), count_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ]

    return ColumnTransformer(transformers=transformers, remainder="drop")


def select_top_features(importances: pd.Series, n: int = 10) -> list[str]:
    """Return the top-n feature names sorted by descending importance."""
    return importances.sort_values(ascending=False).head(n).index.tolist()
