"""Model builder factory.

Every builder has the same signature ``(y_train=None, params=None) -> Pipeline``
and returns a two-step ``Pipeline([("pre", preprocessor), ("clf", estimator)])``.

- ``y_train`` is used only where class-imbalance calculation is needed
  (scale_pos_weight for XGB/LGBM). Other builders accept-and-ignore for
  API uniformity.
- ``params`` merges into the classifier constructor, so tuned
  hyperparameters loaded from ``reports/best_params/{model}.json`` flow
  in without per-model branching.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import CONFIG
from src.features import COLUMN_TYPES, build_preprocessor


def _make_pre():
    return build_preprocessor(
        numeric_cols=COLUMN_TYPES["numeric"],
        count_cols=COLUMN_TYPES["count"],
        cat_cols=COLUMN_TYPES["categorical"],
    )


def _scale_pos_weight(y) -> float:
    if y is None:
        return 1.0
    y = np.asarray(y)
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    return float(neg / max(pos, 1))


def build_lr(y_train=None, params: dict | None = None) -> Pipeline:
    defaults = dict(
        C=1.0, penalty="l2", solver="saga", class_weight="balanced",
        max_iter=1000, random_state=CONFIG.random_state, n_jobs=CONFIG.n_jobs,
    )
    merged = {**defaults, **(params or {})}
    return Pipeline([("pre", _make_pre()), ("clf", LogisticRegression(**merged))])


def build_xgb(y_train=None, params: dict | None = None) -> Pipeline:
    defaults = dict(
        n_estimators=400, max_depth=6, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9,
        tree_method="hist", eval_metric="aucpr",
        scale_pos_weight=_scale_pos_weight(y_train),
        random_state=CONFIG.random_state, seed=CONFIG.random_state,
        n_jobs=CONFIG.n_jobs,
    )
    merged = {**defaults, **(params or {})}
    return Pipeline([("pre", _make_pre()), ("clf", XGBClassifier(**merged))])


def build_lgbm(y_train=None, params: dict | None = None) -> Pipeline:
    defaults = dict(
        n_estimators=400, num_leaves=63, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        scale_pos_weight=_scale_pos_weight(y_train),
        deterministic=True, force_col_wise=True,
        random_state=CONFIG.random_state, seed=CONFIG.random_state,
        n_jobs=CONFIG.n_jobs,
    )
    merged = {**defaults, **(params or {})}
    return Pipeline([("pre", _make_pre()), ("clf", LGBMClassifier(**merged))])


def build_ebm(y_train=None, params: dict | None = None) -> Pipeline:
    defaults = dict(random_state=CONFIG.random_state, n_jobs=CONFIG.n_jobs)
    merged = {**defaults, **(params or {})}
    return Pipeline([("pre", _make_pre()), ("clf", ExplainableBoostingClassifier(**merged))])


def build_pltr(y_train=None, params: dict | None = None) -> Pipeline:
    from src.pltr import PLTR
    defaults = dict(random_state=CONFIG.random_state)
    merged = {**defaults, **(params or {})}
    return Pipeline([("pre", _make_pre()), ("clf", PLTR(**merged))])


def build_dummy(y_train=None, params: dict | None = None) -> Pipeline:
    """Stratified-by-default dummy classifier, wrapped in the standard Pipeline."""
    strategy = (params or {}).get("strategy", "stratified")
    merged = {"strategy": strategy, "random_state": CONFIG.random_state}
    return Pipeline([("pre", _make_pre()), ("clf", DummyClassifier(**merged))])


MODEL_REGISTRY: dict[str, Callable[..., Pipeline]] = {
    "lr": build_lr,
    "xgb": build_xgb,
    "lgbm": build_lgbm,
    "ebm": build_ebm,
    "pltr": build_pltr,
    "dummy": build_dummy,
}
