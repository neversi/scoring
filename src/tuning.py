"""Optuna-based Bayesian hyperparameter tuning with inner time-series CV.

Design:
- Stratified subsample (``CONFIG.tuning_subsample``) is drawn once per study
  so trial-to-trial variance comes from hyperparameters, not from data.
- Inner 3-fold time-series CV on the subsample, using the same splitter as
  the outer reporting CV but with ``n_splits=CONFIG.n_splits_inner``.
- Objective: mean AUPRC across inner folds.
- Best params and study diagnostics (history + importance plots) saved to
  ``reports/best_params/`` and ``reports/tuning/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score

from src.config import CONFIG
from src.validation import make_time_series_splits


def _stratified_subsample(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    n: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Stratified-by-y subsample preserving positive-class ratio."""
    if len(X) <= n:
        return X.reset_index(drop=True), y.reset_index(drop=True), dates.reset_index(drop=True)

    rng = np.random.default_rng(seed)
    pos_idx = np.where(y.to_numpy() == 1)[0]
    neg_idx = np.where(y.to_numpy() == 0)[0]
    pos_ratio = len(pos_idx) / len(y)
    n_pos = int(round(n * pos_ratio))
    n_neg = n - n_pos

    pos_sample = rng.choice(pos_idx, size=min(n_pos, len(pos_idx)), replace=False)
    neg_sample = rng.choice(neg_idx, size=min(n_neg, len(neg_idx)), replace=False)
    take = np.sort(np.concatenate([pos_sample, neg_sample]))
    return (
        X.iloc[take].reset_index(drop=True),
        y.iloc[take].reset_index(drop=True),
        dates.iloc[take].reset_index(drop=True),
    )


def _inner_cv_auprc(
    pipeline_factory: Callable[[], "Pipeline"],
    X_sub: pd.DataFrame,
    y_sub: pd.Series,
    dates_sub: pd.Series,
) -> float:
    """Mean AUPRC across inner time-series folds on the subsample."""
    splits = make_time_series_splits(
        dates_sub,
        n_splits=CONFIG.n_splits_inner,
        embargo_months=CONFIG.embargo_months,
        min_train_months=CONFIG.min_train_months,
    )
    scores: list[float] = []
    for train_idx, val_idx in splits:
        pipe = pipeline_factory()
        pipe.fit(X_sub.iloc[train_idx], y_sub.iloc[train_idx])
        y_prob = pipe.predict_proba(X_sub.iloc[val_idx])[:, 1]
        scores.append(average_precision_score(y_sub.iloc[val_idx], y_prob))
    return float(np.mean(scores))


def _run_study(
    objective_fn: Callable[[optuna.Trial], float],
    n_trials: int,
    seed: int,
    study_name: str,
) -> optuna.Study:
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(
        direction="maximize", sampler=sampler, pruner=pruner, study_name=study_name,
    )
    study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=True)
    return study


def _save_study_artifacts(study: optuna.Study, model_name: str) -> None:
    """Write best_params JSON and history/importance PNGs."""
    params_dir = CONFIG.best_params_dir
    tuning_dir = CONFIG.reports_dir / "tuning"
    params_dir.mkdir(parents=True, exist_ok=True)
    tuning_dir.mkdir(parents=True, exist_ok=True)

    with (params_dir / f"{model_name}.json").open("w") as f:
        json.dump(study.best_params, f, indent=2)

    # Plots use optuna.visualization.matplotlib (bundled with optuna)
    import matplotlib.pyplot as plt
    from optuna.visualization.matplotlib import (
        plot_optimization_history,
        plot_param_importances,
    )

    fig = plot_optimization_history(study).figure
    fig.savefig(tuning_dir / f"{model_name}_history.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    try:
        fig = plot_param_importances(study).figure
        fig.savefig(tuning_dir / f"{model_name}_importance.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except (RuntimeError, ValueError):
        # Need >= ~5 completed trials with variation; skip if Optuna refuses.
        pass


from src.models import build_lgbm, build_lr, build_xgb


def _swap_pre(pipe, pre_fn):
    """Return pipeline with preprocessor replaced by pre_fn(), if provided."""
    if pre_fn is not None:
        pipe.steps[0] = ("pre", pre_fn())
    return pipe


def tune_xgb(
    X_raw: pd.DataFrame, y: pd.Series, dates: pd.Series,
    pre_fn=None,
) -> tuple[dict, optuna.Study]:
    X_sub, y_sub, dates_sub = _stratified_subsample(
        X_raw, y, dates, n=CONFIG.tuning_subsample, seed=CONFIG.optuna_seed,
    )

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 1000, step=50),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        )
        return _inner_cv_auprc(
            lambda: _swap_pre(build_xgb(y_train=y_sub, params=params), pre_fn),
            X_sub, y_sub, dates_sub,
        )

    study = _run_study(objective, CONFIG.n_trials_xgb, CONFIG.optuna_seed, "xgb")
    _save_study_artifacts(study, "xgb")
    return study.best_params, study


def tune_lgbm(
    X_raw: pd.DataFrame, y: pd.Series, dates: pd.Series,
    pre_fn=None,
) -> tuple[dict, optuna.Study]:
    X_sub, y_sub, dates_sub = _stratified_subsample(
        X_raw, y, dates, n=CONFIG.tuning_subsample, seed=CONFIG.optuna_seed,
    )

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 1000, step=50),
            num_leaves=trial.suggest_int("num_leaves", 15, 255),
            max_depth=trial.suggest_int("max_depth", -1, 12),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        )
        return _inner_cv_auprc(
            lambda: _swap_pre(build_lgbm(y_train=y_sub, params=params), pre_fn),
            X_sub, y_sub, dates_sub,
        )

    study = _run_study(objective, CONFIG.n_trials_lgbm, CONFIG.optuna_seed, "lgbm")
    _save_study_artifacts(study, "lgbm")
    return study.best_params, study


def tune_lr(
    X_raw: pd.DataFrame, y: pd.Series, dates: pd.Series,
    pre_fn=None,
) -> tuple[dict, optuna.Study]:
    X_sub, y_sub, dates_sub = _stratified_subsample(
        X_raw, y, dates, n=CONFIG.tuning_subsample, seed=CONFIG.optuna_seed,
    )

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            C=trial.suggest_float("C", 1e-3, 10.0, log=True),
            penalty=trial.suggest_categorical("penalty", ["l1", "l2"]),
            solver="saga",
        )
        return _inner_cv_auprc(
            lambda: _swap_pre(build_lr(y_train=y_sub, params=params), pre_fn),
            X_sub, y_sub, dates_sub,
        )

    study = _run_study(objective, CONFIG.n_trials_lr, CONFIG.optuna_seed, "lr")
    _save_study_artifacts(study, "lr")
    return study.best_params, study
