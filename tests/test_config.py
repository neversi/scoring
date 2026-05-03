from pathlib import Path

from src.config import CONFIG, Config


def test_config_is_frozen():
    c = Config()
    try:
        c.random_state = 7
    except Exception:
        return
    raise AssertionError("Config must be frozen")


def test_config_defaults_match_spec():
    c = Config()
    assert c.train_end == "2015-07-01"
    assert c.test_end == "2018-12-31"
    assert c.date_col == "issue_d"
    assert c.target_col == "target"
    assert c.n_splits_outer == 5
    assert c.n_splits_inner == 3
    assert c.embargo_months == 3
    assert c.min_train_months == 24
    assert c.n_trials_xgb == 30
    assert c.n_trials_lgbm == 30
    assert c.n_trials_lr == 15
    assert c.tuning_subsample == 300_000
    assert c.tuning_metric == "AUPRC"
    assert c.random_state == 42
    assert c.optuna_seed == 42
    assert c.n_jobs == 8


def test_config_paths_are_path_objects():
    c = Config()
    assert isinstance(c.data_dir, Path)
    assert isinstance(c.processed_dir, Path)
    assert isinstance(c.models_dir, Path)
    assert isinstance(c.reports_dir, Path)
    assert isinstance(c.best_params_dir, Path)


def test_config_singleton_exists():
    assert isinstance(CONFIG, Config)
