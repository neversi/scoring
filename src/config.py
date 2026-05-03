"""Project-wide configuration. Slide-relevant numbers and path constants.

All code that previously hardcoded values like 42, "2015-07-01", 300_000
should import CONFIG (or a freshly constructed Config) from this module.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Config:
    # temporal split (matches notebook 01's existing cut)
    train_end: str = "2015-07-01"
    test_end: str = "2018-12-31"
    date_col: str = "issue_d"
    target_col: str = "target"

    # cross-validation
    n_splits_outer: int = 5
    n_splits_inner: int = 3
    embargo_months: int = 3
    min_train_months: int = 24

    # tuning
    n_trials_xgb: int = 30
    n_trials_lgbm: int = 30
    n_trials_lr: int = 15
    tuning_subsample: int = 300_000
    n_bootstraps: int = 1000
    tuning_metric: str = "AUPRC"

    # reproducibility
    random_state: int = 42
    optuna_seed: int = 42
    n_jobs: int = 8  # leave 4 cores free on M4 Pro

    # paths (relative to project root)
    data_dir: Path = field(default_factory=lambda: Path("data"))
    processed_dir: Path = field(default_factory=lambda: Path("data/processed"))
    models_dir: Path = field(default_factory=lambda: Path("models"))
    reports_dir: Path = field(default_factory=lambda: Path("reports"))
    best_params_dir: Path = field(default_factory=lambda: Path("reports/best_params"))


CONFIG = Config()
