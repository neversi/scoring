import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.features import COLUMN_TYPES
from src.models import MODEL_REGISTRY, build_xgb, build_dummy


def _toy_xy(n: int = 200):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "fico": rng.normal(700, 30, size=n),
        "annual_inc": rng.normal(60000, 10000, size=n),
        "open_acc": rng.integers(1, 10, size=n),
        "purpose": rng.choice(["car", "home"], size=n),
    })
    y = pd.Series(rng.integers(0, 2, size=n))
    return X, y


def test_registry_covers_expected_models():
    assert set(MODEL_REGISTRY.keys()) >= {"lr", "xgb", "lgbm", "ebm", "pltr"}


def test_builders_accept_unified_signature():
    X, y = _toy_xy()
    for name, builder in MODEL_REGISTRY.items():
        pipe = builder(y_train=y, params=None)
        assert isinstance(pipe, Pipeline)
        assert "pre" in dict(pipe.steps)
        assert "clf" in dict(pipe.steps)


def test_xgb_accepts_custom_params():
    _, y = _toy_xy()
    pipe = build_xgb(y_train=y, params={"n_estimators": 17, "max_depth": 2})
    clf = pipe.named_steps["clf"]
    # XGBClassifier parameters surfaced via get_params
    assert clf.get_params()["n_estimators"] == 17
    assert clf.get_params()["max_depth"] == 2


def test_dummy_registered():
    from src.models import MODEL_REGISTRY
    assert "dummy" in MODEL_REGISTRY


def test_dummy_returns_pipeline():
    _, y = _toy_xy()
    pipe = build_dummy(y_train=y, params=None)
    assert "pre" in dict(pipe.steps)
    assert "clf" in dict(pipe.steps)
    from sklearn.dummy import DummyClassifier
    assert isinstance(pipe.named_steps["clf"], DummyClassifier)


def test_dummy_strategy_from_params():
    pipe = build_dummy(y_train=None, params={"strategy": "most_frequent"})
    assert pipe.named_steps["clf"].strategy == "most_frequent"
