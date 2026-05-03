import numpy as np
import pytest

from src.threshold import (
    threshold_youden,
    threshold_f1,
    threshold_cost_minimization,
    threshold_expected_profit,
)


def _toy(n: int = 1000, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    noise = rng.normal(0, 0.3, size=n)
    y_prob = np.clip(0.3 + 0.4 * y + noise, 0.0, 1.0)
    return y, y_prob


def test_youden_returns_valid_threshold():
    y, p = _toy()
    t = threshold_youden(y, p)
    assert 0.0 <= t <= 1.0


def test_f1_returns_valid_threshold():
    y, p = _toy()
    t = threshold_f1(y, p)
    assert 0.0 <= t <= 1.0


def test_cost_minimization_responds_to_asymmetry():
    """When FN cost is much higher than FP, optimal threshold should drop."""
    y, p = _toy(n=2000)
    t_sym = threshold_cost_minimization(y, p, cost_fp=1.0, cost_fn=1.0)
    t_asym = threshold_cost_minimization(y, p, cost_fp=1.0, cost_fn=10.0)
    assert t_asym <= t_sym


def test_expected_profit_on_toy_case():
    """Paid loans score low (safe), defaulters score high. Profit maximized around 0.5."""
    n = 100
    y = np.concatenate([np.zeros(50), np.ones(50)]).astype(int)
    y_prob = np.concatenate([np.linspace(0.05, 0.45, 50), np.linspace(0.55, 0.95, 50)])
    loan_amnts = np.full(n, 10000.0)
    int_rates = np.full(n, 0.10)
    terms = np.full(n, 36)
    recoveries = np.zeros(n)
    t = threshold_expected_profit(y, y_prob, loan_amnts, int_rates, terms, recoveries)
    assert 0.4 <= t <= 0.6


def test_youden_grid_robustness():
    y, p = _toy()
    t1 = threshold_youden(y, p)
    t2 = threshold_youden(y, np.clip(p * 0.5, 0.05, 0.95))
    assert 0.0 <= t1 <= 1.0
    assert 0.0 <= t2 <= 1.0
