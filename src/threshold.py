"""Threshold-selection strategies.

All functions sweep np.arange(0.05, 0.96, 0.01).

- threshold_youden: maximizes TPR - FPR.
- threshold_f1: maximizes F1.
- threshold_cost_minimization: minimizes FP*cost_fp + FN*cost_fn.
- threshold_expected_profit: maximizes total dollar profit for lending;
  assumes int_rate is annualized decimal, term is months. Simple-interest
  approximation, no amortization.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score


_GRID = np.arange(0.05, 0.96, 0.01)


def _confusion(y_true, y_prob, t: float):
    y_pred = (np.asarray(y_prob) >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn, fp, fn, tp


def threshold_youden(y_true, y_prob) -> float:
    best_t, best_j = 0.5, -np.inf
    for t in _GRID:
        tn, fp, fn, tp = _confusion(y_true, y_prob, t)
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        j = tpr - fpr
        if j > best_j:
            best_j, best_t = j, float(t)
    return best_t


def threshold_f1(y_true, y_prob) -> float:
    best_t, best_f1 = 0.5, -np.inf
    for t in _GRID:
        y_pred = (np.asarray(y_prob) >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def threshold_cost_minimization(
    y_true, y_prob, cost_fp: float = 1.0, cost_fn: float = 5.0,
) -> float:
    best_t, best_cost = 0.5, np.inf
    for t in _GRID:
        tn, fp, fn, tp = _confusion(y_true, y_prob, t)
        cost = fp * cost_fp + fn * cost_fn
        if cost < best_cost:
            best_cost, best_t = cost, float(t)
    return best_t


def threshold_expected_profit(
    y_true, y_prob, loan_amnts, int_rates, terms, recoveries,
) -> float:
    """Approve loans with score < t. Approved+paid = profit; approved+default = loss."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    loan_amnts = np.asarray(loan_amnts, dtype=float)
    int_rates = np.asarray(int_rates, dtype=float)
    terms = np.asarray(terms, dtype=float)
    recoveries = np.asarray(recoveries, dtype=float)

    profit_if_paid = loan_amnts * int_rates * (terms / 12.0)
    loss_if_default = -(loan_amnts - recoveries)

    best_t, best_profit = 0.5, -np.inf
    for t in _GRID:
        approved = y_prob < t
        paid = approved & (y_true == 0)
        defaulted = approved & (y_true == 1)
        total = profit_if_paid[paid].sum() + loss_if_default[defaulted].sum()
        if total > best_profit:
            best_profit, best_t = total, float(t)
    return best_t
