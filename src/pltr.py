"""Penalized Logistic Tree Regression (PLTR) — sklearn implementation.

Two-step process from arXiv:2509.11389:
1. Single-split decision trees create binary features.
   Two-split trees create pairwise interaction features.
2. Adaptive lasso penalized logistic regression on expanded feature set.
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin


class PLTR(BaseEstimator, ClassifierMixin):
    """Penalized Logistic Tree Regression."""

    def __init__(self, C=1.0, max_interactions=None, random_state=42):
        self.C = C
        self.max_interactions = max_interactions
        self.random_state = random_state

    def _learn_thresholds(self, X, y):
        """Fit single-split trees to learn thresholds (fit-time only)."""
        self.thresholds_ = []
        self.valid_threshold_indices_ = []
        for j in range(X.shape[1]):
            tree = DecisionTreeClassifier(max_depth=1, random_state=self.random_state)
            tree.fit(X[:, [j]], y)
            if tree.tree_.feature[0] == -2:  # no valid split
                self.thresholds_.append(None)
            else:
                self.thresholds_.append(tree.tree_.threshold[0])
                self.valid_threshold_indices_.append(j)

        # Determine interaction pairs
        n_features = X.shape[1]
        max_pairs = self.max_interactions or n_features
        self.interaction_pairs_ = []
        for j in range(min(n_features, max_pairs)):
            if self.thresholds_[j] is None:
                continue
            for q in range(j + 1, n_features):
                if self.thresholds_[q] is None:
                    continue
                self.interaction_pairs_.append((j, q))

    def _apply_thresholds(self, X):
        """Apply learned thresholds to create expanded features (fit or predict time)."""
        # Binary features from single-split thresholds
        binary_features = []
        for j in self.valid_threshold_indices_:
            binary_features.append((X[:, j] > self.thresholds_[j]).astype(float))

        # Interaction features
        interaction_features = []
        for j, q in self.interaction_pairs_:
            feat = ((X[:, j] <= self.thresholds_[j]) & (X[:, q] > self.thresholds_[q])).astype(float)
            interaction_features.append(feat)

        parts = [X]
        if binary_features:
            parts.append(np.column_stack(binary_features))
        if interaction_features:
            parts.append(np.column_stack(interaction_features))
        return np.hstack(parts)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        # Step 1: learn thresholds from training data
        self._learn_thresholds(X, y)

        # Expand features
        X_expanded = self._apply_thresholds(X)

        # Step 2: initial L1 fit for adaptive weights
        lr_init = LogisticRegression(
            C=self.C, solver="saga", l1_ratio=1.0,
            max_iter=2000, random_state=self.random_state,
            class_weight="balanced",
        )
        lr_init.fit(X_expanded, y)

        # Step 3: adaptive lasso — reweight by 1/|coef|
        coefs = np.abs(lr_init.coef_[0]) + 1e-8
        weights = 1.0 / coefs
        X_reweighted = X_expanded / weights

        self.model_ = LogisticRegression(
            C=self.C, solver="saga", l1_ratio=1.0,
            max_iter=2000, random_state=self.random_state,
            class_weight="balanced",
        )
        self.model_.fit(X_reweighted, y)
        self.weights_ = weights
        self.n_original_features_ = X.shape[1]
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        X_expanded = self._apply_thresholds(X)
        X_reweighted = X_expanded / self.weights_
        return self.model_.predict_proba(X_reweighted)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        X_expanded = self._apply_thresholds(X)
        X_reweighted = X_expanded / self.weights_
        return self.model_.predict(X_reweighted)
