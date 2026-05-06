from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression

from src.preprocessing import CorrelationFilter


class L1LogisticSelector(BaseEstimator, TransformerMixin):
    def __init__(self, C: float = 0.1, max_features: int | None = None, random_state: int = 42):
        self.C = C
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y):
        self.model_ = LogisticRegression(
            penalty="l1",
            solver="saga",
            C=self.C,
            max_iter=5000,
            class_weight="balanced",
            random_state=self.random_state,
        )
        self.model_.fit(X, y)
        weights = np.abs(self.model_.coef_).ravel()
        order = np.argsort(weights)[::-1]
        if self.max_features:
            order = order[: self.max_features]
        self.support_ = np.zeros_like(weights, dtype=bool)
        self.support_[order[weights[order] > 0]] = True
        if not self.support_.any():
            self.support_[order[: min(self.max_features or 10, len(order))]] = True
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.support_]

    def get_support(self):
        return self.support_


def make_selector(method: str = "anova", k: int = 50, random_state: int = 42):
    if method == "anova":
        return SelectKBest(score_func=f_classif, k=k)
    if method == "mutual_info":
        score_func = partial(mutual_info_classif, random_state=random_state)
        return SelectKBest(score_func=score_func, k=k)
    if method == "lasso":
        return L1LogisticSelector(max_features=k, random_state=random_state)
    raise ValueError(f"Unknown selector method: {method}")


def filter_features(X: pd.DataFrame, corr_threshold: float = 0.90) -> pd.DataFrame:
    variance = VarianceThreshold(threshold=1e-8)
    X_var = pd.DataFrame(variance.fit_transform(X), columns=X.columns[variance.get_support()], index=X.index)
    corr = CorrelationFilter(threshold=corr_threshold)
    return corr.fit_transform(X_var)
