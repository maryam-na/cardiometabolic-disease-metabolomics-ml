from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def inverse_probability_weighting(
    df: pd.DataFrame,
    treatment: str,
    outcome: str = "t2d_status",
    covariates: list[str] | None = None,
) -> dict[str, float]:
    covariates = covariates or []
    data = df[[treatment, outcome] + covariates].dropna().copy()
    if data.empty:
        raise ValueError("No complete rows available for causal estimation.")

    threshold = data[treatment].median()
    data["treatment_binary"] = (data[treatment] >= threshold).astype(int)
    X = data[covariates] if covariates else np.ones((len(data), 1))
    propensity = LogisticRegression(max_iter=2000).fit(X, data["treatment_binary"]).predict_proba(X)[:, 1]
    propensity = np.clip(propensity, 0.01, 0.99)
    treated = data["treatment_binary"] == 1
    y = data[outcome].to_numpy()
    ate = np.mean(treated * y / propensity - (1 - treated) * y / (1 - propensity))
    weights = np.where(treated, 1 / propensity, 1 / (1 - propensity))
    se = np.std(weights * (y - ate), ddof=1) / np.sqrt(len(data))
    return {
        "treatment": treatment,
        "threshold": float(threshold),
        "ate": float(ate),
        "ci_low": float(ate - 1.96 * se),
        "ci_high": float(ate + 1.96 * se),
        "n": int(len(data)),
    }
