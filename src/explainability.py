from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from src.utils import FIGURES_DIR, RESULTS_DIR


def extract_feature_names(pipeline, original_features: list[str]) -> list[str]:
    names = list(original_features)
    for step_name, step in pipeline.named_steps.items():
        if step_name == "model":
            break
        if hasattr(step, "get_support"):
            support = step.get_support()
            names = list(np.array(names)[support])
        elif hasattr(step, "get_feature_names_out"):
            try:
                names = list(step.get_feature_names_out(names))
            except Exception:
                pass
    return names


def save_feature_importance(pipeline, feature_names: list[str], out_path: Path = RESULTS_DIR / "feature_importance.csv") -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_).ravel()
    else:
        importance = np.zeros(len(feature_names))
    table = pd.DataFrame({"feature": feature_names, "importance": importance}).sort_values("importance", ascending=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)
    return table


def compute_shap_artifacts(
    pipeline,
    X: pd.DataFrame,
    feature_names: list[str],
    max_samples: int = 60,
    random_state: int = 42,
) -> pd.DataFrame:
    X_sample = X.sample(n=min(max_samples, len(X)), random_state=random_state) if len(X) > max_samples else X
    transformed = pipeline[:-1].transform(X_sample)
    model = pipeline.named_steps["model"]
    explainer = shap.Explainer(model, transformed, feature_names=feature_names)
    values = explainer(transformed)
    shap_values = values.values
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, -1]
    mean_abs = np.abs(shap_values).mean(axis=0)
    shap_table = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    shap_table.to_csv(RESULTS_DIR / "shap_feature_ranking.csv", index=False)
    joblib.dump(explainer, MODELS_DIR / "shap_explainer.joblib")
    return shap_table


from src.utils import MODELS_DIR  # noqa: E402
