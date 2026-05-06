from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.causal_inference import inverse_probability_weighting
from src.evaluation import compute_metrics, curve_data, save_metrics
from src.explainability import compute_shap_artifacts, extract_feature_names, save_feature_importance
from src.feature_selection import make_selector
from src.preprocessing import build_preprocessor, load_and_merge_data
from src.utils import MODELS_DIR, PROCESSED_DIR, RESULTS_DIR, ensure_dirs
from src.visualization import volcano_plot


DEFAULT_RANDOM_STATE = 42


def make_pipeline(model, selector_method: str = "anova", k: int = 50, seed: int = DEFAULT_RANDOM_STATE) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("select", make_selector(selector_method, k, random_state=seed)),
            ("model", model),
        ]
    )


def baseline_models(seed: int = DEFAULT_RANDOM_STATE, n_jobs: int = 1) -> dict[str, Pipeline]:
    return {
        "elastic_net_logistic": make_pipeline(
            LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                l1_ratio=0.5,
                max_iter=5000,
                class_weight="balanced",
                random_state=seed,
            ),
            "anova",
            50,
            seed,
        ),
        "random_forest": make_pipeline(
            RandomForestClassifier(
                n_estimators=500,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=n_jobs,
            ),
            "mutual_info",
            50,
            seed,
        ),
    }


def optimize_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 5,
    seed: int = DEFAULT_RANDOM_STATE,
    n_jobs: int = 1,
) -> dict:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 2.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "eval_metric": "logloss",
            "tree_method": "hist",
            "random_state": seed,
            "n_jobs": n_jobs,
        }
        k = trial.suggest_categorical("k", [10, 20, 50, 100])
        selector = trial.suggest_categorical("selector", ["anova", "mutual_info", "lasso"])
        model = XGBClassifier(**params)
        pipe = make_pipeline(model, selector, min(k, X_train.shape[1]), seed)
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1)
        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", study_name="st003390_xgboost", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def train(
    n_trials: int = 5,
    seed: int = DEFAULT_RANDOM_STATE,
    n_jobs: int = 1,
    skip_causal: bool = False,
    shap_samples: int = 60,
) -> dict[str, float]:
    ensure_dirs()
    loaded = load_and_merge_data()
    df = loaded.data
    X = df[loaded.feature_columns]
    y = df[loaded.label_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=seed
    )
    train_df = df.loc[X_train.index].copy()
    test_df = df.loc[X_test.index].copy()
    X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)

    model_metrics = {}
    for name, pipe in baseline_models(seed=seed, n_jobs=n_jobs).items():
        pipe.fit(X_train, y_train)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        model_metrics[name] = compute_metrics(y_test, y_prob)
        joblib.dump(pipe, MODELS_DIR / f"{name}.joblib")

    best = optimize_xgboost(X_train, y_train, n_trials=n_trials, seed=seed, n_jobs=n_jobs)
    k = min(int(best.pop("k")), X_train.shape[1])
    selector = best.pop("selector")
    best.update({"eval_metric": "logloss", "tree_method": "hist", "random_state": seed, "n_jobs": n_jobs})
    xgb_pipe = make_pipeline(XGBClassifier(**best), selector, k, seed)
    xgb_pipe.fit(X_train, y_train)

    calibrated = CalibratedClassifierCV(xgb_pipe, method="isotonic", cv=5)
    calibrated.fit(X_train, y_train)
    y_prob = calibrated.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_prob)
    model_metrics["xgboost_calibrated"] = metrics
    save_metrics(metrics, RESULTS_DIR / "test_metrics.json")
    (RESULTS_DIR / "all_model_metrics.json").write_text(json.dumps(model_metrics, indent=2), encoding="utf-8")

    curves = curve_data(y_test, y_prob)
    for name, table in curves.items():
        table.to_csv(RESULTS_DIR / f"{name}.csv", index=False)

    joblib.dump(calibrated, MODELS_DIR / "xgboost_calibrated.joblib")
    joblib.dump(xgb_pipe, MODELS_DIR / "xgboost_uncalibrated_pipeline.joblib")
    selected_names = extract_feature_names(xgb_pipe, loaded.feature_columns)
    pd.DataFrame({"biomarker": selected_names}).to_csv(RESULTS_DIR / "selected_biomarkers.csv", index=False)
    save_feature_importance(xgb_pipe, selected_names)
    try:
        compute_shap_artifacts(
            xgb_pipe,
            X_test,
            selected_names,
            max_samples=shap_samples,
            random_state=seed,
        )
    except Exception as exc:
        (RESULTS_DIR / "shap_error.txt").write_text(str(exc), encoding="utf-8")

    train_biomarker_table, _ = volcano_plot(train_df, loaded.feature_columns, loaded.label_column)
    test_biomarker_table, _ = volcano_plot(test_df, loaded.feature_columns, loaded.label_column)
    train_biomarker_table.to_csv(RESULTS_DIR / "train_ranked_biomarkers.csv", index=False)
    test_biomarker_table.to_csv(RESULTS_DIR / "test_validation_biomarkers.csv", index=False)
    train_biomarker_table.to_csv(RESULTS_DIR / "ranked_biomarkers.csv", index=False)

    if not skip_causal and not train_biomarker_table.empty:
        top_biomarker = str(train_biomarker_table.iloc[0]["feature"])
        try:
            causal_estimate = inverse_probability_weighting(train_df, treatment=top_biomarker)
            (RESULTS_DIR / "causal_association.json").write_text(
                json.dumps(causal_estimate, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            (RESULTS_DIR / "causal_error.txt").write_text(str(exc), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the ST003390 cardiometabolic disease model pipeline.")
    parser.add_argument("--trials", type=int, default=5, help="Number of Optuna trials for XGBoost tuning.")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_STATE, help="Random seed for splits, tuning, and selectors.")
    parser.add_argument("--n-jobs", type=int, default=1, help="Worker count for estimators that support parallelism.")
    parser.add_argument("--skip-causal", action="store_true", help="Skip exploratory causal/association estimate export.")
    parser.add_argument("--shap-samples", type=int, default=60, help="Maximum held-out samples used for SHAP artifact generation.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            train(
                n_trials=args.trials,
                seed=args.seed,
                n_jobs=args.n_jobs,
                skip_causal=args.skip_causal,
                shap_samples=args.shap_samples,
            ),
            indent=2,
        )
    )
