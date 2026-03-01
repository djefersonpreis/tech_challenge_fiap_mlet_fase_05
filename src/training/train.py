"""Model training module."""

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler

from src.utils.constants import (
    MODELS_DIR,
    MODEL_PATH,
    PIPELINE_PATH,
    RANDOM_STATE,
    REFERENCE_DATA_PATH,
)


def get_candidate_models() -> dict:
    """Return dictionary of candidate models for evaluation.

    Returns:
        Dict mapping model name to sklearn estimator.
    """
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
        ),
    }


def cross_validate_models(
    X: pd.DataFrame,
    y: pd.Series,
    models: dict | None = None,
    cv_folds: int = 5,
) -> dict:
    """Cross-validate multiple models and return metrics.

    Args:
        X: Feature matrix.
        y: Target vector.
        models: Dict of model name -> estimator. Uses defaults if None.
        cv_folds: Number of cross-validation folds.

    Returns:
        Dict mapping model name to dict of metric arrays.
    """
    if models is None:
        models = get_candidate_models()

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["accuracy", "f1", "precision", "recall", "roc_auc"]

    results = {}
    for name, model in models.items():
        logger.info(f"Cross-validating {name}...")
        cv_results = cross_validate(
            model, X_scaled, y, cv=cv, scoring=scoring, return_train_score=False
        )
        results[name] = {
            metric: cv_results[f"test_{metric}"] for metric in scoring
        }
        logger.info(
            f"  {name}: F1={np.mean(results[name]['f1']):.4f} "
            f"(±{np.std(results[name]['f1']):.4f}), "
            f"Recall={np.mean(results[name]['recall']):.4f}, "
            f"AUC={np.mean(results[name]['roc_auc']):.4f}"
        )

    return results


def select_best_model(cv_results: dict, metric: str = "f1") -> str:
    """Select the best model based on mean CV score for given metric.

    Args:
        cv_results: Output from cross_validate_models.
        metric: Metric to optimize for.

    Returns:
        Name of the best model.
    """
    best_name = max(cv_results, key=lambda k: np.mean(cv_results[k][metric]))
    best_score = np.mean(cv_results[best_name][metric])
    logger.info(f"Best model: {best_name} with mean {metric}={best_score:.4f}")
    return best_name


def train_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str | None = None,
) -> tuple:
    """Train the final model on full training data and save artifacts.

    Args:
        X: Feature matrix.
        y: Target vector.
        model_name: Name of model to train. If None, auto-selects best.

    Returns:
        Tuple of (trained model, scaler, model_name).
    """
    models = get_candidate_models()

    if model_name is None:
        cv_results = cross_validate_models(X, y, models)
        model_name = select_best_model(cv_results)

    model = models[model_name]
    scaler = StandardScaler()

    logger.info(f"Training final model: {model_name}")
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    model.fit(X_scaled, y)

    # Save artifacts
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, PIPELINE_PATH)
    logger.info(f"Model saved to {MODEL_PATH}")
    logger.info(f"Scaler saved to {PIPELINE_PATH}")

    # Save reference data for drift monitoring
    ref_df = X.copy()
    ref_df["target"] = y.values
    ref_df.to_parquet(REFERENCE_DATA_PATH, index=False)
    logger.info(f"Reference data saved to {REFERENCE_DATA_PATH}")

    return model, scaler, model_name
