"""CLI script to run the full training pipeline."""

import json

import numpy as np
from loguru import logger
from sklearn.model_selection import train_test_split

from src.evaluation.evaluate import evaluate_model
from src.preprocessing.pipeline import build_pipeline
from src.training.train import cross_validate_models, select_best_model, train_final_model
from src.utils.constants import MODELS_DIR, RANDOM_STATE, TEST_SIZE


def main():
    """Run the complete training pipeline."""
    logger.info("=" * 60)
    logger.info("Starting full training pipeline")
    logger.info("=" * 60)

    # Step 1: Preprocess
    X, y = build_pipeline()
    logger.info(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")

    # Step 2: Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Step 3: Cross-validate & select best
    cv_results = cross_validate_models(X_train, y_train)
    best_model_name = select_best_model(cv_results, metric="f1")

    # Step 4: Train final model
    model, scaler, model_name = train_final_model(
        X_train, y_train, model_name=best_model_name
    )

    # Step 5: Evaluate on test set
    metrics = evaluate_model(model, X_test, y_test, scaler=scaler)

    # Step 6: Save training metrics for dashboard
    _save_training_metrics(cv_results, metrics, model_name, X, y, X_train, X_test)

    logger.info("=" * 60)
    logger.info(f"Final model: {model_name}")
    logger.info(f"Test F1: {metrics['f1']:.4f}")
    logger.info(f"Test Recall: {metrics['recall']:.4f}")
    logger.info(f"Test AUC-ROC: {metrics.get('roc_auc', 'N/A')}")
    logger.info("=" * 60)

    return model, scaler, metrics


def _save_training_metrics(
    cv_results: dict,
    test_metrics: dict,
    model_name: str,
    X: "pd.DataFrame",
    y: "pd.Series",
    X_train: "pd.DataFrame",
    X_test: "pd.DataFrame",
) -> None:
    """Persist training metrics to JSON so the dashboard can display them."""
    training_metrics = {
        "best_model": model_name,
        "cv_results": {
            name: {
                metric: (
                    values.tolist()
                    if isinstance(values, np.ndarray)
                    else values
                )
                for metric, values in model_metrics.items()
            }
            for name, model_metrics in cv_results.items()
        },
        "test_metrics": {
            k: v
            for k, v in test_metrics.items()
            if k not in ("confusion_matrix", "classification_report")
        },
        "confusion_matrix": test_metrics.get("confusion_matrix"),
        "classification_report": test_metrics.get("classification_report"),
        "dataset": {
            "total_samples": int(X.shape[0]),
            "features": int(X.shape[1]),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "target_distribution": {
                str(k): int(v) for k, v in y.value_counts().to_dict().items()
            },
        },
    }
    metrics_path = MODELS_DIR / "training_metrics.json"
    MODELS_DIR.mkdir(exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(training_metrics, f, indent=2, default=str)
    logger.info(f"Training metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
