"""CLI script to run the full training pipeline."""

from loguru import logger
from sklearn.model_selection import train_test_split

from src.evaluation.evaluate import evaluate_model
from src.preprocessing.pipeline import build_pipeline
from src.training.train import cross_validate_models, select_best_model, train_final_model
from src.utils.constants import RANDOM_STATE, TEST_SIZE


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

    logger.info("=" * 60)
    logger.info(f"Final model: {model_name}")
    logger.info(f"Test F1: {metrics['f1']:.4f}")
    logger.info(f"Test Recall: {metrics['recall']:.4f}")
    logger.info(f"Test AUC-ROC: {metrics.get('roc_auc', 'N/A')}")
    logger.info("=" * 60)

    return model, scaler, metrics


if __name__ == "__main__":
    main()
