"""Model evaluation module."""

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    scaler=None,
) -> dict:
    """Evaluate a trained model on the test set.

    Args:
        model: Trained sklearn model.
        X_test: Test features.
        y_test: Test target.
        scaler: Fitted StandardScaler. If provided, scales X_test.

    Returns:
        Dict with evaluation metrics.
    """
    logger.info("Evaluating model on test set")

    X_eval = X_test.copy()
    if scaler is not None:
        X_eval = pd.DataFrame(
            scaler.transform(X_eval), columns=X_eval.columns, index=X_eval.index
        )

    y_pred = model.predict(X_eval)
    y_proba = (
        model.predict_proba(X_eval)[:, 1]
        if hasattr(model, "predict_proba")
        else None
    )

    metrics = compute_metrics(y_test, y_pred, y_proba)

    logger.info(
        f"Evaluation: Accuracy={metrics['accuracy']:.4f}, "
        f"F1={metrics['f1']:.4f}, Recall={metrics['recall']:.4f}, "
        f"AUC-ROC={metrics.get('roc_auc', 'N/A')}"
    )
    logger.info(f"\n{metrics['classification_report']}")

    return metrics


def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict:
    """Compute classification metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities for positive class.

    Returns:
        Dict with all computed metrics.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred),
    }

    if y_proba is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))

    return metrics
