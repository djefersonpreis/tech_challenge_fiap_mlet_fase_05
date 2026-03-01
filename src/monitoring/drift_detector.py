"""Drift monitoring module using Evidently AI."""

import json
from pathlib import Path

import pandas as pd
from loguru import logger

from src.utils.constants import LOGS_DIR, REFERENCE_DATA_PATH


def load_reference_data() -> pd.DataFrame:
    """Load reference (training) data for drift comparison.

    Returns:
        DataFrame with training features and target.
    """
    if not REFERENCE_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Reference data not found at {REFERENCE_DATA_PATH}. Run training first."
        )
    return pd.read_parquet(REFERENCE_DATA_PATH)


def load_production_data(limit: int = 500) -> pd.DataFrame:
    """Load recent production predictions from log file.

    Args:
        limit: Maximum number of recent predictions to load.

    Returns:
        DataFrame with production input features.
    """
    predictions_log = LOGS_DIR / "predictions.jsonl"
    if not predictions_log.exists():
        logger.warning("No production predictions log found")
        return pd.DataFrame()

    records = []
    with open(predictions_log, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if "input" in record:
                    records.append(record["input"])
            except json.JSONDecodeError:
                continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records[-limit:])
    logger.info(f"Loaded {len(df)} production records")
    return df


def generate_drift_report(
    reference: pd.DataFrame | None = None,
    current: pd.DataFrame | None = None,
    output_path: str | None = None,
) -> dict:
    """Generate a data drift report comparing reference and production data.

    Args:
        reference: Reference dataset (training data). Loads from file if None.
        current: Current/production dataset. Loads from logs if None.
        output_path: Path to save the HTML report.

    Returns:
        Dict with drift detection results.
    """
    try:
        from evidently import ColumnMapping
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report
    except ImportError:
        logger.warning("Evidently not installed. Skipping drift report.")
        return {"error": "Evidently not installed"}

    if reference is None:
        reference = load_reference_data()
    if current is None:
        current = load_production_data()

    if current.empty:
        logger.warning("No production data available for drift detection")
        return {"error": "No production data"}

    # Align columns (exclude target from reference)
    feature_cols = [c for c in reference.columns if c != "target"]
    common_cols = [c for c in feature_cols if c in current.columns]

    if not common_cols:
        logger.warning("No common columns between reference and production data")
        return {"error": "No common columns"}

    ref_aligned = reference[common_cols]
    cur_aligned = current[common_cols]

    column_mapping = ColumnMapping(
        numerical_features=[c for c in common_cols if ref_aligned[c].dtype in ["float64", "int64"]],
        categorical_features=[c for c in common_cols if ref_aligned[c].dtype == "object"],
    )

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_aligned, current_data=cur_aligned, column_mapping=column_mapping)

    if output_path is None:
        output_path = str(LOGS_DIR / "drift_report.html")

    report.save_html(output_path)
    logger.info(f"Drift report saved to {output_path}")

    result = report.as_dict()
    drift_detected = _extract_drift_summary(result)

    return drift_detected


def _extract_drift_summary(report_dict: dict) -> dict:
    """Extract a summary of drift results from Evidently report dict.

    Args:
        report_dict: Raw Evidently report as dict.

    Returns:
        Simplified drift summary dict.
    """
    try:
        metrics = report_dict.get("metrics", [])
        for metric in metrics:
            result = metric.get("result", {})
            if "drift_share" in result:
                return {
                    "dataset_drift": result.get("dataset_drift", False),
                    "drift_share": result.get("drift_share", 0.0),
                    "number_of_columns": result.get("number_of_columns", 0),
                    "number_of_drifted_columns": result.get("number_of_drifted_columns", 0),
                    "drifted_columns": [
                        col_name
                        for col_name, col_data in result.get("drift_by_columns", {}).items()
                        if col_data.get("drift_detected", False)
                    ],
                }
    except Exception as e:
        logger.warning(f"Error parsing drift report: {e}")

    return {"dataset_drift": False, "drift_share": 0.0}
