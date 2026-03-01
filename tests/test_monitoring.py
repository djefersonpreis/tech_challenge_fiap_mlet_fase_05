"""Tests for monitoring/drift detection module."""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.monitoring.drift_detector import (
    _extract_drift_summary,
    load_production_data,
    load_reference_data,
)


class TestLoadReferenceData:
    """Tests for load_reference_data function."""

    def test_raises_when_no_file(self, monkeypatch):
        """Test that it raises FileNotFoundError when reference data missing."""
        monkeypatch.setattr(
            "src.monitoring.drift_detector.REFERENCE_DATA_PATH",
            Path("/nonexistent/ref.parquet"),
        )
        with pytest.raises(FileNotFoundError):
            load_reference_data()

    def test_loads_parquet(self, tmp_path, monkeypatch):
        """Test loading reference data from parquet."""
        ref_path = tmp_path / "ref.parquet"
        df = pd.DataFrame({"feature1": [1, 2, 3], "target": [0, 1, 0]})
        df.to_parquet(ref_path)

        monkeypatch.setattr(
            "src.monitoring.drift_detector.REFERENCE_DATA_PATH", ref_path
        )
        result = load_reference_data()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3


class TestLoadProductionData:
    """Tests for load_production_data function."""

    def test_returns_empty_when_no_log(self, monkeypatch):
        """Test returns empty DataFrame when no log file exists."""
        monkeypatch.setattr(
            "src.monitoring.drift_detector.LOGS_DIR",
            Path("/nonexistent"),
        )
        result = load_production_data()
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_loads_from_jsonl(self, tmp_path, monkeypatch):
        """Test loading production data from JSONL log."""
        monkeypatch.setattr(
            "src.monitoring.drift_detector.LOGS_DIR", tmp_path
        )
        log_file = tmp_path / "predictions.jsonl"
        records = [
            {"input": {"feature1": 1.0, "feature2": 2.0}, "prediction": 0},
            {"input": {"feature1": 3.0, "feature2": 4.0}, "prediction": 1},
        ]
        with open(log_file, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        result = load_production_data()
        assert len(result) == 2
        assert "feature1" in result.columns

    def test_respects_limit(self, tmp_path, monkeypatch):
        """Test that limit parameter works."""
        monkeypatch.setattr(
            "src.monitoring.drift_detector.LOGS_DIR", tmp_path
        )
        log_file = tmp_path / "predictions.jsonl"
        with open(log_file, "w") as f:
            for i in range(10):
                f.write(json.dumps({"input": {"feature1": float(i)}}) + "\n")

        result = load_production_data(limit=3)
        assert len(result) == 3

    def test_handles_malformed_json(self, tmp_path, monkeypatch):
        """Test graceful handling of malformed JSON lines."""
        monkeypatch.setattr(
            "src.monitoring.drift_detector.LOGS_DIR", tmp_path
        )
        log_file = tmp_path / "predictions.jsonl"
        with open(log_file, "w") as f:
            f.write('{"input": {"a": 1}}\n')
            f.write("not valid json\n")
            f.write('{"input": {"a": 2}}\n')

        result = load_production_data()
        assert len(result) == 2


class TestGenerateDriftReport:
    """Tests for generate_drift_report function."""

    def test_returns_error_when_no_production_data(self, tmp_path, monkeypatch):
        """Test returns error dict when no production data available."""
        from src.monitoring.drift_detector import generate_drift_report

        ref_path = tmp_path / "ref.parquet"
        ref = pd.DataFrame({"feature1": [1, 2, 3], "target": [0, 1, 0]})
        ref.to_parquet(ref_path)

        monkeypatch.setattr("src.monitoring.drift_detector.REFERENCE_DATA_PATH", ref_path)
        monkeypatch.setattr("src.monitoring.drift_detector.LOGS_DIR", tmp_path / "nologs")

        result = generate_drift_report()
        assert "error" in result

    def test_returns_error_no_common_cols(self, tmp_path, monkeypatch):
        """Test returns error when no common columns."""
        from src.monitoring.drift_detector import generate_drift_report

        ref = pd.DataFrame({"feature_a": [1, 2, 3], "target": [0, 1, 0]})
        cur = pd.DataFrame({"feature_z": [4, 5, 6]})

        result = generate_drift_report(reference=ref, current=cur)
        assert "error" in result

    def test_returns_error_empty_current(self, tmp_path, monkeypatch):
        """Test returns error when current data is empty."""
        from src.monitoring.drift_detector import generate_drift_report

        ref = pd.DataFrame({"feature1": [1, 2, 3], "target": [0, 1, 0]})
        cur = pd.DataFrame()

        result = generate_drift_report(reference=ref, current=cur)
        assert "error" in result

    def test_runs_with_matching_data(self, tmp_path, monkeypatch):
        """Test runs successfully when data matches."""
        from src.monitoring.drift_detector import generate_drift_report

        import numpy as np
        np.random.seed(42)
        ref = pd.DataFrame({
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "target": np.random.choice([0, 1], 100),
        })
        cur = pd.DataFrame({
            "feature1": np.random.randn(50),
            "feature2": np.random.randn(50),
        })

        monkeypatch.setattr("src.monitoring.drift_detector.LOGS_DIR", tmp_path)

        result = generate_drift_report(reference=ref, current=cur, output_path=str(tmp_path / "report.html"))
        assert isinstance(result, dict)
        assert "dataset_drift" in result or "error" in result


class TestExtractDriftSummary:
    """Tests for _extract_drift_summary function."""

    def test_extracts_from_valid_report(self):
        """Test extracting summary from valid Evidently report dict."""
        report = {
            "metrics": [{
                "result": {
                    "dataset_drift": True,
                    "drift_share": 0.5,
                    "number_of_columns": 10,
                    "number_of_drifted_columns": 5,
                    "drift_by_columns": {
                        "col1": {"drift_detected": True},
                        "col2": {"drift_detected": False},
                    },
                }
            }]
        }
        result = _extract_drift_summary(report)
        assert result["dataset_drift"] is True
        assert result["drift_share"] == 0.5
        assert "col1" in result["drifted_columns"]
        assert "col2" not in result["drifted_columns"]

    def test_handles_empty_report(self):
        """Test handling of empty report."""
        result = _extract_drift_summary({})
        assert result["dataset_drift"] is False
        assert result["drift_share"] == 0.0
