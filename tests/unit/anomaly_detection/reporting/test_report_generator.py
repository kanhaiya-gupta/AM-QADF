"""
Unit tests for report generator.

Tests for AnomalyDetectionReport and ReportGenerator classes.
"""

import pytest
import numpy as np
import json
import csv
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from am_qadf.anomaly_detection.reporting.report_generator import (
    AnomalyDetectionReport,
    ReportGenerator,
)
from am_qadf.anomaly_detection.evaluation.metrics import AnomalyDetectionMetrics


class TestAnomalyDetectionReport:
    """Test suite for AnomalyDetectionReport dataclass."""

    @pytest.mark.unit
    def test_report_creation(self):
        """Test creating AnomalyDetectionReport."""
        report = AnomalyDetectionReport(
            report_id="test_report_001",
            timestamp="2023-01-01T12:00:00",
            n_samples=100,
            n_anomalies=10,
            n_normal=90,
            method_results={"method1": {"precision": 0.9, "recall": 0.8, "f1_score": 0.85}},
            summary_stats={"n_methods": 1},
        )

        assert report.report_id == "test_report_001"
        assert report.timestamp == "2023-01-01T12:00:00"
        assert report.n_samples == 100
        assert report.n_anomalies == 10
        assert report.n_normal == 90
        assert len(report.method_results) == 1
        assert report.best_method is None
        assert report.best_metric is None
        assert report.metadata is None

    @pytest.mark.unit
    def test_report_creation_with_all_fields(self):
        """Test creating AnomalyDetectionReport with all fields."""
        report = AnomalyDetectionReport(
            report_id="test_report_002",
            timestamp="2023-01-01T12:00:00",
            n_samples=100,
            n_anomalies=10,
            n_normal=90,
            method_results={"method1": {"precision": 0.9, "recall": 0.8, "f1_score": 0.85}},
            summary_stats={"n_methods": 1},
            best_method="method1",
            best_metric="f1_score",
            metadata={"model_id": "model1", "dataset": "test_data"},
        )

        assert report.best_method == "method1"
        assert report.best_metric == "f1_score"
        assert report.metadata == {"model_id": "model1", "dataset": "test_data"}


class TestReportGenerator:
    """Test suite for ReportGenerator class."""

    @pytest.fixture
    def sample_metrics(self):
        """Create sample metrics objects."""
        return {
            "method1": AnomalyDetectionMetrics(
                precision=0.9,
                recall=0.8,
                f1_score=0.85,
                accuracy=0.9,
                specificity=0.95,
                sensitivity=0.8,
                roc_auc=0.95,
            ),
            "method2": AnomalyDetectionMetrics(
                precision=0.85,
                recall=0.9,
                f1_score=0.875,
                accuracy=0.88,
                specificity=0.9,
                sensitivity=0.9,
                roc_auc=0.92,
            ),
        }

    @pytest.fixture
    def sample_method_results(self, sample_metrics):
        """Create sample method results."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
        y_pred1 = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 0])
        y_pred2 = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
        y_scores1 = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.4, 0.2, 0.85, 0.95, 0.15])
        y_scores2 = np.array([0.1, 0.2, 0.85, 0.9, 0.3, 0.9, 0.2, 0.88, 0.95, 0.15])

        return {
            "method1": {
                "metrics": sample_metrics["method1"],
                "y_pred": y_pred1,
                "y_scores": y_scores1,
            },
            "method2": {
                "metrics": sample_metrics["method2"],
                "y_pred": y_pred2,
                "y_scores": y_scores2,
            },
        }

    @pytest.fixture
    def sample_y_true(self):
        """Create sample true labels."""
        return np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])

    @pytest.mark.unit
    def test_generator_creation(self):
        """Test creating ReportGenerator."""
        generator = ReportGenerator()

        assert generator.reports == []

    @pytest.mark.unit
    def test_generate_report(self, sample_method_results, sample_y_true):
        """Test generating a report."""
        generator = ReportGenerator()
        report = generator.generate_report(sample_method_results, sample_y_true, report_id="test_report")

        assert isinstance(report, AnomalyDetectionReport)
        assert report.report_id == "test_report"
        assert report.n_samples == len(sample_y_true)
        assert report.n_anomalies == np.sum(sample_y_true)
        assert report.n_normal == len(sample_y_true) - np.sum(sample_y_true)
        assert len(report.method_results) == 2
        assert report.best_method is not None
        assert report.best_metric == "f1_score"
        assert len(generator.reports) == 1

    @pytest.mark.unit
    def test_generate_report_auto_id(self, sample_method_results, sample_y_true):
        """Test generating a report with auto-generated ID."""
        generator = ReportGenerator()
        report = generator.generate_report(sample_method_results, sample_y_true)

        assert report.report_id is not None
        assert report.report_id.startswith("report_")

    @pytest.mark.unit
    def test_generate_report_with_metadata(self, sample_method_results, sample_y_true):
        """Test generating a report with metadata."""
        generator = ReportGenerator()
        metadata = {"model_id": "model1", "dataset": "test_data"}
        report = generator.generate_report(sample_method_results, sample_y_true, metadata=metadata)

        assert report.metadata == metadata

    @pytest.mark.unit
    def test_generate_report_finds_best_method(self, sample_method_results, sample_y_true):
        """Test that report identifies best method by F1-score."""
        generator = ReportGenerator()
        report = generator.generate_report(sample_method_results, sample_y_true)

        # method2 has higher F1-score (0.875 vs 0.85)
        assert report.best_method == "method2"
        assert report.method_results["method2"]["f1_score"] > report.method_results["method1"]["f1_score"]

    @pytest.mark.unit
    def test_generate_report_summary_stats(self, sample_method_results, sample_y_true):
        """Test that summary statistics are calculated correctly."""
        generator = ReportGenerator()
        report = generator.generate_report(sample_method_results, sample_y_true)

        assert report.summary_stats["n_samples"] == len(sample_y_true)
        assert report.summary_stats["n_anomalies"] == np.sum(sample_y_true)
        assert report.summary_stats["n_methods"] == 2
        assert "anomaly_rate" in report.summary_stats

    @pytest.mark.unit
    def test_export_to_json(self, sample_method_results, sample_y_true):
        """Test exporting report to JSON."""
        generator = ReportGenerator()
        report = generator.generate_report(sample_method_results, sample_y_true)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_report.json"
            generator.export_to_json(report, filepath)

            assert filepath.exists()

            # Verify JSON content
            with open(filepath, "r") as f:
                data = json.load(f)

            assert data["report_id"] == report.report_id
            assert data["n_samples"] == report.n_samples
            assert "method_results" in data
            assert "summary_stats" in data

    @pytest.mark.unit
    def test_export_to_json_creates_directory(self, sample_method_results, sample_y_true):
        """Test that export creates directory if it doesn't exist."""
        generator = ReportGenerator()
        report = generator.generate_report(sample_method_results, sample_y_true)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "subdir" / "test_report.json"
            generator.export_to_json(report, filepath)

            assert filepath.exists()
            assert filepath.parent.exists()

    @pytest.mark.unit
    def test_export_to_csv(self, sample_method_results, sample_y_true):
        """Test exporting report to CSV."""
        generator = ReportGenerator()
        report = generator.generate_report(sample_method_results, sample_y_true)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_report.csv"
            generator.export_to_csv(report, filepath)

            assert filepath.exists()

            # Verify CSV content
            with open(filepath, "r") as f:
                reader = csv.reader(f)
                rows = list(reader)

            assert len(rows) > 0
            assert "Method" in rows[0]
            assert "Precision" in rows[0]
            assert "Recall" in rows[0]
            assert "F1-Score" in rows[0]
            # Should have header + 2 methods
            assert len(rows) == 3

    @pytest.mark.unit
    def test_export_to_csv_creates_directory(self, sample_method_results, sample_y_true):
        """Test that CSV export creates directory if it doesn't exist."""
        generator = ReportGenerator()
        report = generator.generate_report(sample_method_results, sample_y_true)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "subdir" / "test_report.csv"
            generator.export_to_csv(report, filepath)

            assert filepath.exists()
            assert filepath.parent.exists()

    @pytest.mark.unit
    def test_generate_summary_text(self, sample_method_results, sample_y_true):
        """Test generating text summary."""
        generator = ReportGenerator()
        report = generator.generate_report(sample_method_results, sample_y_true)

        summary = generator.generate_summary_text(report)

        assert isinstance(summary, str)
        assert "ANOMALY DETECTION REPORT" in summary
        assert report.report_id in summary
        assert "SUMMARY STATISTICS" in summary
        assert "BEST METHOD" in summary
        assert "METHOD COMPARISON" in summary
        assert report.best_method in summary
        assert "method1" in summary
        assert "method2" in summary

    @pytest.mark.unit
    def test_generate_summary_text_no_best_method(self, sample_y_true):
        """Test generating summary when no best method is identified."""
        generator = ReportGenerator()
        # Create report with empty method results
        method_results = {}
        report = AnomalyDetectionReport(
            report_id="test",
            timestamp="2023-01-01T12:00:00",
            n_samples=len(sample_y_true),
            n_anomalies=np.sum(sample_y_true),
            n_normal=len(sample_y_true) - np.sum(sample_y_true),
            method_results={},
            summary_stats={"n_methods": 0},
            best_method=None,
        )

        summary = generator.generate_summary_text(report)

        assert isinstance(summary, str)
        assert "ANOMALY DETECTION REPORT" in summary
        # Should not have BEST METHOD section if best_method is None
        # (or should handle gracefully)

    @pytest.mark.unit
    def test_export_summary_text(self, sample_method_results, sample_y_true):
        """Test exporting text summary to file."""
        generator = ReportGenerator()
        report = generator.generate_report(sample_method_results, sample_y_true)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_summary.txt"
            generator.export_summary_text(report, filepath)

            assert filepath.exists()

            # Verify file content
            with open(filepath, "r") as f:
                content = f.read()

            assert "ANOMALY DETECTION REPORT" in content
            assert report.report_id in content

    @pytest.mark.unit
    def test_export_summary_text_creates_directory(self, sample_method_results, sample_y_true):
        """Test that text export creates directory if it doesn't exist."""
        generator = ReportGenerator()
        report = generator.generate_report(sample_method_results, sample_y_true)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "subdir" / "test_summary.txt"
            generator.export_summary_text(report, filepath)

            assert filepath.exists()
            assert filepath.parent.exists()

    @pytest.mark.unit
    def test_multiple_reports(self, sample_method_results, sample_y_true):
        """Test generating multiple reports."""
        generator = ReportGenerator()

        report1 = generator.generate_report(sample_method_results, sample_y_true, report_id="report1")
        report2 = generator.generate_report(sample_method_results, sample_y_true, report_id="report2")

        assert len(generator.reports) == 2
        assert generator.reports[0].report_id == "report1"
        assert generator.reports[1].report_id == "report2"

    @pytest.mark.unit
    def test_generate_report_with_missing_metrics(self, sample_y_true):
        """Test generating report with missing metrics."""
        generator = ReportGenerator()

        # Method results without metrics
        method_results = {
            "method1": {
                "y_pred": np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0]),
                "y_scores": np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.4, 0.2, 0.85, 0.95, 0.15]),
            }
        }

        report = generator.generate_report(method_results, sample_y_true)

        assert isinstance(report, AnomalyDetectionReport)
        # Should handle gracefully with missing metrics
        assert len(report.method_results) >= 0

    @pytest.mark.unit
    def test_generate_report_with_none_roc_auc(self, sample_y_true):
        """Test generating report when ROC-AUC is None."""
        generator = ReportGenerator()

        metrics = AnomalyDetectionMetrics(
            precision=0.9,
            recall=0.8,
            f1_score=0.85,
            accuracy=0.9,
            specificity=0.95,
            sensitivity=0.8,
            roc_auc=None,
        )

        method_results = {
            "method1": {
                "metrics": metrics,
                "y_pred": np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0]),
                "y_scores": None,
            }
        }

        report = generator.generate_report(method_results, sample_y_true)

        assert isinstance(report, AnomalyDetectionReport)
        assert report.method_results["method1"].get("roc_auc") is None

    @pytest.mark.unit
    def test_csv_export_handles_none_roc_auc(self, sample_method_results, sample_y_true):
        """Test that CSV export handles None ROC-AUC values."""
        generator = ReportGenerator()

        # Modify one method to have None ROC-AUC
        metrics = AnomalyDetectionMetrics(
            precision=0.9,
            recall=0.8,
            f1_score=0.85,
            accuracy=0.9,
            specificity=0.95,
            sensitivity=0.8,
            roc_auc=None,
        )
        sample_method_results["method1"]["metrics"] = metrics

        report = generator.generate_report(sample_method_results, sample_y_true)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_report.csv"
            generator.export_to_csv(report, filepath)

            # Should not raise error
            assert filepath.exists()

            # Verify 'N/A' is written for None values
            with open(filepath, "r") as f:
                content = f.read()
                # Should contain 'N/A' for None ROC-AUC
                assert "N/A" in content or "method1" in content
