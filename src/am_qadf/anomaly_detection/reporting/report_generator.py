"""
Report Generator for Anomaly Detection

Generates comprehensive reports for anomaly detection results.
"""

import numpy as np
import json
import csv
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class AnomalyDetectionReport:
    """Comprehensive anomaly detection report."""

    # Report metadata
    report_id: str
    timestamp: str
    n_samples: int
    n_anomalies: int
    n_normal: int

    # Method results
    method_results: Dict[str, Dict[str, Any]]

    # Summary statistics
    summary_stats: Dict[str, Any]

    # Best method
    best_method: Optional[str] = None
    best_metric: Optional[str] = None

    # Additional info
    metadata: Dict[str, Any] = None


class ReportGenerator:
    """
    Generator for anomaly detection reports.
    """

    def __init__(self):
        """Initialize report generator."""
        self.reports: List[AnomalyDetectionReport] = []

    def generate_report(
        self,
        method_results: Dict[str, Any],
        y_true: np.ndarray,
        report_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AnomalyDetectionReport:
        """
        Generate a comprehensive anomaly detection report.

        Args:
            method_results: Dictionary mapping method names to result dictionaries
                Each result dict should contain:
                - 'metrics': Metrics object
                - 'y_pred': Predicted labels
                - 'y_scores': Optional anomaly scores
            y_true: True labels
            report_id: Optional report ID (auto-generated if None)
            metadata: Optional metadata dictionary

        Returns:
            AnomalyDetectionReport object
        """
        if report_id is None:
            report_id = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Extract method metrics
        method_metrics = {}
        for method_name, results in method_results.items():
            metrics = results.get("metrics")
            if metrics:
                method_metrics[method_name] = {
                    "precision": getattr(metrics, "precision", 0.0),
                    "recall": getattr(metrics, "recall", 0.0),
                    "f1_score": getattr(metrics, "f1_score", 0.0),
                    "accuracy": getattr(metrics, "accuracy", 0.0),
                    "roc_auc": getattr(metrics, "roc_auc", None),
                    "n_detected": (int(np.sum(results.get("y_pred", []))) if results.get("y_pred") is not None else 0),
                }

        # Calculate summary statistics
        summary_stats = {
            "n_samples": int(len(y_true)),
            "n_anomalies": int(np.sum(y_true)),
            "n_normal": int(len(y_true) - np.sum(y_true)),
            "anomaly_rate": float(np.mean(y_true)),
            "n_methods": len(method_results),
        }

        # Find best method by F1-score
        best_method = None
        best_f1 = -1.0
        for method_name, metrics_dict in method_metrics.items():
            f1 = metrics_dict.get("f1_score", 0.0)
            if f1 > best_f1:
                best_f1 = f1
                best_method = method_name

        # Create report
        report = AnomalyDetectionReport(
            report_id=report_id,
            timestamp=datetime.now().isoformat(),
            n_samples=summary_stats["n_samples"],
            n_anomalies=summary_stats["n_anomalies"],
            n_normal=summary_stats["n_normal"],
            method_results=method_metrics,
            summary_stats=summary_stats,
            best_method=best_method,
            best_metric="f1_score",
            metadata=metadata or {},
        )

        self.reports.append(report)
        return report

    def export_to_json(self, report: AnomalyDetectionReport, filepath: Union[str, Path]) -> None:
        """
        Export report to JSON file.

        Args:
            report: Report to export
            filepath: Path to output file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Convert dataclass to dict
        report_dict = asdict(report)

        with open(filepath, "w") as f:
            json.dump(report_dict, f, indent=2)

        logger.info(f"Report exported to JSON: {filepath}")

    def export_to_csv(self, report: AnomalyDetectionReport, filepath: Union[str, Path]) -> None:
        """
        Export method comparison to CSV file.

        Args:
            report: Report to export
            filepath: Path to output file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow(
                [
                    "Method",
                    "Precision",
                    "Recall",
                    "F1-Score",
                    "Accuracy",
                    "ROC-AUC",
                    "N_Detected",
                ]
            )

            # Write data
            for method_name, metrics in report.method_results.items():
                writer.writerow(
                    [
                        method_name,
                        metrics.get("precision", 0.0),
                        metrics.get("recall", 0.0),
                        metrics.get("f1_score", 0.0),
                        metrics.get("accuracy", 0.0),
                        metrics.get("roc_auc", "N/A"),
                        metrics.get("n_detected", 0),
                    ]
                )

        logger.info(f"Report exported to CSV: {filepath}")

    def generate_summary_text(self, report: AnomalyDetectionReport) -> str:
        """
        Generate human-readable text summary.

        Args:
            report: Report to summarize

        Returns:
            Text summary string
        """
        lines = []
        lines.append("=" * 80)
        lines.append("ANOMALY DETECTION REPORT")
        lines.append("=" * 80)
        lines.append(f"Report ID: {report.report_id}")
        lines.append(f"Timestamp: {report.timestamp}")
        lines.append("")
        lines.append("SUMMARY STATISTICS")
        lines.append("-" * 80)
        lines.append(f"Total Samples: {report.n_samples}")
        lines.append(f"Anomalies: {report.n_anomalies} ({100*report.n_anomalies/report.n_samples:.1f}%)")
        lines.append(f"Normal: {report.n_normal} ({100*report.n_normal/report.n_samples:.1f}%)")
        lines.append(f"Methods Evaluated: {report.summary_stats.get('n_methods', 0)}")
        lines.append("")

        if report.best_method:
            lines.append("BEST METHOD")
            lines.append("-" * 80)
            lines.append(f"Method: {report.best_method}")
            best_metrics = report.method_results[report.best_method]
            lines.append(f"F1-Score: {best_metrics.get('f1_score', 0.0):.3f}")
            lines.append(f"Precision: {best_metrics.get('precision', 0.0):.3f}")
            lines.append(f"Recall: {best_metrics.get('recall', 0.0):.3f}")
            lines.append("")

        lines.append("METHOD COMPARISON")
        lines.append("-" * 80)
        lines.append(f"{'Method':<30} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'ROC-AUC':<10}")
        lines.append("-" * 80)

        for method_name, metrics in report.method_results.items():
            roc_auc = metrics.get("roc_auc", None)
            roc_str = f"{roc_auc:.3f}" if roc_auc is not None else "N/A"
            lines.append(
                f"{method_name:<30} "
                f"{metrics.get('precision', 0.0):<10.3f} "
                f"{metrics.get('recall', 0.0):<10.3f} "
                f"{metrics.get('f1_score', 0.0):<10.3f} "
                f"{roc_str:<10}"
            )

        lines.append("=" * 80)

        return "\n".join(lines)

    def export_summary_text(self, report: AnomalyDetectionReport, filepath: Union[str, Path]) -> None:
        """
        Export text summary to file.

        Args:
            report: Report to export
            filepath: Path to output file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        summary_text = self.generate_summary_text(report)

        with open(filepath, "w") as f:
            f.write(summary_text)

        logger.info(f"Summary exported to text: {filepath}")
