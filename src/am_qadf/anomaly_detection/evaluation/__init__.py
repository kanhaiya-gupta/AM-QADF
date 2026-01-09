"""
Evaluation Framework for Anomaly Detection

This module provides comprehensive evaluation capabilities including
performance metrics, cross-validation, and method comparison.
"""

from .metrics import (
    AnomalyDetectionMetrics,
    calculate_classification_metrics,
    calculate_ranking_metrics,
    calculate_statistical_metrics,
)
from .cross_validation import AnomalyDetectionCV, k_fold_cv, time_series_cv, spatial_cv
from .comparison import (
    AnomalyDetectionComparison,
    compare_detectors,
    statistical_significance_test,
)

__all__ = [
    "AnomalyDetectionMetrics",
    "calculate_classification_metrics",
    "calculate_ranking_metrics",
    "calculate_statistical_metrics",
    "AnomalyDetectionCV",
    "k_fold_cv",
    "time_series_cv",
    "spatial_cv",
    "AnomalyDetectionComparison",
    "compare_detectors",
    "statistical_significance_test",
]
