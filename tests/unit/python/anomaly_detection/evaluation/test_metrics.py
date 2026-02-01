"""
Unit tests for anomaly detection metrics.

Tests for AnomalyDetectionMetrics, calculate_classification_metrics, calculate_ranking_metrics, and calculate_statistical_metrics.
"""

import pytest
import numpy as np
from unittest.mock import patch
from am_qadf.anomaly_detection.evaluation.metrics import (
    AnomalyDetectionMetrics,
    calculate_classification_metrics,
    calculate_ranking_metrics,
    calculate_statistical_metrics,
)


class TestAnomalyDetectionMetrics:
    """Test suite for AnomalyDetectionMetrics dataclass."""

    @pytest.mark.unit
    def test_metrics_creation(self):
        """Test creating AnomalyDetectionMetrics."""
        metrics = AnomalyDetectionMetrics(
            precision=0.9,
            recall=0.8,
            f1_score=0.85,
            accuracy=0.9,
            specificity=0.95,
            sensitivity=0.8,
        )

        assert metrics.precision == 0.9
        assert metrics.recall == 0.8
        assert metrics.f1_score == 0.85
        assert metrics.accuracy == 0.9
        assert metrics.specificity == 0.95
        assert metrics.sensitivity == 0.8
        assert metrics.roc_auc is None
        assert metrics.pr_auc is None

    @pytest.mark.unit
    def test_metrics_creation_with_all_fields(self):
        """Test creating AnomalyDetectionMetrics with all fields."""
        cm = np.array([[90, 10], [5, 95]])
        metrics = AnomalyDetectionMetrics(
            precision=0.9,
            recall=0.95,
            f1_score=0.925,
            accuracy=0.925,
            specificity=0.9,
            sensitivity=0.95,
            roc_auc=0.98,
            pr_auc=0.97,
            average_precision=0.97,
            false_positive_rate=0.1,
            false_negative_rate=0.05,
            true_positive_rate=0.95,
            matthews_correlation=0.85,
            confusion_matrix=cm,
            n_samples=200,
            n_anomalies=100,
            n_normal=100,
        )

        assert metrics.roc_auc == 0.98
        assert metrics.pr_auc == 0.97
        assert metrics.confusion_matrix is not None
        assert metrics.n_samples == 200
        assert metrics.n_anomalies == 100
        assert metrics.n_normal == 100


try:
    from sklearn.metrics import precision_score

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
class TestCalculateClassificationMetrics:
    """Test suite for calculate_classification_metrics function."""

    @pytest.fixture
    def sample_labels_and_predictions(self):
        """Create sample labels and predictions."""
        np.random.seed(42)
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 0])
        y_scores = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.4, 0.2, 0.85, 0.95, 0.15])
        return y_true, y_pred, y_scores

    @pytest.mark.unit
    def test_calculate_classification_metrics(self, sample_labels_and_predictions):
        """Test calculating classification metrics."""

        y_true, y_pred, y_scores = sample_labels_and_predictions
        metrics = calculate_classification_metrics(y_true, y_pred, y_scores)

        assert isinstance(metrics, AnomalyDetectionMetrics)
        assert 0 <= metrics.precision <= 1
        assert 0 <= metrics.recall <= 1
        assert 0 <= metrics.f1_score <= 1
        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.specificity <= 1
        assert metrics.sensitivity == metrics.recall
        assert metrics.n_samples == len(y_true)
        assert metrics.n_anomalies == np.sum(y_true)
        assert metrics.n_normal == len(y_true) - np.sum(y_true)

    @pytest.mark.unit
    def test_calculate_classification_metrics_without_scores(self):
        """Test calculating metrics without scores."""

        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 0])

        metrics = calculate_classification_metrics(y_true, y_pred)

        assert isinstance(metrics, AnomalyDetectionMetrics)
        assert metrics.roc_auc is None
        assert metrics.pr_auc is None

    @pytest.mark.unit
    def test_calculate_classification_metrics_perfect_predictions(self):
        """Test metrics with perfect predictions."""

        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])
        y_scores = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.95])

        metrics = calculate_classification_metrics(y_true, y_pred, y_scores)

        assert metrics.accuracy == 1.0
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1_score == 1.0

    @pytest.mark.unit
    def test_calculate_classification_metrics_all_wrong(self):
        """Test metrics with all wrong predictions."""

        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        y_scores = np.array([0.9, 0.8, 0.1, 0.2])

        metrics = calculate_classification_metrics(y_true, y_pred, y_scores)

        assert metrics.accuracy == 0.0
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0

    @pytest.mark.unit
    def test_calculate_classification_metrics_confusion_matrix(self):
        """Test confusion matrix calculation."""

        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 0])

        metrics = calculate_classification_metrics(y_true, y_pred)

        assert metrics.confusion_matrix is not None
        assert metrics.confusion_matrix.shape == (2, 2)

    @pytest.mark.unit
    def test_calculate_classification_metrics_matthews_correlation(self):
        """Test Matthews correlation coefficient calculation."""

        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 0])

        metrics = calculate_classification_metrics(y_true, y_pred)

        assert metrics.matthews_correlation is not None
        assert -1 <= metrics.matthews_correlation <= 1


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
class TestCalculateRankingMetrics:
    """Test suite for calculate_ranking_metrics function."""

    @pytest.mark.unit
    def test_calculate_ranking_metrics(self):
        """Test calculating ranking metrics."""

        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_scores = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.95])

        metrics = calculate_ranking_metrics(y_true, y_scores)

        assert isinstance(metrics, dict)
        assert "roc_auc" in metrics
        assert "pr_auc" in metrics
        assert "average_precision" in metrics
        if metrics["roc_auc"] is not None:
            assert 0 <= metrics["roc_auc"] <= 1
        if metrics["pr_auc"] is not None:
            assert 0 <= metrics["pr_auc"] <= 1

    @pytest.mark.unit
    def test_calculate_ranking_metrics_perfect_separation(self):
        """Test ranking metrics with perfect score separation."""

        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.8, 0.9])

        metrics = calculate_ranking_metrics(y_true, y_scores)

        if metrics["roc_auc"] is not None:
            assert metrics["roc_auc"] == 1.0

    @pytest.mark.unit
    def test_calculate_ranking_metrics_single_class(self):
        """Test ranking metrics with single class."""

        y_true = np.array([0, 0, 0, 0])
        y_scores = np.array([0.1, 0.2, 0.3, 0.4])

        metrics = calculate_ranking_metrics(y_true, y_scores)

        # Should handle gracefully (may return None)
        assert isinstance(metrics, dict)


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
class TestCalculateStatisticalMetrics:
    """Test suite for calculate_statistical_metrics function."""

    @pytest.mark.unit
    def test_calculate_statistical_metrics(self):
        """Test calculating statistical metrics."""

        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 0])

        metrics = calculate_statistical_metrics(y_true, y_pred)

        assert isinstance(metrics, dict)
        assert "false_positive_rate" in metrics
        assert "false_negative_rate" in metrics
        assert "true_positive_rate" in metrics
        assert "true_negative_rate" in metrics
        assert 0 <= metrics["false_positive_rate"] <= 1
        assert 0 <= metrics["false_negative_rate"] <= 1
        assert 0 <= metrics["true_positive_rate"] <= 1
        assert 0 <= metrics["true_negative_rate"] <= 1

    @pytest.mark.unit
    def test_calculate_statistical_metrics_matthews_correlation(self):
        """Test Matthews correlation in statistical metrics."""

        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 0])

        metrics = calculate_statistical_metrics(y_true, y_pred)

        if "matthews_correlation" in metrics:
            assert -1 <= metrics["matthews_correlation"] <= 1

    @pytest.mark.unit
    def test_calculate_statistical_metrics_perfect_predictions(self):
        """Test statistical metrics with perfect predictions."""

        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])

        metrics = calculate_statistical_metrics(y_true, y_pred)

        assert metrics["false_positive_rate"] == 0.0
        assert metrics["false_negative_rate"] == 0.0
        assert metrics["true_positive_rate"] == 1.0
        assert metrics["true_negative_rate"] == 1.0
