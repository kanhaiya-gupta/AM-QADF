"""
Unit tests for comparison visualization.

Tests for ComparisonVisualizer class.
"""

import pytest
import numpy as np
from unittest.mock import patch, Mock
from am_qadf.anomaly_detection.visualization.comparison_visualization import (
    ComparisonVisualizer,
)
from am_qadf.anomaly_detection.evaluation.metrics import AnomalyDetectionMetrics


class TestComparisonVisualizer:
    """Test suite for ComparisonVisualizer class."""

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
            "method3": AnomalyDetectionMetrics(
                precision=0.8,
                recall=0.75,
                f1_score=0.775,
                accuracy=0.85,
                specificity=0.9,
                sensitivity=0.75,
                roc_auc=0.88,
            ),
        }

    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0] * 5)
        y_pred1 = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 0] * 5)
        y_pred2 = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0] * 5)
        y_pred3 = np.array([0, 0, 1, 0, 0, 1, 0, 1, 1, 0] * 5)
        y_scores1 = np.random.rand(50)
        y_scores2 = np.random.rand(50)
        y_scores3 = np.random.rand(50)
        return y_true, y_pred1, y_pred2, y_pred3, y_scores1, y_scores2, y_scores3

    @pytest.mark.unit
    def test_visualizer_creation(self):
        """Test creating ComparisonVisualizer."""
        visualizer = ComparisonVisualizer()

        assert visualizer.method_results == {}

    @pytest.mark.unit
    def test_add_method_results(self, sample_metrics, sample_predictions):
        """Test adding method results."""
        y_true, y_pred1, _, _, y_scores1, _, _ = sample_predictions
        visualizer = ComparisonVisualizer()
        visualizer.add_method_results("method1", sample_metrics["method1"], y_true, y_pred1, y_scores1)

        assert "method1" in visualizer.method_results
        assert visualizer.method_results["method1"]["metrics"] == sample_metrics["method1"]
        assert np.array_equal(visualizer.method_results["method1"]["y_true"], y_true)

    @pytest.mark.unit
    def test_add_multiple_methods(self, sample_metrics, sample_predictions):
        """Test adding multiple method results."""
        y_true, y_pred1, y_pred2, _, y_scores1, y_scores2, _ = sample_predictions
        visualizer = ComparisonVisualizer()

        visualizer.add_method_results("method1", sample_metrics["method1"], y_true, y_pred1, y_scores1)
        visualizer.add_method_results("method2", sample_metrics["method2"], y_true, y_pred2, y_scores2)

        assert len(visualizer.method_results) == 2
        assert "method1" in visualizer.method_results
        assert "method2" in visualizer.method_results

    @pytest.mark.unit
    @patch("matplotlib.pyplot.show")
    def test_plot_method_comparison(self, mock_show, sample_metrics, sample_predictions):
        """Test plotting method comparison."""
        y_true, y_pred1, y_pred2, _, y_scores1, y_scores2, _ = sample_predictions
        visualizer = ComparisonVisualizer()
        visualizer.add_method_results("method1", sample_metrics["method1"], y_true, y_pred1, y_scores1)
        visualizer.add_method_results("method2", sample_metrics["method2"], y_true, y_pred2, y_scores2)

        fig = visualizer.plot_method_comparison(show=False)

        assert fig is not None
        mock_show.assert_not_called()

    @pytest.mark.unit
    def test_plot_method_comparison_no_results(self):
        """Test plotting comparison with no results."""
        visualizer = ComparisonVisualizer()
        fig = visualizer.plot_method_comparison(show=False)

        assert fig is None

    @pytest.mark.unit
    @patch("matplotlib.pyplot.show")
    def test_plot_method_comparison_different_metrics(self, mock_show, sample_metrics, sample_predictions):
        """Test plotting comparison with different metrics."""
        y_true, y_pred1, y_pred2, _, y_scores1, y_scores2, _ = sample_predictions
        visualizer = ComparisonVisualizer()
        visualizer.add_method_results("method1", sample_metrics["method1"], y_true, y_pred1, y_scores1)
        visualizer.add_method_results("method2", sample_metrics["method2"], y_true, y_pred2, y_scores2)

        for metric in ["f1_score", "precision", "recall", "accuracy"]:
            fig = visualizer.plot_method_comparison(metric=metric, show=False)
            assert fig is not None

    @pytest.mark.unit
    @patch("matplotlib.pyplot.show")
    def test_plot_precision_recall_comparison(self, mock_show, sample_metrics, sample_predictions):
        """Test plotting precision-recall comparison."""
        y_true, y_pred1, y_pred2, _, y_scores1, y_scores2, _ = sample_predictions
        visualizer = ComparisonVisualizer()
        visualizer.add_method_results("method1", sample_metrics["method1"], y_true, y_pred1, y_scores1)
        visualizer.add_method_results("method2", sample_metrics["method2"], y_true, y_pred2, y_scores2)

        fig = visualizer.plot_precision_recall_comparison(show=False)

        assert fig is not None
        mock_show.assert_not_called()

    @pytest.mark.unit
    def test_plot_precision_recall_comparison_no_results(self):
        """Test plotting precision-recall with no results."""
        visualizer = ComparisonVisualizer()
        fig = visualizer.plot_precision_recall_comparison(show=False)

        assert fig is None

    @pytest.mark.unit
    @patch("matplotlib.pyplot.show")
    @patch(
        "am_qadf.anomaly_detection.visualization.comparison_visualization.SKLEARN_AVAILABLE",
        True,
    )
    def test_plot_roc_curves(self, mock_show, sample_metrics, sample_predictions):
        """Test plotting ROC curves."""
        y_true, y_pred1, y_pred2, _, y_scores1, y_scores2, _ = sample_predictions
        visualizer = ComparisonVisualizer()
        visualizer.add_method_results("method1", sample_metrics["method1"], y_true, y_pred1, y_scores1)
        visualizer.add_method_results("method2", sample_metrics["method2"], y_true, y_pred2, y_scores2)

        fig = visualizer.plot_roc_curves(show=False)

        assert fig is not None
        mock_show.assert_not_called()

    @pytest.mark.unit
    def test_plot_roc_curves_no_results(self):
        """Test plotting ROC curves with no results."""
        visualizer = ComparisonVisualizer()
        fig = visualizer.plot_roc_curves(show=False)

        assert fig is None

    @pytest.mark.unit
    @patch("matplotlib.pyplot.show")
    @patch(
        "am_qadf.anomaly_detection.visualization.comparison_visualization.SKLEARN_AVAILABLE",
        True,
    )
    def test_plot_confusion_matrices(self, mock_show, sample_metrics, sample_predictions):
        """Test plotting confusion matrices."""
        y_true, y_pred1, y_pred2, _, y_scores1, y_scores2, _ = sample_predictions
        visualizer = ComparisonVisualizer()
        visualizer.add_method_results("method1", sample_metrics["method1"], y_true, y_pred1, y_scores1)
        visualizer.add_method_results("method2", sample_metrics["method2"], y_true, y_pred2, y_scores2)

        fig = visualizer.plot_confusion_matrices(show=False)

        assert fig is not None
        mock_show.assert_not_called()

    @pytest.mark.unit
    @patch("matplotlib.pyplot.show")
    @patch(
        "am_qadf.anomaly_detection.visualization.comparison_visualization.SKLEARN_AVAILABLE",
        True,
    )
    def test_plot_confusion_matrices_specific_methods(self, mock_show, sample_metrics, sample_predictions):
        """Test plotting confusion matrices for specific methods."""
        y_true, y_pred1, y_pred2, _, y_scores1, y_scores2, _ = sample_predictions
        visualizer = ComparisonVisualizer()
        visualizer.add_method_results("method1", sample_metrics["method1"], y_true, y_pred1, y_scores1)
        visualizer.add_method_results("method2", sample_metrics["method2"], y_true, y_pred2, y_scores2)

        fig = visualizer.plot_confusion_matrices(methods=["method1"], show=False)

        assert fig is not None

    @pytest.mark.unit
    def test_plot_confusion_matrices_no_results(self):
        """Test plotting confusion matrices with no results."""
        visualizer = ComparisonVisualizer()
        fig = visualizer.plot_confusion_matrices(show=False)

        assert fig is None

    @pytest.mark.unit
    @patch("matplotlib.pyplot.show")
    def test_plot_comprehensive_comparison(self, mock_show, sample_metrics, sample_predictions):
        """Test plotting comprehensive comparison."""
        y_true, y_pred1, y_pred2, y_pred3, y_scores1, y_scores2, y_scores3 = sample_predictions
        visualizer = ComparisonVisualizer()
        visualizer.add_method_results("method1", sample_metrics["method1"], y_true, y_pred1, y_scores1)
        visualizer.add_method_results("method2", sample_metrics["method2"], y_true, y_pred2, y_scores2)
        visualizer.add_method_results("method3", sample_metrics["method3"], y_true, y_pred3, y_scores3)

        fig = visualizer.plot_comprehensive_comparison(show=False)

        assert fig is not None
        mock_show.assert_not_called()

    @pytest.mark.unit
    def test_plot_comprehensive_comparison_no_results(self):
        """Test plotting comprehensive comparison with no results."""
        visualizer = ComparisonVisualizer()
        fig = visualizer.plot_comprehensive_comparison(show=False)

        assert fig is None

    @pytest.mark.unit
    @patch("matplotlib.pyplot.show")
    def test_plot_comprehensive_comparison_single_method(self, mock_show, sample_metrics, sample_predictions):
        """Test plotting comprehensive comparison with single method."""
        y_true, y_pred1, _, _, y_scores1, _, _ = sample_predictions
        visualizer = ComparisonVisualizer()
        visualizer.add_method_results("method1", sample_metrics["method1"], y_true, y_pred1, y_scores1)

        fig = visualizer.plot_comprehensive_comparison(show=False)

        assert fig is not None
