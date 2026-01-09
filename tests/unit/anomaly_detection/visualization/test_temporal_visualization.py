"""
Unit tests for temporal anomaly visualization.

Tests for TemporalAnomalyVisualizer class.
"""

import pytest
import numpy as np
from unittest.mock import patch
from am_qadf.anomaly_detection.visualization.temporal_visualization import (
    TemporalAnomalyVisualizer,
)


class TestTemporalAnomalyVisualizer:
    """Test suite for TemporalAnomalyVisualizer class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample temporal data."""
        np.random.seed(42)
        n_samples = 100
        time_series = np.arange(n_samples)
        layer_numbers = np.repeat(np.arange(10), 10)
        anomaly_scores = np.random.rand(n_samples)
        anomaly_labels = anomaly_scores > 0.7
        return time_series, layer_numbers, anomaly_scores, anomaly_labels

    @pytest.mark.unit
    def test_visualizer_creation(self):
        """Test creating TemporalAnomalyVisualizer."""
        visualizer = TemporalAnomalyVisualizer()

        assert visualizer.time_series is None
        assert visualizer.layer_numbers is None
        assert visualizer.anomaly_scores is None
        assert visualizer.anomaly_labels is None

    @pytest.mark.unit
    def test_set_data(self, sample_data):
        """Test setting data for visualization."""
        time_series, layer_numbers, anomaly_scores, anomaly_labels = sample_data
        visualizer = TemporalAnomalyVisualizer()
        visualizer.set_data(time_series, layer_numbers, anomaly_scores, anomaly_labels)

        assert visualizer.time_series is not None
        assert visualizer.layer_numbers is not None
        assert visualizer.anomaly_scores is not None
        assert visualizer.anomaly_labels is not None

    @pytest.mark.unit
    def test_set_data_minimal(self):
        """Test setting data with minimal information."""
        visualizer = TemporalAnomalyVisualizer()
        anomaly_scores = np.random.rand(50)
        visualizer.set_data(anomaly_scores=anomaly_scores)

        assert visualizer.anomaly_scores is not None
        assert visualizer.time_series is None
        assert visualizer.layer_numbers is None

    @pytest.mark.unit
    @patch("matplotlib.pyplot.show")
    def test_plot_time_series(self, mock_show, sample_data):
        """Test plotting time series."""
        time_series, layer_numbers, anomaly_scores, anomaly_labels = sample_data
        visualizer = TemporalAnomalyVisualizer()
        visualizer.set_data(time_series, layer_numbers, anomaly_scores, anomaly_labels)

        fig = visualizer.plot_time_series(show=False)

        assert fig is not None
        mock_show.assert_not_called()

    @pytest.mark.unit
    @patch("matplotlib.pyplot.show")
    def test_plot_time_series_with_threshold(self, mock_show, sample_data):
        """Test plotting time series with threshold."""
        time_series, layer_numbers, anomaly_scores, anomaly_labels = sample_data
        visualizer = TemporalAnomalyVisualizer()
        visualizer.set_data(time_series, layer_numbers, anomaly_scores, anomaly_labels)

        fig = visualizer.plot_time_series(threshold=0.5, show=False)

        assert fig is not None

    @pytest.mark.unit
    def test_plot_time_series_no_data(self):
        """Test plotting time series without data."""
        visualizer = TemporalAnomalyVisualizer()
        fig = visualizer.plot_time_series(show=False)

        assert fig is None

    @pytest.mark.unit
    @patch("matplotlib.pyplot.show")
    def test_plot_time_series_with_layer_numbers(self, mock_show):
        """Test plotting time series using layer numbers."""
        visualizer = TemporalAnomalyVisualizer()
        layer_numbers = np.repeat(np.arange(10), 5)
        anomaly_scores = np.random.rand(50)
        visualizer.set_data(layer_numbers=layer_numbers, anomaly_scores=anomaly_scores)

        fig = visualizer.plot_time_series(show=False)

        assert fig is not None

    @pytest.mark.unit
    @patch("matplotlib.pyplot.show")
    def test_plot_layer_analysis(self, mock_show, sample_data):
        """Test plotting layer analysis."""
        time_series, layer_numbers, anomaly_scores, anomaly_labels = sample_data
        visualizer = TemporalAnomalyVisualizer()
        visualizer.set_data(time_series, layer_numbers, anomaly_scores, anomaly_labels)

        fig = visualizer.plot_layer_analysis(show=False)

        assert fig is not None
        mock_show.assert_not_called()

    @pytest.mark.unit
    def test_plot_layer_analysis_no_layer_numbers(self):
        """Test plotting layer analysis without layer numbers."""
        visualizer = TemporalAnomalyVisualizer()
        anomaly_scores = np.random.rand(50)
        visualizer.set_data(anomaly_scores=anomaly_scores)

        fig = visualizer.plot_layer_analysis(show=False)

        assert fig is None

    @pytest.mark.unit
    @patch("matplotlib.pyplot.show")
    def test_plot_anomaly_timeline(self, mock_show, sample_data):
        """Test plotting anomaly timeline."""
        time_series, layer_numbers, anomaly_scores, anomaly_labels = sample_data
        visualizer = TemporalAnomalyVisualizer()
        visualizer.set_data(time_series, layer_numbers, anomaly_scores, anomaly_labels)

        fig = visualizer.plot_anomaly_timeline(show=False)

        assert fig is not None
        mock_show.assert_not_called()

    @pytest.mark.unit
    def test_plot_anomaly_timeline_no_labels(self):
        """Test plotting timeline without labels."""
        visualizer = TemporalAnomalyVisualizer()
        anomaly_scores = np.random.rand(50)
        visualizer.set_data(anomaly_scores=anomaly_scores)

        fig = visualizer.plot_anomaly_timeline(show=False)

        assert fig is None

    @pytest.mark.unit
    @patch("matplotlib.pyplot.show")
    def test_plot_trend_analysis(self, mock_show, sample_data):
        """Test plotting trend analysis."""
        time_series, layer_numbers, anomaly_scores, anomaly_labels = sample_data
        visualizer = TemporalAnomalyVisualizer()
        visualizer.set_data(time_series, layer_numbers, anomaly_scores, anomaly_labels)

        fig = visualizer.plot_trend_analysis(show=False)

        assert fig is not None
        mock_show.assert_not_called()

    @pytest.mark.unit
    @patch("matplotlib.pyplot.show")
    def test_plot_trend_analysis_custom_window(self, mock_show, sample_data):
        """Test plotting trend analysis with custom window size."""
        time_series, layer_numbers, anomaly_scores, anomaly_labels = sample_data
        visualizer = TemporalAnomalyVisualizer()
        visualizer.set_data(time_series, layer_numbers, anomaly_scores, anomaly_labels)

        fig = visualizer.plot_trend_analysis(window_size=20, show=False)

        assert fig is not None

    @pytest.mark.unit
    def test_plot_trend_analysis_no_data(self):
        """Test plotting trend analysis without data."""
        visualizer = TemporalAnomalyVisualizer()
        fig = visualizer.plot_trend_analysis(show=False)

        assert fig is None

    @pytest.mark.unit
    @patch("matplotlib.pyplot.show")
    def test_plot_trend_analysis_small_window(self, mock_show):
        """Test plotting trend analysis with small dataset."""
        visualizer = TemporalAnomalyVisualizer()
        anomaly_scores = np.random.rand(5)  # Very small dataset
        visualizer.set_data(anomaly_scores=anomaly_scores)

        fig = visualizer.plot_trend_analysis(window_size=10, show=False)

        # Should handle gracefully
        assert fig is not None
