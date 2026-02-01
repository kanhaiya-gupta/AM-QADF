"""
Unit tests for statistical analysis client.

Tests for AdvancedAnalyticsClient.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from am_qadf.analytics.statistical_analysis.client import AdvancedAnalyticsClient


class MockVoxelData:
    """Mock voxel data object for testing."""

    def __init__(self, signals: dict, dims: tuple = (10, 10, 10)):
        """Initialize with signal dictionary."""
        self._signals = signals
        self.dims = dims
        self.available_signals = list(signals.keys())

    def get_signal_array(self, signal_name: str, default: float = 0.0) -> np.ndarray:
        """Get signal array by name."""
        return self._signals.get(signal_name, np.full(self.dims, default))


class TestAdvancedAnalyticsClient:
    """Test suite for AdvancedAnalyticsClient class."""

    @pytest.fixture
    def client(self):
        """Create an AdvancedAnalyticsClient instance."""
        return AdvancedAnalyticsClient()

    @pytest.fixture
    def mock_voxel_data(self):
        """Create mock voxel data."""
        signals = {
            "signal1": np.ones((10, 10, 10)),
            "signal2": np.ones((10, 10, 10)) * 2.0,
        }
        return MockVoxelData(signals, dims=(10, 10, 10))

    @pytest.mark.unit
    def test_client_creation(self, client):
        """Test creating AdvancedAnalyticsClient."""
        assert client is not None
        assert client.stats_analyzer is not None
        assert client.correlation_analyzer is not None
        assert client.trend_analyzer is not None
        assert client.pattern_analyzer is not None

    @pytest.mark.unit
    def test_calculate_descriptive_statistics(self, client, mock_voxel_data):
        """Test calculating descriptive statistics."""
        result = client.calculate_descriptive_statistics(mock_voxel_data, ["signal1", "signal2"])

        assert isinstance(result, dict)
        assert "signal1" in result
        assert "signal2" in result
        assert result["signal1"].mean > 0
        assert result["signal2"].mean > result["signal1"].mean

    @pytest.mark.unit
    def test_calculate_descriptive_statistics_all_signals(self, client, mock_voxel_data):
        """Test calculating descriptive statistics for all signals."""
        result = client.calculate_descriptive_statistics(mock_voxel_data)

        assert isinstance(result, dict)
        assert len(result) == 2

    @pytest.mark.unit
    def test_analyze_correlations(self, client, mock_voxel_data):
        """Test analyzing correlations."""
        result = client.analyze_correlations(mock_voxel_data, ["signal1", "signal2"])

        assert result is not None
        assert hasattr(result, "signal_correlations")
        assert hasattr(result, "to_dict")

    @pytest.mark.unit
    def test_analyze_correlations_with_spatial(self, client, mock_voxel_data):
        """Test analyzing correlations with spatial analysis."""
        result = client.analyze_correlations(mock_voxel_data, ["signal1", "signal2"], include_spatial=True)

        assert result is not None

    @pytest.mark.unit
    def test_analyze_correlations_with_temporal(self, client, mock_voxel_data):
        """Test analyzing correlations with temporal analysis."""
        result = client.analyze_correlations(mock_voxel_data, ["signal1", "signal2"], include_temporal=True)

        assert result is not None

    @pytest.mark.unit
    def test_analyze_trends(self, client, mock_voxel_data):
        """Test analyzing trends."""
        result = client.analyze_trends(mock_voxel_data, ["signal1", "signal2"])

        assert result is not None
        assert hasattr(result, "temporal_trends")
        assert hasattr(result, "spatial_trends")

    @pytest.mark.unit
    def test_analyze_trends_with_spatial(self, client, mock_voxel_data):
        """Test analyzing trends with spatial analysis."""
        result = client.analyze_trends(mock_voxel_data, ["signal1", "signal2"], include_spatial=True)

        assert result is not None

    @pytest.mark.unit
    def test_analyze_patterns(self, client, mock_voxel_data):
        """Test analyzing patterns."""
        result = client.analyze_patterns(mock_voxel_data, ["signal1", "signal2"])

        assert result is not None
        assert hasattr(result, "spatial_patterns")
        assert hasattr(result, "temporal_patterns")

    @pytest.mark.unit
    def test_analyze_patterns_with_anomalies(self, client, mock_voxel_data):
        """Test analyzing patterns with anomaly detection."""
        result = client.analyze_patterns(mock_voxel_data, ["signal1", "signal2"], include_anomalies=True)

        assert result is not None

    @pytest.mark.unit
    def test_analyze_patterns_with_process(self, client, mock_voxel_data):
        """Test analyzing patterns with process pattern detection."""
        result = client.analyze_patterns(mock_voxel_data, ["signal1", "signal2"], include_process=True)

        assert result is not None

    @pytest.mark.unit
    def test_comprehensive_analysis(self, client, mock_voxel_data):
        """Test comprehensive analysis."""
        result = client.comprehensive_analysis(mock_voxel_data, ["signal1", "signal2"])

        assert isinstance(result, dict)
        assert "descriptive_statistics" in result
        assert "correlations" in result
        assert "trends" in result
        assert "patterns" in result
        assert "summary" in result

    @pytest.mark.unit
    def test_comprehensive_analysis_all_options(self, client, mock_voxel_data):
        """Test comprehensive analysis with all options enabled."""
        result = client.comprehensive_analysis(
            mock_voxel_data,
            ["signal1", "signal2"],
            include_spatial=True,
            include_temporal=True,
            include_anomalies=True,
            include_process=True,
        )

        assert isinstance(result, dict)
        assert "summary" in result

    @pytest.mark.unit
    def test_generate_analysis_report(self, client, mock_voxel_data):
        """Test generating analysis report."""
        analysis_results = client.comprehensive_analysis(mock_voxel_data, ["signal1", "signal2"])

        report = client.generate_analysis_report(analysis_results)

        assert isinstance(report, str)
        assert len(report) > 0
