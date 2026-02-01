"""
Unit tests for trend analysis.

Tests for TrendResults and TrendAnalyzer.
"""

import pytest
import numpy as np
from am_qadf.analytics.statistical_analysis.trends import (
    TrendResults,
    TrendAnalyzer,
)


class TestTrendResults:
    """Test suite for TrendResults dataclass."""

    @pytest.mark.unit
    def test_results_creation(self):
        """Test creating TrendResults."""
        temporal_trends = {"signal1": {"slope": 0.5, "trend_direction": "increasing"}}
        spatial_trends = {"signal1": {"slope": 0.3, "trend_direction": "increasing"}}

        results = TrendResults(temporal_trends=temporal_trends, spatial_trends=spatial_trends)

        assert len(results.temporal_trends) == 1
        assert len(results.spatial_trends) == 1
        assert results.build_progression is None
        assert results.quality_evolution is None

    @pytest.mark.unit
    def test_results_to_dict(self):
        """Test converting TrendResults to dictionary."""
        temporal_trends = {"signal1": {"slope": 0.5}}
        spatial_trends = {"signal1": {"slope": 0.3}}

        results = TrendResults(temporal_trends=temporal_trends, spatial_trends=spatial_trends)

        result_dict = results.to_dict()

        assert isinstance(result_dict, dict)
        assert "temporal_trends" in result_dict
        assert "spatial_trends" in result_dict


class TestTrendAnalyzer:
    """Test suite for TrendAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create a TrendAnalyzer instance."""
        return TrendAnalyzer()

    @pytest.mark.unit
    def test_analyzer_creation(self, analyzer):
        """Test creating TrendAnalyzer."""
        assert analyzer is not None

    @pytest.mark.unit
    def test_analyze_temporal_trend(self, analyzer):
        """Test analyzing temporal trend."""
        # Create signal array with increasing trend along z-axis
        signal_array = np.zeros((5, 5, 10))
        for z in range(10):
            signal_array[:, :, z] = z * 0.1

        trend = analyzer.analyze_temporal_trend(signal_array, "test_signal")

        assert isinstance(trend, dict)
        assert "slope" in trend
        assert "intercept" in trend
        assert "r_value" in trend
        assert "p_value" in trend
        assert "trend_direction" in trend
        assert trend["slope"] > 0  # Should be increasing
        assert trend["trend_direction"] == "increasing"

    @pytest.mark.unit
    def test_analyze_temporal_trend_decreasing(self, analyzer):
        """Test analyzing decreasing temporal trend."""
        # Create signal array with decreasing trend
        signal_array = np.zeros((5, 5, 10))
        for z in range(10):
            signal_array[:, :, z] = (10 - z) * 0.1

        trend = analyzer.analyze_temporal_trend(signal_array, "test_signal")

        assert trend["slope"] < 0  # Should be decreasing
        assert trend["trend_direction"] == "decreasing"

    @pytest.mark.unit
    def test_analyze_temporal_trend_insufficient_layers(self, analyzer):
        """Test analyzing temporal trend with insufficient layers."""
        signal_array = np.zeros((5, 5, 1))

        trend = analyzer.analyze_temporal_trend(signal_array, "test_signal")

        assert trend["slope"] == 0.0
        assert trend["trend_direction"] == "none"

    @pytest.mark.unit
    def test_analyze_spatial_trend_x_axis(self, analyzer):
        """Test analyzing spatial trend along x-axis."""
        # Create signal array with increasing trend along x-axis
        signal_array = np.zeros((10, 5, 5))
        for x in range(10):
            signal_array[x, :, :] = x * 0.1

        trend = analyzer.analyze_spatial_trend(signal_array, axis=0, signal_name="test_signal")

        assert isinstance(trend, dict)
        assert "slope" in trend
        assert trend["slope"] > 0

    @pytest.mark.unit
    def test_analyze_spatial_trend_y_axis(self, analyzer):
        """Test analyzing spatial trend along y-axis."""
        signal_array = np.zeros((5, 10, 5))
        for y in range(10):
            signal_array[:, y, :] = y * 0.1

        trend = analyzer.analyze_spatial_trend(signal_array, axis=1, signal_name="test_signal")

        assert isinstance(trend, dict)
        assert trend["slope"] > 0

    @pytest.mark.unit
    def test_analyze_spatial_trend_invalid_axis(self, analyzer):
        """Test analyzing spatial trend with invalid axis."""
        signal_array = np.zeros((5, 5, 5))

        trend = analyzer.analyze_spatial_trend(signal_array, axis=5, signal_name="test_signal")

        assert trend["slope"] == 0.0
        assert trend["trend_direction"] == "none"

    @pytest.mark.unit
    def test_analyze_trends(self, analyzer):
        """Test comprehensive trend analysis."""

        class MockVoxelData:
            def __init__(self):
                self.available_signals = ["signal1"]

            def get_signal_array(self, signal_name, default=0.0):
                signal_array = np.zeros((5, 5, 10))
                for z in range(10):
                    signal_array[:, :, z] = z * 0.1
                return signal_array

        voxel_data = MockVoxelData()
        result = analyzer.analyze_trends(voxel_data, ["signal1"])

        assert isinstance(result, TrendResults)
        assert "signal1" in result.temporal_trends
        assert "signal1" in result.spatial_trends

    @pytest.mark.unit
    def test_analyze_build_progression(self, analyzer):
        """Test analyzing build progression."""

        class MockVoxelData:
            def get_signal_array(self, signal_name, default=0.0):
                signal_array = np.zeros((5, 5, 10))
                for z in range(10):
                    signal_array[:, :, z] = z * 0.1
                return signal_array

        voxel_data = MockVoxelData()
        progression = analyzer.analyze_build_progression(voxel_data, ["signal1"])

        assert isinstance(progression, dict)
        assert "layer_progression" in progression or "progression_metrics" in progression
