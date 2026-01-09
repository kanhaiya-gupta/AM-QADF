"""
Unit tests for descriptive statistics.

Tests for DescriptiveStatistics and DescriptiveStatsAnalyzer.
"""

import pytest
import numpy as np
from am_qadf.analytics.statistical_analysis.descriptive_stats import (
    DescriptiveStatistics,
    DescriptiveStatsAnalyzer,
)


class TestDescriptiveStatistics:
    """Test suite for DescriptiveStatistics dataclass."""

    @pytest.mark.unit
    def test_statistics_creation(self):
        """Test creating DescriptiveStatistics."""
        stats = DescriptiveStatistics(
            signal_name="test_signal",
            mean=10.0,
            median=9.5,
            std=2.0,
            min=5.0,
            max=15.0,
            q25=8.0,
            q75=12.0,
            q95=14.0,
            q99=14.5,
            skewness=0.1,
            kurtosis=0.2,
            valid_count=1000,
            total_count=1000,
        )

        assert stats.signal_name == "test_signal"
        assert stats.mean == 10.0
        assert stats.median == 9.5
        assert stats.std == 2.0
        assert stats.min == 5.0
        assert stats.max == 15.0
        assert stats.valid_count == 1000
        assert stats.total_count == 1000

    @pytest.mark.unit
    def test_statistics_to_dict(self):
        """Test converting DescriptiveStatistics to dictionary."""
        stats = DescriptiveStatistics(
            signal_name="test_signal",
            mean=10.0,
            median=9.5,
            std=2.0,
            min=5.0,
            max=15.0,
            q25=8.0,
            q75=12.0,
            q95=14.0,
            q99=14.5,
            skewness=0.1,
            kurtosis=0.2,
            valid_count=1000,
            total_count=1000,
        )

        result = stats.to_dict()

        assert isinstance(result, dict)
        assert result["signal_name"] == "test_signal"
        assert result["mean"] == 10.0
        assert "valid_ratio" in result
        assert result["valid_ratio"] == 1.0


class TestDescriptiveStatsAnalyzer:
    """Test suite for DescriptiveStatsAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create a DescriptiveStatsAnalyzer instance."""
        return DescriptiveStatsAnalyzer()

    @pytest.mark.unit
    def test_analyzer_creation(self, analyzer):
        """Test creating DescriptiveStatsAnalyzer."""
        assert analyzer is not None

    @pytest.mark.unit
    def test_calculate_statistics(self, analyzer):
        """Test calculating descriptive statistics."""
        signal_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

        stats = analyzer.calculate_statistics(signal_array, "test_signal")

        assert isinstance(stats, DescriptiveStatistics)
        assert stats.signal_name == "test_signal"
        assert stats.mean > 0
        assert stats.median > 0
        assert stats.std >= 0
        assert stats.min <= stats.max
        assert stats.valid_count == 10
        assert stats.total_count == 10

    @pytest.mark.unit
    def test_calculate_statistics_2d(self, analyzer):
        """Test calculating statistics for 2D array."""
        signal_array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        stats = analyzer.calculate_statistics(signal_array, "test_signal")

        assert isinstance(stats, DescriptiveStatistics)
        assert stats.valid_count == 6

    @pytest.mark.unit
    def test_calculate_statistics_3d(self, analyzer):
        """Test calculating statistics for 3D array."""
        signal_array = np.ones((5, 5, 5))

        stats = analyzer.calculate_statistics(signal_array, "test_signal")

        assert isinstance(stats, DescriptiveStatistics)
        assert stats.valid_count == 125

    @pytest.mark.unit
    def test_calculate_statistics_with_nan(self, analyzer):
        """Test calculating statistics with NaN values."""
        signal_array = np.array([1.0, 2.0, np.nan, 4.0, 5.0])

        stats = analyzer.calculate_statistics(signal_array, "test_signal")

        assert isinstance(stats, DescriptiveStatistics)
        assert stats.valid_count == 4
        assert stats.total_count == 5

    @pytest.mark.unit
    def test_calculate_statistics_with_zeros(self, analyzer):
        """Test calculating statistics with zero values."""
        signal_array = np.array([1.0, 2.0, 0.0, 4.0, 5.0])

        stats = analyzer.calculate_statistics(signal_array, "test_signal")

        assert isinstance(stats, DescriptiveStatistics)
        assert stats.valid_count == 4
        assert stats.total_count == 5

    @pytest.mark.unit
    def test_calculate_statistics_empty(self, analyzer):
        """Test calculating statistics for empty array."""
        signal_array = np.array([])

        stats = analyzer.calculate_statistics(signal_array, "test_signal")

        assert isinstance(stats, DescriptiveStatistics)
        assert stats.valid_count == 0
        assert stats.mean == 0.0

    @pytest.mark.unit
    def test_calculate_statistics_all_invalid(self, analyzer):
        """Test calculating statistics when all values are invalid."""
        signal_array = np.array([np.nan, 0.0, np.nan, 0.0])

        stats = analyzer.calculate_statistics(signal_array, "test_signal")

        assert isinstance(stats, DescriptiveStatistics)
        assert stats.valid_count == 0
        assert stats.mean == 0.0
        assert stats.std == 0.0

    @pytest.mark.unit
    def test_calculate_per_voxel_statistics(self, analyzer):
        """Test calculating per-voxel statistics."""
        signal_arrays = {
            "signal1": np.array([1.0, 2.0, 3.0]),
            "signal2": np.array([4.0, 5.0, 6.0]),
        }

        result = analyzer.calculate_per_voxel_statistics(signal_arrays, "mean")

        assert isinstance(result, np.ndarray)
        assert len(result) == 3

    @pytest.mark.unit
    def test_calculate_per_voxel_statistics_median(self, analyzer):
        """Test calculating per-voxel median statistics."""
        signal_arrays = {
            "signal1": np.array([1.0, 2.0, 3.0]),
            "signal2": np.array([4.0, 5.0, 6.0]),
        }

        result = analyzer.calculate_per_voxel_statistics(signal_arrays, "median")

        assert isinstance(result, np.ndarray)
        assert len(result) == 3

    @pytest.mark.unit
    def test_assess_all_signals(self, analyzer):
        """Test assessing all signals from voxel data."""

        class MockVoxelData:
            def __init__(self):
                self.available_signals = ["signal1", "signal2"]

            def get_signal_array(self, signal_name, default=0.0):
                if signal_name == "signal1":
                    return np.array([1.0, 2.0, 3.0])
                elif signal_name == "signal2":
                    return np.array([4.0, 5.0, 6.0])
                return np.array([0.0])

        voxel_data = MockVoxelData()
        result = analyzer.assess_all_signals(voxel_data)

        assert isinstance(result, dict)
        assert "signal1" in result
        assert "signal2" in result
        assert isinstance(result["signal1"], DescriptiveStatistics)
        assert isinstance(result["signal2"], DescriptiveStatistics)

    @pytest.mark.unit
    def test_assess_all_signals_specific_list(self, analyzer):
        """Test assessing specific signals."""

        class MockVoxelData:
            def get_signal_array(self, signal_name, default=0.0):
                if signal_name == "signal1":
                    return np.array([1.0, 2.0, 3.0])
                return np.array([0.0])

        voxel_data = MockVoxelData()
        result = analyzer.assess_all_signals(voxel_data, ["signal1"])

        assert isinstance(result, dict)
        assert "signal1" in result
        assert len(result) == 1
