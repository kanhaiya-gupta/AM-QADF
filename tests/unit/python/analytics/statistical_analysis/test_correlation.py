"""
Unit tests for correlation analysis.

Tests for CorrelationResults and CorrelationAnalyzer.
"""

import pytest
import numpy as np
from am_qadf.analytics.statistical_analysis.correlation import (
    CorrelationResults,
    CorrelationAnalyzer,
)


class TestCorrelationResults:
    """Test suite for CorrelationResults dataclass."""

    @pytest.mark.unit
    def test_results_creation(self):
        """Test creating CorrelationResults."""
        signal_correlations = {("signal1", "signal2"): 0.8}

        results = CorrelationResults(
            signal_correlations=signal_correlations,
            correlation_matrix=np.array([[1.0, 0.8], [0.8, 1.0]]),
            signal_names=["signal1", "signal2"],
        )

        assert len(results.signal_correlations) == 1
        assert results.correlation_matrix is not None
        assert len(results.signal_names) == 2

    @pytest.mark.unit
    def test_results_to_dict(self):
        """Test converting CorrelationResults to dictionary."""
        signal_correlations = {("signal1", "signal2"): 0.8}

        results = CorrelationResults(
            signal_correlations=signal_correlations,
            correlation_matrix=np.array([[1.0, 0.8], [0.8, 1.0]]),
            signal_names=["signal1", "signal2"],
        )

        result_dict = results.to_dict()

        assert isinstance(result_dict, dict)
        assert "signal_correlations" in result_dict
        assert "num_correlations" in result_dict
        assert result_dict["num_correlations"] == 1
        assert "correlation_matrix_shape" in result_dict


class TestCorrelationAnalyzer:
    """Test suite for CorrelationAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create a CorrelationAnalyzer instance."""
        return CorrelationAnalyzer()

    @pytest.mark.unit
    def test_analyzer_creation(self, analyzer):
        """Test creating CorrelationAnalyzer."""
        assert analyzer is not None

    @pytest.mark.unit
    def test_calculate_signal_correlations(self, analyzer):
        """Test calculating signal correlations."""
        signal_arrays = {
            "signal1": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            "signal2": np.array([2.0, 4.0, 6.0, 8.0, 10.0]),  # Perfect correlation
        }

        correlations = analyzer.calculate_signal_correlations(signal_arrays)

        assert isinstance(correlations, dict)
        assert ("signal1", "signal2") in correlations
        assert abs(correlations[("signal1", "signal2")] - 1.0) < 0.01

    @pytest.mark.unit
    def test_calculate_signal_correlations_negative(self, analyzer):
        """Test calculating negative correlations."""
        signal_arrays = {
            "signal1": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            "signal2": np.array([5.0, 4.0, 3.0, 2.0, 1.0]),  # Negative correlation
        }

        correlations = analyzer.calculate_signal_correlations(signal_arrays)

        assert isinstance(correlations, dict)
        assert ("signal1", "signal2") in correlations
        assert correlations[("signal1", "signal2")] < 0

    @pytest.mark.unit
    def test_calculate_signal_correlations_three_signals(self, analyzer):
        """Test calculating correlations for three signals."""
        signal_arrays = {
            "signal1": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            "signal2": np.array([2.0, 4.0, 6.0, 8.0, 10.0]),
            "signal3": np.array([1.0, 1.0, 1.0, 1.0, 1.0]),  # Constant (no correlation)
        }

        correlations = analyzer.calculate_signal_correlations(signal_arrays)

        assert isinstance(correlations, dict)
        assert len(correlations) >= 2  # At least signal1-signal2 and signal1-signal3

    @pytest.mark.unit
    def test_calculate_signal_correlations_with_nan(self, analyzer):
        """Test calculating correlations with NaN values."""
        signal_arrays = {
            "signal1": np.array([1.0, 2.0, np.nan, 4.0, 5.0]),
            "signal2": np.array([2.0, 4.0, 6.0, 8.0, 10.0]),
        }

        correlations = analyzer.calculate_signal_correlations(signal_arrays)

        assert isinstance(correlations, dict)
        # Should handle NaN values gracefully

    @pytest.mark.unit
    def test_calculate_signal_correlations_insufficient_samples(self, analyzer):
        """Test calculating correlations with insufficient samples."""
        signal_arrays = {
            "signal1": np.array([1.0, 2.0]),
            "signal2": np.array([2.0, 4.0]),
        }

        correlations = analyzer.calculate_signal_correlations(signal_arrays, min_samples=10)

        assert isinstance(correlations, dict)
        assert len(correlations) == 0  # Should return empty due to insufficient samples

    @pytest.mark.unit
    def test_calculate_correlation_matrix(self, analyzer):
        """Test calculating correlation matrix."""
        signal_arrays = {
            "signal1": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            "signal2": np.array([2.0, 4.0, 6.0, 8.0, 10.0]),
        }

        matrix, signal_names = analyzer.calculate_correlation_matrix(signal_arrays)

        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (2, 2)
        assert len(signal_names) == 2
        assert np.allclose(matrix[0, 0], 1.0)  # Diagonal should be 1.0
        assert np.allclose(matrix[1, 1], 1.0)

    @pytest.mark.unit
    def test_calculate_correlation_matrix_empty(self, analyzer):
        """Test calculating correlation matrix with empty signal arrays."""
        signal_arrays = {}

        matrix, signal_names = analyzer.calculate_correlation_matrix(signal_arrays)

        assert len(matrix) == 0
        assert len(signal_names) == 0

    @pytest.mark.unit
    def test_calculate_spatial_autocorrelation(self, analyzer):
        """Test calculating spatial autocorrelation."""
        signal_array = np.random.rand(10, 10, 10)

        autocorr = analyzer.calculate_spatial_autocorrelation(signal_array, "test_signal")

        assert isinstance(autocorr, float)
        assert -1.0 <= autocorr <= 1.0

    @pytest.mark.unit
    def test_calculate_temporal_autocorrelation(self, analyzer):
        """Test calculating temporal autocorrelation."""
        signal_array = np.random.rand(10, 10, 10)

        autocorr = analyzer.calculate_temporal_autocorrelation(signal_array, "test_signal")

        assert isinstance(autocorr, float)
        assert -1.0 <= autocorr <= 1.0

    @pytest.mark.unit
    def test_analyze_correlations(self, analyzer):
        """Test comprehensive correlation analysis."""

        class MockVoxelData:
            def __init__(self):
                self.available_signals = ["signal1", "signal2"]

            def get_signal_array(self, signal_name, default=0.0):
                if signal_name == "signal1":
                    return np.array([1.0, 2.0, 3.0, 4.0, 5.0])
                elif signal_name == "signal2":
                    return np.array([2.0, 4.0, 6.0, 8.0, 10.0])
                return np.array([0.0])

        voxel_data = MockVoxelData()
        result = analyzer.analyze_correlations(voxel_data, ["signal1", "signal2"])

        assert isinstance(result, CorrelationResults)
        assert len(result.signal_correlations) > 0
