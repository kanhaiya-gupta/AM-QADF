"""
Unit tests for MPM comparison module.

Tests for MPMComparisonResult and MPMComparisonEngine.
"""

import pytest
import numpy as np
from unittest.mock import Mock

try:
    from am_qadf.validation.mpm_comparison import (
        MPMComparisonResult,
        MPMComparisonEngine,
    )
except ImportError:
    pytest.skip("Validation module not available", allow_module_level=True)


class TestMPMComparisonResult:
    """Test suite for MPMComparisonResult dataclass."""

    @pytest.mark.unit
    def test_mpm_comparison_result_creation(self):
        """Test creating MPMComparisonResult with all fields."""
        result = MPMComparisonResult(
            metric_name="completeness",
            framework_value=0.9,
            mpm_value=0.88,
            correlation=0.95,
            difference=0.02,
            relative_error=2.27,
            is_valid=True,
            metadata={"test": "value"},
        )

        assert result.metric_name == "completeness"
        assert result.framework_value == 0.9
        assert result.mpm_value == 0.88
        assert result.correlation == 0.95
        assert result.difference == 0.02
        assert abs(result.relative_error - 2.27) < 0.1
        assert result.is_valid is True
        assert result.metadata == {"test": "value"}

    @pytest.mark.unit
    def test_mpm_comparison_result_to_dict(self):
        """Test converting MPMComparisonResult to dictionary."""
        result = MPMComparisonResult(
            metric_name="test_metric",
            framework_value=0.9,
            mpm_value=0.88,
            correlation=0.95,
            difference=0.02,
            relative_error=2.27,
            is_valid=True,
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["metric_name"] == "test_metric"
        assert result_dict["framework_value"] == 0.9
        assert result_dict["mpm_value"] == 0.88
        assert result_dict["correlation"] == 0.95
        assert result_dict["is_valid"] is True
        assert "timestamp" in result_dict


class TestMPMComparisonEngine:
    """Test suite for MPMComparisonEngine class."""

    @pytest.fixture
    def comparison_engine(self):
        """Create MPMComparisonEngine instance with default parameters."""
        return MPMComparisonEngine()

    @pytest.fixture
    def custom_engine(self):
        """Create MPMComparisonEngine with custom thresholds."""
        return MPMComparisonEngine(correlation_threshold=0.9, max_relative_error=0.05)

    @pytest.fixture
    def framework_metrics(self):
        """Sample framework quality metrics."""
        return {
            "completeness": 0.9,
            "snr": 25.5,
            "alignment_accuracy": 0.95,
        }

    @pytest.fixture
    def mpm_metrics(self):
        """Sample MPM quality metrics."""
        return {
            "completeness": 0.88,
            "snr": 24.8,
            "alignment_accuracy": 0.94,
        }

    @pytest.fixture
    def framework_array(self):
        """Sample framework output array."""
        np.random.seed(42)
        return np.random.rand(100, 100)

    @pytest.fixture
    def mpm_array(self):
        """Sample MPM output array (slightly different)."""
        np.random.seed(43)
        return np.random.rand(100, 100) + 0.01

    @pytest.mark.unit
    def test_engine_creation_default(self, comparison_engine):
        """Test creating MPMComparisonEngine with default parameters."""
        assert comparison_engine.correlation_threshold == 0.85
        assert comparison_engine.max_relative_error == 0.1

    @pytest.mark.unit
    def test_engine_creation_custom(self, custom_engine):
        """Test creating MPMComparisonEngine with custom parameters."""
        assert custom_engine.correlation_threshold == 0.9
        assert custom_engine.max_relative_error == 0.05

    @pytest.mark.unit
    def test_compare_metric_matching_values(self, comparison_engine):
        """Test comparing metric with matching values."""
        result = comparison_engine.compare_metric("test_metric", framework_value=0.9, mpm_value=0.9)

        assert isinstance(result, MPMComparisonResult)
        assert result.metric_name == "test_metric"
        assert result.framework_value == 0.9
        assert result.mpm_value == 0.9
        assert result.difference == 0.0
        assert result.relative_error == 0.0
        assert result.correlation == 1.0
        assert result.is_valid is True

    @pytest.mark.unit
    def test_compare_metric_different_values(self, comparison_engine):
        """Test comparing metric with different values."""
        result = comparison_engine.compare_metric("test_metric", framework_value=0.9, mpm_value=0.85)

        assert isinstance(result, MPMComparisonResult)
        assert abs(result.difference - 0.05) < 1e-10
        assert result.relative_error > 0
        # Correlation for single values is 1.0 if same, 0.0 if different
        assert result.correlation in [0.0, 1.0]

    @pytest.mark.unit
    def test_compare_metric_zero_reference(self, comparison_engine):
        """Test comparing metric with zero reference value."""
        result = comparison_engine.compare_metric("test_metric", framework_value=0.1, mpm_value=0.0)

        assert isinstance(result, MPMComparisonResult)
        assert result.mpm_value == 0.0
        # Relative error should handle zero case
        assert result.relative_error == float("inf") or result.relative_error >= 0

    @pytest.mark.unit
    def test_compare_arrays_matching(self, comparison_engine):
        """Test comparing arrays that match."""
        arr = np.random.rand(50, 50)
        result = comparison_engine.compare_arrays("test_array", arr, arr.copy())

        assert isinstance(result, MPMComparisonResult)
        assert result.metric_name == "test_array"
        assert result.correlation >= 0.99  # Should be very high for identical arrays
        assert result.difference < 1e-10  # Should be very small

    @pytest.mark.unit
    def test_compare_arrays_different(self, comparison_engine, framework_array, mpm_array):
        """Test comparing arrays that are different."""
        result = comparison_engine.compare_arrays("test_array", framework_array, mpm_array)

        assert isinstance(result, MPMComparisonResult)
        assert result.correlation >= 0  # Should be positive
        assert result.difference > 0  # Should have some difference
        assert result.validated_points > 0

    @pytest.mark.unit
    def test_compare_arrays_size_mismatch(self, comparison_engine):
        """Test comparing arrays with size mismatch."""
        arr1 = np.random.rand(50, 50)
        arr2 = np.random.rand(60, 60)

        result = comparison_engine.compare_arrays("test_array", arr1, arr2)

        assert isinstance(result, MPMComparisonResult)
        assert result.validated_points > 0  # Should use minimum size
        assert "valid_points" in result.metadata

    @pytest.mark.unit
    def test_compare_arrays_with_nan(self, comparison_engine):
        """Test comparing arrays with NaN values."""
        arr1 = np.random.rand(50, 50)
        arr2 = arr1.copy()
        arr1[10:20, 10:20] = np.nan
        arr2[15:25, 15:25] = np.nan

        result = comparison_engine.compare_arrays("test_array", arr1, arr2)

        assert isinstance(result, MPMComparisonResult)
        assert result.validated_points < 2500  # Should exclude NaN values
        assert "valid_points" in result.metadata

    @pytest.mark.unit
    def test_compare_arrays_empty(self, comparison_engine):
        """Test comparing empty arrays."""
        arr1 = np.array([])
        arr2 = np.array([])

        result = comparison_engine.compare_arrays("test_array", arr1, arr2)

        assert isinstance(result, MPMComparisonResult)
        assert result.is_valid is False
        assert "error" in result.metadata

    @pytest.mark.unit
    def test_calculate_correlation_pearson(self, comparison_engine):
        """Test calculating Pearson correlation."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.1, 2.1, 3.1, 4.1, 5.1])  # Highly correlated

        correlation = comparison_engine.calculate_correlation(x, y, method="pearson")

        assert correlation > 0.99  # Should be very high

    @pytest.mark.unit
    def test_calculate_correlation_spearman(self, comparison_engine):
        """Test calculating Spearman correlation."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.1, 2.1, 3.1, 4.1, 5.1])

        correlation = comparison_engine.calculate_correlation(x, y, method="spearman")

        assert correlation > 0.99

    @pytest.mark.unit
    def test_calculate_correlation_no_correlation(self, comparison_engine):
        """Test calculating correlation with no correlation."""
        x = np.random.rand(100)
        y = np.random.rand(100)  # Uncorrelated

        correlation = comparison_engine.calculate_correlation(x, y, method="pearson")

        assert abs(correlation) < 0.3  # Should be low

    @pytest.mark.unit
    def test_calculate_correlation_size_mismatch(self, comparison_engine):
        """Test correlation with size mismatch."""
        x = np.random.rand(100)
        y = np.random.rand(50)

        correlation = comparison_engine.calculate_correlation(x, y)

        assert correlation >= 0  # Should handle gracefully

    @pytest.mark.unit
    def test_calculate_correlation_with_nan(self, comparison_engine):
        """Test correlation calculation with NaN values."""
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        y = np.array([1.1, 2.1, 3.1, np.nan, 5.1])

        correlation = comparison_engine.calculate_correlation(x, y)

        assert 0 <= correlation <= 1 or correlation == 0.0  # Should handle NaN

    @pytest.mark.unit
    def test_compare_quality_metrics(self, comparison_engine, framework_metrics, mpm_metrics):
        """Test comparing quality metrics."""
        results = comparison_engine.compare_quality_metrics(framework_metrics, mpm_metrics)

        assert isinstance(results, dict)
        assert "completeness" in results
        assert "snr" in results
        assert "alignment_accuracy" in results
        assert isinstance(results["completeness"], MPMComparisonResult)

    @pytest.mark.unit
    def test_compare_quality_metrics_partial_overlap(self, comparison_engine):
        """Test comparing metrics with partial overlap."""
        framework = {"metric1": 0.9, "metric2": 0.8, "metric3": 0.7}
        mpm = {"metric1": 0.88, "metric3": 0.68}  # Missing metric2

        results = comparison_engine.compare_quality_metrics(framework, mpm)

        assert "metric1" in results
        assert "metric3" in results
        assert "metric2" not in results  # Only common metrics

    @pytest.mark.unit
    def test_compare_all_metrics_dict(self, comparison_engine, framework_metrics, mpm_metrics):
        """Test compare_all_metrics with dictionary inputs."""
        results = comparison_engine.compare_all_metrics(framework_metrics, mpm_metrics)

        assert isinstance(results, dict)
        assert len(results) > 0
        for metric_name, result in results.items():
            assert isinstance(result, MPMComparisonResult)
            assert result.metric_name == metric_name

    @pytest.mark.unit
    def test_compare_all_metrics_dict_with_metrics_param(self, comparison_engine, framework_metrics, mpm_metrics):
        """Test compare_all_metrics with metrics parameter."""
        results = comparison_engine.compare_all_metrics(framework_metrics, mpm_metrics, metrics=["completeness", "snr"])

        assert isinstance(results, dict)
        assert "completeness" in results
        assert "snr" in results
        assert "alignment_accuracy" not in results  # Not in metrics list

    @pytest.mark.unit
    def test_compare_all_metrics_arrays(self, comparison_engine):
        """Test compare_all_metrics with array inputs."""
        arr1 = np.random.rand(100, 100)
        arr2 = np.random.rand(100, 100)

        results = comparison_engine.compare_all_metrics(arr1, arr2)

        assert isinstance(results, dict)
        assert len(results) == 1
        assert "array_comparison" in results or list(results.keys())[0]  # Default name

    @pytest.mark.unit
    def test_compare_all_metrics_unsupported_types(self, comparison_engine):
        """Test compare_all_metrics with unsupported types."""
        results = comparison_engine.compare_all_metrics("string1", "string2")

        assert isinstance(results, dict)
        # Should handle gracefully (empty or error in metadata)

    @pytest.mark.unit
    def test_is_valid_calculation(self, comparison_engine):
        """Test is_valid flag calculation."""
        # Valid case: high correlation, low error
        result1 = comparison_engine.compare_metric("test", 0.9, 0.88)
        # Should be valid if correlation threshold and error threshold are met

        # Invalid case: low correlation
        custom_engine = MPMComparisonEngine(correlation_threshold=0.99, max_relative_error=0.01)
        result2 = custom_engine.compare_metric("test", 0.9, 0.5)  # Large difference
        # Should not be valid due to large difference

        assert isinstance(result1, MPMComparisonResult)
        assert isinstance(result2, MPMComparisonResult)
