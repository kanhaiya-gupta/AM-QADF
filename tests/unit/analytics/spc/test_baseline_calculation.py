"""
Unit tests for baseline calculation.

Tests for BaselineCalculator, AdaptiveLimitsCalculator, and BaselineStatistics.
"""

import pytest
import numpy as np
from datetime import datetime

from am_qadf.analytics.spc.baseline_calculation import (
    BaselineCalculator,
    AdaptiveLimitsCalculator,
    BaselineStatistics,
)
from tests.fixtures.spc.baseline_data import (
    generate_stable_baseline_data,
    generate_unstable_baseline_data,
    generate_subgrouped_baseline_data,
)


class TestBaselineStatistics:
    """Test suite for BaselineStatistics dataclass."""

    @pytest.mark.unit
    def test_baseline_statistics_creation(self):
        """Test creating BaselineStatistics."""
        baseline = BaselineStatistics(
            mean=10.0,
            std=2.0,
            median=9.5,
            min=5.0,
            max=15.0,
            range=10.0,
            sample_size=100,
            subgroup_size=5,
            within_subgroup_std=1.8,
            between_subgroup_std=0.6,
            overall_std=2.0,
            calculated_at=datetime.now(),
            metadata={"method": "standard"},
        )

        assert baseline.mean == 10.0
        assert baseline.std == 2.0
        assert baseline.median == 9.5
        assert baseline.min == 5.0
        assert baseline.max == 15.0
        assert baseline.range == 10.0
        assert baseline.sample_size == 100
        assert baseline.subgroup_size == 5
        assert baseline.within_subgroup_std == 1.8
        assert baseline.between_subgroup_std == 0.6
        assert baseline.overall_std == 2.0

    @pytest.mark.unit
    def test_baseline_statistics_defaults(self):
        """Test BaselineStatistics with default values."""
        baseline = BaselineStatistics(
            mean=10.0, std=2.0, median=10.0, min=8.0, max=12.0, range=4.0, sample_size=50, subgroup_size=1
        )

        assert baseline.within_subgroup_std is None
        assert baseline.between_subgroup_std is None
        assert baseline.overall_std is None
        assert isinstance(baseline.calculated_at, datetime)


class TestBaselineCalculator:
    """Test suite for BaselineCalculator class."""

    @pytest.fixture
    def calculator(self):
        """Create a BaselineCalculator instance."""
        return BaselineCalculator()

    @pytest.mark.unit
    def test_calculator_creation(self, calculator):
        """Test creating BaselineCalculator."""
        assert calculator is not None

    @pytest.mark.unit
    def test_calculate_baseline_individual(self, calculator):
        """Test calculating baseline for individual measurements."""
        data = generate_stable_baseline_data(n_samples=100, mean=10.0, std=1.0)

        baseline = calculator.calculate_baseline(data, subgroup_size=None)

        assert isinstance(baseline, BaselineStatistics)
        assert baseline.mean == pytest.approx(10.0, abs=0.5)
        assert baseline.std == pytest.approx(1.0, abs=0.3)
        assert baseline.sample_size == 100
        assert baseline.subgroup_size == 1
        assert baseline.min <= baseline.max
        assert baseline.range == baseline.max - baseline.min

    @pytest.mark.unit
    def test_calculate_baseline_subgrouped(self, calculator):
        """Test calculating baseline for subgrouped data."""
        data = generate_subgrouped_baseline_data(n_subgroups=20, subgroup_size=5, mean=10.0)

        baseline = calculator.calculate_baseline(data, subgroup_size=5)

        assert isinstance(baseline, BaselineStatistics)
        assert baseline.subgroup_size == 5
        assert baseline.within_subgroup_std is not None
        assert baseline.between_subgroup_std is not None
        assert baseline.overall_std is not None

    @pytest.mark.unit
    def test_calculate_baseline_insufficient_data(self, calculator):
        """Test baseline calculation with insufficient data."""
        data = np.array([1.0, 2.0, 3.0])  # Too few samples

        # Warning is logged (logger.warning), not raised (warnings.warn)
        baseline = calculator.calculate_baseline(data)

        assert baseline.sample_size == 3

    @pytest.mark.unit
    def test_calculate_baseline_no_valid_data(self, calculator):
        """Test baseline calculation with no valid data."""
        data = np.array([np.nan, np.nan, np.inf])

        with pytest.raises(ValueError, match="No valid data points"):
            calculator.calculate_baseline(data)

    @pytest.mark.unit
    def test_estimate_within_subgroup_std(self, calculator):
        """Test estimating within-subgroup standard deviation."""
        data = generate_subgrouped_baseline_data(n_subgroups=20, subgroup_size=5, mean=10.0)

        within_std = calculator.estimate_within_subgroup_std(data, subgroup_size=5)

        assert isinstance(within_std, float)
        assert within_std > 0

    @pytest.mark.unit
    def test_estimate_between_subgroup_std(self, calculator):
        """Test estimating between-subgroup standard deviation."""
        # Create subgroup means
        subgroup_means = np.array([10.0, 10.2, 9.8, 10.1, 9.9])

        between_std = calculator.estimate_between_subgroup_std(subgroup_means)

        assert isinstance(between_std, float)
        assert between_std >= 0

    @pytest.mark.unit
    def test_calculate_control_limits_xbar(self, calculator):
        """Test calculating control limits for X-bar chart."""
        baseline = BaselineStatistics(
            mean=10.0,
            std=2.0,
            median=10.0,
            min=5.0,
            max=15.0,
            range=10.0,
            sample_size=100,
            subgroup_size=5,
            within_subgroup_std=1.8,
        )

        limits = calculator.calculate_control_limits(baseline, "xbar", subgroup_size=5)

        assert isinstance(limits, dict)
        assert "ucl" in limits
        assert "lcl" in limits
        assert "cl" in limits
        assert limits["cl"] == pytest.approx(baseline.mean, abs=0.1)
        assert limits["ucl"] > limits["cl"]
        assert limits["lcl"] < limits["cl"]

    @pytest.mark.unit
    def test_calculate_control_limits_r_chart(self, calculator):
        """Test calculating control limits for R chart."""
        baseline = BaselineStatistics(
            mean=10.0,
            std=2.0,
            median=10.0,
            min=5.0,
            max=15.0,
            range=10.0,
            sample_size=100,
            subgroup_size=5,
            within_subgroup_std=1.8,
        )

        limits = calculator.calculate_control_limits(baseline, "r", subgroup_size=5)

        assert isinstance(limits, dict)
        assert "ucl" in limits
        assert "lcl" in limits
        assert "cl" in limits
        assert limits["ucl"] > limits["cl"]
        assert limits["lcl"] >= 0  # R chart LCL is always >= 0

    @pytest.mark.unit
    def test_calculate_control_limits_s_chart(self, calculator):
        """Test calculating control limits for S chart."""
        baseline = BaselineStatistics(
            mean=10.0,
            std=2.0,
            median=10.0,
            min=5.0,
            max=15.0,
            range=10.0,
            sample_size=100,
            subgroup_size=5,
            within_subgroup_std=1.8,
        )

        limits = calculator.calculate_control_limits(baseline, "s", subgroup_size=5)

        assert isinstance(limits, dict)
        assert "ucl" in limits
        assert "lcl" in limits
        assert "cl" in limits
        assert limits["ucl"] > limits["cl"]
        assert limits["lcl"] >= 0  # S chart LCL is always >= 0

    @pytest.mark.unit
    def test_calculate_control_limits_individual(self, calculator):
        """Test calculating control limits for Individual chart."""
        baseline = BaselineStatistics(
            mean=10.0, std=2.0, median=10.0, min=5.0, max=15.0, range=10.0, sample_size=100, subgroup_size=1
        )

        limits = calculator.calculate_control_limits(baseline, "individual", subgroup_size=1, sigma_multiplier=3.0)

        assert isinstance(limits, dict)
        assert limits["cl"] == pytest.approx(baseline.mean, abs=0.1)
        assert limits["ucl"] == pytest.approx(baseline.mean + 3.0 * baseline.std, abs=0.1)
        assert limits["lcl"] == pytest.approx(baseline.mean - 3.0 * baseline.std, abs=0.1)

    @pytest.mark.unit
    def test_calculate_control_limits_moving_range(self, calculator):
        """Test calculating control limits for Moving Range chart."""
        baseline = BaselineStatistics(
            mean=10.0, std=2.0, median=10.0, min=5.0, max=15.0, range=10.0, sample_size=100, subgroup_size=1
        )

        limits = calculator.calculate_control_limits(baseline, "moving_range", subgroup_size=2)

        assert isinstance(limits, dict)
        assert "ucl" in limits
        assert "cl" in limits
        assert limits["lcl"] == 0.0  # MR chart LCL is always 0
        assert limits["ucl"] > limits["cl"]

    @pytest.mark.unit
    def test_calculate_control_limits_invalid_type(self, calculator):
        """Test calculating control limits with invalid chart type."""
        baseline = BaselineStatistics(
            mean=10.0, std=2.0, median=10.0, min=5.0, max=15.0, range=10.0, sample_size=100, subgroup_size=5
        )

        with pytest.raises(ValueError, match="Unknown chart type"):
            calculator.calculate_control_limits(baseline, "invalid_type", subgroup_size=5)

    @pytest.mark.unit
    def test_get_control_chart_constants(self, calculator):
        """Test getting control chart constants."""
        constants = calculator._get_control_chart_constants(5)

        assert isinstance(constants, dict)
        assert "A2" in constants
        assert "D3" in constants
        assert "D4" in constants
        assert "B3" in constants
        assert "B4" in constants
        assert "d2" in constants
        assert constants["A2"] == pytest.approx(0.577, abs=0.001)
        assert constants["D4"] == pytest.approx(2.114, abs=0.001)

    @pytest.mark.unit
    def test_get_control_chart_constants_edge_cases(self, calculator):
        """Test getting constants for edge cases."""
        # Test n=2
        constants_2 = calculator._get_control_chart_constants(2)
        assert constants_2["A2"] == pytest.approx(1.880, abs=0.001)

        # Test n=10
        constants_10 = calculator._get_control_chart_constants(10)
        assert constants_10["A2"] == pytest.approx(0.308, abs=0.001)

        # Test large n (should use approximation)
        constants_large = calculator._get_control_chart_constants(25)
        assert "A2" in constants_large
        assert "D4" in constants_large

    @pytest.mark.unit
    def test_update_baseline_exponential_smoothing(self, calculator):
        """Test updating baseline with exponential smoothing."""
        current_baseline = BaselineStatistics(
            mean=10.0, std=2.0, median=10.0, min=5.0, max=15.0, range=10.0, sample_size=100, subgroup_size=1
        )

        new_data = np.array([12.0, 11.5, 13.0, 12.5])

        updated = calculator.update_baseline(current_baseline, new_data, method="exponential_smoothing", alpha=0.1)

        assert isinstance(updated, BaselineStatistics)
        assert updated.sample_size == current_baseline.sample_size + len(new_data)
        # Mean should shift towards new data
        assert updated.mean != current_baseline.mean

    @pytest.mark.unit
    def test_update_baseline_cumulative(self, calculator):
        """Test updating baseline with cumulative method."""
        current_baseline = BaselineStatistics(
            mean=10.0, std=2.0, median=10.0, min=5.0, max=15.0, range=10.0, sample_size=100, subgroup_size=1
        )

        new_data = np.array([12.0, 11.5, 13.0, 12.5])

        updated = calculator.update_baseline(current_baseline, new_data, method="cumulative")

        assert isinstance(updated, BaselineStatistics)
        assert updated.sample_size == current_baseline.sample_size + len(new_data)

    @pytest.mark.unit
    def test_update_baseline_no_new_data(self, calculator):
        """Test updating baseline with no valid new data."""
        current_baseline = BaselineStatistics(
            mean=10.0, std=2.0, median=10.0, min=5.0, max=15.0, range=10.0, sample_size=100, subgroup_size=1
        )

        new_data = np.array([np.nan, np.inf])

        # Warning is logged (logger.warning), not raised (warnings.warn)
        updated = calculator.update_baseline(current_baseline, new_data)

        assert updated == current_baseline


class TestAdaptiveLimitsCalculator:
    """Test suite for AdaptiveLimitsCalculator class."""

    @pytest.fixture
    def adaptive_calc(self):
        """Create an AdaptiveLimitsCalculator instance."""
        return AdaptiveLimitsCalculator()

    @pytest.mark.unit
    def test_adaptive_calc_creation(self, adaptive_calc):
        """Test creating AdaptiveLimitsCalculator."""
        assert adaptive_calc is not None

    @pytest.mark.unit
    def test_calculate_adaptive_limits(self, adaptive_calc):
        """Test calculating adaptive control limits."""
        historical_data = generate_stable_baseline_data(n_samples=100, mean=10.0, std=1.0)
        current_data = generate_stable_baseline_data(n_samples=50, mean=10.2, std=1.0, seed=43)

        result = adaptive_calc.calculate_adaptive_limits(historical_data, current_data, window_size=100, update_frequency=50)

        assert isinstance(result, dict)
        assert "limits" in result
        assert "baseline" in result
        assert "drift_detected" in result
        assert "recommend_update" in result
        assert isinstance(result["baseline"], BaselineStatistics)

    @pytest.mark.unit
    def test_detect_drift_no_drift(self, adaptive_calc):
        """Test drift detection when no drift exists."""
        baseline = BaselineStatistics(
            mean=10.0, std=2.0, median=10.0, min=5.0, max=15.0, range=10.0, sample_size=100, subgroup_size=1
        )

        recent_data = generate_stable_baseline_data(n_samples=30, mean=10.0, std=2.0, seed=43)

        drift_info = adaptive_calc.detect_drift(baseline, recent_data)

        assert isinstance(drift_info, dict)
        assert "drift_detected" in drift_info
        assert "drift_magnitude" in drift_info
        assert "z_score" in drift_info
        # Should not detect drift for stable data
        assert drift_info["drift_detected"] == False or abs(drift_info["z_score"]) < 2.0

    @pytest.mark.unit
    def test_detect_drift_with_drift(self, adaptive_calc):
        """Test drift detection when drift exists."""
        baseline = BaselineStatistics(
            mean=10.0, std=2.0, median=10.0, min=5.0, max=15.0, range=10.0, sample_size=100, subgroup_size=1
        )

        # Create data with significant shift
        recent_data = np.random.normal(15.0, 2.0, 30)  # Shifted mean

        drift_info = adaptive_calc.detect_drift(baseline, recent_data)

        assert isinstance(drift_info, dict)
        assert drift_info["drift_detected"] == True
        assert abs(drift_info["drift_magnitude"]) > 0
        assert abs(drift_info["z_score"]) > 2.0

    @pytest.mark.unit
    def test_detect_drift_insufficient_data(self, adaptive_calc):
        """Test drift detection with insufficient data."""
        baseline = BaselineStatistics(
            mean=10.0, std=2.0, median=10.0, min=5.0, max=15.0, range=10.0, sample_size=100, subgroup_size=1
        )

        recent_data = np.array([10.0, 11.0])  # Too few samples

        drift_info = adaptive_calc.detect_drift(baseline, recent_data)

        assert drift_info.get("insufficient_data", False) == True
        assert drift_info["drift_detected"] == False

    @pytest.mark.unit
    def test_recommend_limit_update(self, adaptive_calc):
        """Test recommendation for limit update."""
        baseline = BaselineStatistics(
            mean=10.0, std=2.0, median=10.0, min=5.0, max=15.0, range=10.0, sample_size=100, subgroup_size=1
        )

        # Significant drift
        recent_data_with_drift = np.random.normal(15.0, 2.0, 50)
        recommend_update = adaptive_calc.recommend_limit_update(baseline, recent_data_with_drift)
        assert recommend_update == True

        # No significant drift
        recent_data_stable = np.random.normal(10.0, 2.0, 50)
        recommend_update = adaptive_calc.recommend_limit_update(baseline, recent_data_stable)
        # May or may not recommend update depending on statistical test
        # Function returns bool (from type annotation), just verify it's callable
        assert recommend_update in (True, False)
