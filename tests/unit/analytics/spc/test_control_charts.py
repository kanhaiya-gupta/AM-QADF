"""
Unit tests for control charts.

Tests for ControlChartGenerator and all chart types (X-bar, R, S, Individual, Moving Range).
"""

import pytest
import numpy as np

from am_qadf.analytics.spc.control_charts import (
    ControlChartGenerator,
    ControlChartResult,
    XBarChart,
    RChart,
    SChart,
    IndividualChart,
    MovingRangeChart,
)
from am_qadf.analytics.spc.spc_client import SPCConfig
from tests.fixtures.spc.control_chart_data import (
    generate_in_control_data,
    generate_out_of_control_data,
    generate_trend_data,
    generate_subgrouped_data,
    generate_xbar_r_chart_data,
    generate_individual_mr_data,
)


class TestControlChartResult:
    """Test suite for ControlChartResult dataclass."""

    @pytest.mark.unit
    def test_control_chart_result_creation(self):
        """Test creating ControlChartResult."""
        result = ControlChartResult(
            chart_type="xbar",
            center_line=10.0,
            upper_control_limit=13.0,
            lower_control_limit=7.0,
            upper_warning_limit=12.0,
            lower_warning_limit=8.0,
            sample_values=np.array([10.0, 11.0, 9.0]),
            sample_indices=np.array([0, 1, 2]),
            out_of_control_points=[1],
            rule_violations={},
            baseline_stats={"mean": 10.0, "std": 1.0},
            metadata={"subgroup_size": 5},
        )

        assert result.chart_type == "xbar"
        assert result.center_line == 10.0
        assert result.upper_control_limit == 13.0
        assert result.lower_control_limit == 7.0
        assert result.upper_warning_limit == 12.0
        assert result.lower_warning_limit == 8.0
        assert len(result.sample_values) == 3
        assert len(result.out_of_control_points) == 1

    @pytest.mark.unit
    def test_control_chart_result_defaults(self):
        """Test ControlChartResult with default values."""
        result = ControlChartResult(
            chart_type="individual",
            center_line=10.0,
            upper_control_limit=13.0,
            lower_control_limit=7.0,
            sample_values=np.array([10.0]),
            sample_indices=np.array([0]),
        )

        assert result.upper_warning_limit is None
        assert result.lower_warning_limit is None
        assert len(result.out_of_control_points) == 0
        assert len(result.rule_violations) == 0


class TestControlChartGenerator:
    """Test suite for ControlChartGenerator class."""

    @pytest.fixture
    def generator(self):
        """Create a ControlChartGenerator instance."""
        return ControlChartGenerator()

    @pytest.fixture
    def config(self):
        """Create an SPCConfig instance."""
        return SPCConfig(control_limit_sigma=3.0, subgroup_size=5, enable_warnings=True, warning_sigma=2.0)

    @pytest.mark.unit
    def test_generator_creation(self, generator):
        """Test creating ControlChartGenerator."""
        assert generator is not None

    @pytest.mark.unit
    def test_create_xbar_chart(self, generator, config):
        """Test creating X-bar chart."""
        data = generate_subgrouped_data(n_subgroups=20, subgroup_size=5, mean=10.0, std=1.0)

        result = generator.create_xbar_chart(data, subgroup_size=5, config=config)

        assert isinstance(result, ControlChartResult)
        assert result.chart_type == "xbar"
        assert result.center_line > 0
        assert result.upper_control_limit > result.center_line
        assert result.lower_control_limit < result.center_line
        assert len(result.sample_values) > 0
        assert result.metadata["subgroup_size"] == 5

    @pytest.mark.unit
    def test_create_xbar_chart_insufficient_data(self, generator, config):
        """Test creating X-bar chart with insufficient data."""
        data = np.array([1.0, 2.0, 3.0])  # Too few samples

        with pytest.raises(ValueError, match="Insufficient data"):
            generator.create_xbar_chart(data, subgroup_size=5, config=config)

    @pytest.mark.unit
    def test_create_r_chart(self, generator, config):
        """Test creating R chart."""
        data = generate_subgrouped_data(n_subgroups=20, subgroup_size=5, mean=10.0, std=1.0)

        result = generator.create_r_chart(data, subgroup_size=5, config=config)

        assert isinstance(result, ControlChartResult)
        assert result.chart_type == "r"
        assert result.center_line >= 0  # Range is always >= 0
        assert result.upper_control_limit > result.center_line
        assert result.lower_control_limit >= 0  # R chart LCL is always >= 0
        assert len(result.sample_values) > 0

    @pytest.mark.unit
    def test_create_s_chart(self, generator, config):
        """Test creating S chart."""
        data = generate_subgrouped_data(n_subgroups=20, subgroup_size=5, mean=10.0, std=1.0)

        result = generator.create_s_chart(data, subgroup_size=5, config=config)

        assert isinstance(result, ControlChartResult)
        assert result.chart_type == "s"
        assert result.center_line >= 0  # Standard deviation is always >= 0
        assert result.upper_control_limit > result.center_line
        assert result.lower_control_limit >= 0  # S chart LCL is always >= 0
        assert len(result.sample_values) > 0

    @pytest.mark.unit
    def test_create_individual_chart(self, generator, config):
        """Test creating Individual chart."""
        data = generate_in_control_data(n_samples=50, mean=10.0, std=1.0)

        result = generator.create_individual_chart(data, config=config)

        assert isinstance(result, ControlChartResult)
        assert result.chart_type == "individual"
        assert result.center_line == pytest.approx(np.mean(data), abs=0.5)
        assert result.upper_control_limit > result.center_line
        assert result.lower_control_limit < result.center_line
        assert result.upper_warning_limit is not None  # Warnings enabled
        assert result.lower_warning_limit is not None
        assert len(result.sample_values) == len(data)

    @pytest.mark.unit
    def test_create_individual_chart_no_warnings(self, generator):
        """Test creating Individual chart without warnings."""
        config_no_warn = SPCConfig(enable_warnings=False)
        data = generate_in_control_data(n_samples=50, mean=10.0, std=1.0)

        result = generator.create_individual_chart(data, config=config_no_warn)

        assert result.upper_warning_limit is None
        assert result.lower_warning_limit is None

    @pytest.mark.unit
    def test_create_individual_chart_insufficient_data(self, generator, config):
        """Test creating Individual chart with insufficient data."""
        data = np.array([1.0, 2.0])  # Too few samples

        with pytest.raises(ValueError, match="Need at least 10 samples"):
            generator.create_individual_chart(data, config=config)

    @pytest.mark.unit
    def test_create_moving_range_chart(self, generator, config):
        """Test creating Moving Range chart."""
        data = generate_in_control_data(n_samples=50, mean=10.0, std=1.0)

        result = generator.create_moving_range_chart(data, window_size=2, config=config)

        assert isinstance(result, ControlChartResult)
        assert result.chart_type == "moving_range"
        assert result.center_line >= 0
        assert result.upper_control_limit > result.center_line
        assert result.lower_control_limit == 0  # MR chart LCL is always 0
        assert len(result.sample_values) == len(data) - 1  # MR has one fewer point

    @pytest.mark.unit
    def test_create_moving_range_chart_insufficient_data(self, generator, config):
        """Test creating Moving Range chart with insufficient data."""
        data = np.array([1.0, 2.0])  # Need at least 3 for window_size=2

        with pytest.raises(ValueError, match="Need at least"):
            generator.create_moving_range_chart(data, window_size=2, config=config)

    @pytest.mark.unit
    def test_create_xbar_r_charts(self, generator, config):
        """Test creating paired X-bar and R charts."""
        data = generate_subgrouped_data(n_subgroups=20, subgroup_size=5, mean=10.0, std=1.0)

        xbar_result, r_result = generator.create_xbar_r_charts(data, subgroup_size=5, config=config)

        assert isinstance(xbar_result, ControlChartResult)
        assert isinstance(r_result, ControlChartResult)
        assert xbar_result.chart_type == "xbar"
        assert r_result.chart_type == "r"
        assert len(xbar_result.sample_values) == len(r_result.sample_values)

    @pytest.mark.unit
    def test_create_xbar_s_charts(self, generator, config):
        """Test creating paired X-bar and S charts."""
        data = generate_subgrouped_data(n_subgroups=20, subgroup_size=5, mean=10.0, std=1.0)

        xbar_result, s_result = generator.create_xbar_s_charts(data, subgroup_size=5, config=config)

        assert isinstance(xbar_result, ControlChartResult)
        assert isinstance(s_result, ControlChartResult)
        assert xbar_result.chart_type == "xbar"
        assert s_result.chart_type == "s"
        assert len(xbar_result.sample_values) == len(s_result.sample_values)

    @pytest.mark.unit
    def test_detect_out_of_control(self, generator):
        """Test detecting out-of-control points."""
        values = np.array([10.0, 11.0, 12.0, 20.0, 10.0, 11.0])  # One OOC point
        ucl = 15.0
        lcl = 5.0

        ooc_indices = generator.detect_out_of_control(values, ucl, lcl)

        assert isinstance(ooc_indices, list)
        assert 3 in ooc_indices  # Point at index 3 is out of control

    @pytest.mark.unit
    def test_detect_out_of_control_no_violations(self, generator):
        """Test detecting out-of-control when all points are in control."""
        values = np.array([10.0, 11.0, 12.0, 10.5, 11.5])
        ucl = 15.0
        lcl = 5.0

        ooc_indices = generator.detect_out_of_control(values, ucl, lcl)

        assert len(ooc_indices) == 0

    @pytest.mark.unit
    def test_detect_out_of_control_nan_values(self, generator):
        """Test detecting out-of-control with NaN values."""
        values = np.array([10.0, np.nan, 12.0, 20.0, np.inf])
        ucl = 15.0
        lcl = 5.0

        ooc_indices = generator.detect_out_of_control(values, ucl, lcl)

        assert 3 in ooc_indices  # Only valid OOC point
        # NaN and inf should not be included

    @pytest.mark.unit
    def test_update_control_limits(self, generator, config):
        """Test updating control limits."""
        # Create initial chart
        initial_data = generate_in_control_data(n_samples=50, mean=10.0, std=1.0)
        chart_result = generator.create_individual_chart(initial_data, config=config)

        original_cl = chart_result.center_line

        # Update with new data
        new_data = generate_in_control_data(n_samples=20, mean=10.5, std=1.0, seed=43)
        updated_result = generator.update_control_limits(chart_result, new_data, method="adaptive")

        assert isinstance(updated_result, ControlChartResult)
        # Center line may shift slightly
        assert abs(updated_result.center_line - original_cl) < 1.0
        # Sample size may or may not increase with exponential smoothing method
        # (exponential smoothing doesn't track cumulative sample size)
        assert updated_result.baseline_stats["sample_size"] >= chart_result.baseline_stats["sample_size"]


class TestConvenienceChartClasses:
    """Test suite for convenience chart classes."""

    @pytest.fixture
    def config(self):
        """Create an SPCConfig instance."""
        return SPCConfig(control_limit_sigma=3.0, subgroup_size=5)

    @pytest.mark.unit
    def test_xbar_chart_class(self, config):
        """Test XBarChart convenience class."""
        chart = XBarChart()
        data = generate_subgrouped_data(n_subgroups=20, subgroup_size=5)

        result = chart.create(data, subgroup_size=5, config=config)

        assert isinstance(result, ControlChartResult)
        assert result.chart_type == "xbar"

    @pytest.mark.unit
    def test_r_chart_class(self, config):
        """Test RChart convenience class."""
        chart = RChart()
        data = generate_subgrouped_data(n_subgroups=20, subgroup_size=5)

        result = chart.create(data, subgroup_size=5, config=config)

        assert isinstance(result, ControlChartResult)
        assert result.chart_type == "r"

    @pytest.mark.unit
    def test_s_chart_class(self, config):
        """Test SChart convenience class."""
        chart = SChart()
        data = generate_subgrouped_data(n_subgroups=20, subgroup_size=5)

        result = chart.create(data, subgroup_size=5, config=config)

        assert isinstance(result, ControlChartResult)
        assert result.chart_type == "s"

    @pytest.mark.unit
    def test_individual_chart_class(self, config):
        """Test IndividualChart convenience class."""
        chart = IndividualChart()
        data = generate_in_control_data(n_samples=50)

        result = chart.create(data, config=config)

        assert isinstance(result, ControlChartResult)
        assert result.chart_type == "individual"

    @pytest.mark.unit
    def test_moving_range_chart_class(self, config):
        """Test MovingRangeChart convenience class."""
        chart = MovingRangeChart()
        data = generate_in_control_data(n_samples=50)

        result = chart.create(data, window_size=2, config=config)

        assert isinstance(result, ControlChartResult)
        assert result.chart_type == "moving_range"
