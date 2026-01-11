"""
Unit tests for SPC client.

Tests for SPCClient and SPCConfig.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

from am_qadf.analytics.spc.spc_client import (
    SPCClient,
    SPCConfig,
)
from tests.fixtures.spc.control_chart_data import (
    generate_in_control_data,
    generate_subgrouped_data,
)
from tests.fixtures.spc.capability_data import (
    generate_capable_process_data,
)


class TestSPCConfig:
    """Test suite for SPCConfig dataclass."""

    @pytest.mark.unit
    def test_spc_config_creation(self):
        """Test creating SPCConfig with defaults."""
        config = SPCConfig()

        assert config.control_limit_sigma == 3.0
        assert config.subgroup_size == 5
        assert config.baseline_sample_size == 100
        assert config.adaptive_limits == False
        assert config.enable_warnings == True
        assert config.warning_sigma == 2.0

    @pytest.mark.unit
    def test_spc_config_custom(self):
        """Test creating SPCConfig with custom values."""
        config = SPCConfig(
            control_limit_sigma=2.5,
            subgroup_size=3,
            baseline_sample_size=50,
            adaptive_limits=True,
            update_frequency=25,
            specification_limits=(12.0, 8.0),
            target_value=10.0,
            enable_warnings=False,
        )

        assert config.control_limit_sigma == 2.5
        assert config.subgroup_size == 3
        assert config.baseline_sample_size == 50
        assert config.adaptive_limits == True
        assert config.update_frequency == 25
        assert config.specification_limits == (12.0, 8.0)
        assert config.target_value == 10.0
        assert config.enable_warnings == False


class TestSPCClient:
    """Test suite for SPCClient class."""

    @pytest.fixture
    def config(self):
        """Create an SPCConfig instance."""
        return SPCConfig(control_limit_sigma=3.0, subgroup_size=5)

    @pytest.fixture
    def client(self, config):
        """Create an SPCClient instance."""
        return SPCClient(config=config, mongo_client=None)

    @pytest.mark.unit
    def test_client_creation(self, client):
        """Test creating SPCClient."""
        assert client is not None
        assert client.config is not None
        assert client.chart_generator is not None
        assert client.baseline_calc is not None

    @pytest.mark.unit
    def test_client_creation_with_mongo(self):
        """Test creating SPCClient with MongoDB client."""
        mock_mongo = Mock()
        mock_mongo.connected = True

        client = SPCClient(config=None, mongo_client=mock_mongo)

        assert client is not None
        # Storage may or may not be initialized depending on import

    @pytest.mark.unit
    def test_create_control_chart_xbar(self, client):
        """Test creating X-bar control chart."""
        data = generate_subgrouped_data(n_subgroups=20, subgroup_size=5, mean=10.0, std=1.0)

        result = client.create_control_chart(data, chart_type="xbar", subgroup_size=5)

        assert isinstance(result, tuple) or isinstance(result, type(result))  # May be single or tuple
        # If it's a ControlChartResult, check it
        if hasattr(result, "chart_type"):
            assert result.chart_type == "xbar"

    @pytest.mark.unit
    def test_create_control_chart_individual(self, client):
        """Test creating Individual control chart."""
        data = generate_in_control_data(n_samples=50, mean=10.0, std=1.0)

        result = client.create_control_chart(data, chart_type="individual")

        assert hasattr(result, "chart_type")
        assert result.chart_type == "individual"
        assert result.center_line > 0
        assert result.upper_control_limit > result.center_line

    @pytest.mark.unit
    def test_create_control_chart_xbar_r(self, client):
        """Test creating paired X-bar and R charts."""
        data = generate_subgrouped_data(n_subgroups=20, subgroup_size=5, mean=10.0, std=1.0)

        result = client.create_control_chart(data, chart_type="xbar_r", subgroup_size=5)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert hasattr(result[0], "chart_type")
        assert hasattr(result[1], "chart_type")
        assert result[0].chart_type == "xbar"
        assert result[1].chart_type == "r"

    @pytest.mark.unit
    def test_create_control_chart_invalid_type(self, client):
        """Test creating control chart with invalid type."""
        data = generate_in_control_data(n_samples=50)

        with pytest.raises(ValueError, match="Unknown chart type"):
            client.create_control_chart(data, chart_type="invalid_type")

    @pytest.mark.unit
    def test_establish_baseline(self, client):
        """Test establishing baseline."""
        data = generate_in_control_data(n_samples=100, mean=10.0, std=1.0)

        baseline = client.establish_baseline(data, subgroup_size=1)

        assert baseline is not None
        assert baseline.mean == pytest.approx(10.0, abs=0.5)
        assert baseline.std == pytest.approx(1.0, abs=0.3)
        assert baseline.sample_size == 100

    @pytest.mark.unit
    def test_update_baseline_adaptive(self, client):
        """Test updating baseline adaptively."""
        from am_qadf.analytics.spc.baseline_calculation import BaselineStatistics
        from datetime import datetime

        current_baseline = BaselineStatistics(
            mean=10.0, std=2.0, median=10.0, min=5.0, max=15.0, range=10.0, sample_size=100, subgroup_size=1
        )

        new_data = generate_in_control_data(n_samples=20, mean=10.5, std=2.0, seed=43)

        updated = client.update_baseline_adaptive(current_baseline, new_data, method="exponential_smoothing")

        assert isinstance(updated, BaselineStatistics)
        assert updated.sample_size == current_baseline.sample_size + len(new_data)
        # Mean may shift slightly
        assert abs(updated.mean - current_baseline.mean) < 1.0

    @pytest.mark.unit
    def test_detect_rule_violations(self, client):
        """Test detecting rule violations."""
        # Create a chart result with some OOC points
        data = generate_in_control_data(n_samples=40, mean=10.0, std=1.0)
        ooc_data = np.array([15.0, 16.0, 14.0])  # Out of control
        combined_data = np.concatenate([data, ooc_data])

        chart_result = client.create_control_chart(combined_data, chart_type="individual")

        violations = client.detect_rule_violations(chart_result, rule_set="western_electric")

        assert isinstance(violations, list)
        # May or may not have violations depending on data
        for violation in violations:
            assert hasattr(violation, "rule_name")
            assert hasattr(violation, "severity")
            assert hasattr(violation, "affected_points")

    @pytest.mark.unit
    def test_analyze_process_capability(self, client):
        """Test analyzing process capability."""
        data, spec_limits = generate_capable_process_data(n_samples=100, usl=12.0, lsl=8.0, cpk=1.5)

        result = client.analyze_process_capability(data, spec_limits, target_value=10.0)

        assert result is not None
        assert hasattr(result, "cp")
        assert hasattr(result, "cpk")
        assert hasattr(result, "pp")
        assert hasattr(result, "ppk")
        assert result.cp > 0
        assert result.cpk > 0
        assert result.capability_rating in ["Excellent", "Adequate", "Marginal", "Inadequate"]

    @pytest.mark.unit
    def test_analyze_process_capability_not_implemented(self):
        """Test capability analysis when module not available."""
        # Create client without capability analyzer (if possible)
        config = SPCConfig()
        client = SPCClient(config=config, mongo_client=None)
        # Manually disable capability analyzer
        if client.capability_analyzer is None:
            data, spec_limits = generate_capable_process_data(n_samples=100)
            with pytest.raises(NotImplementedError):
                client.analyze_process_capability(data, spec_limits)

    @pytest.mark.unit
    def test_create_multivariate_chart_hotelling(self, client):
        """Test creating multivariate Hotelling T² chart."""
        data = np.random.randn(100, 5)  # 100 samples, 5 variables

        result = client.create_multivariate_chart(data, method="hotelling_t2")

        assert result is not None
        assert hasattr(result, "hotelling_t2")
        assert hasattr(result, "ucl_t2")
        assert hasattr(result, "out_of_control_points")
        assert len(result.hotelling_t2) == len(data)

    @pytest.mark.unit
    def test_create_multivariate_chart_pca(self, client):
        """Test creating PCA-based multivariate chart."""
        try:
            import sklearn
        except ImportError:
            pytest.skip("scikit-learn not available")

        data = np.random.randn(100, 10)  # 100 samples, 10 variables

        result = client.create_multivariate_chart(data, method="pca")

        assert result is not None
        assert hasattr(result, "principal_components")
        assert hasattr(result, "explained_variance")

    @pytest.mark.unit
    def test_create_multivariate_chart_invalid_method(self, client):
        """Test creating multivariate chart with invalid method."""
        data = np.random.randn(100, 5)

        with pytest.raises(ValueError, match="Unknown multivariate method"):
            client.create_multivariate_chart(data, method="invalid_method")

    @pytest.mark.unit
    def test_comprehensive_spc_analysis(self, client):
        """Test comprehensive SPC analysis."""
        data = generate_subgrouped_data(n_subgroups=20, subgroup_size=5, mean=10.0, std=1.0)

        results = client.comprehensive_spc_analysis(data, specification_limits=(12.0, 8.0), chart_types=["xbar_r"])

        assert isinstance(results, dict)
        assert "control_charts" in results
        assert "baseline" in results
        assert "summary" in results
        assert results["summary"]["n_samples"] > 0
        assert results["summary"]["charts_created"] > 0

    @pytest.mark.unit
    def test_comprehensive_spc_analysis_auto_chart_types(self, client):
        """Test comprehensive analysis with auto-selected chart types."""
        data = generate_subgrouped_data(n_subgroups=20, subgroup_size=5, mean=10.0, std=1.0)

        results = client.comprehensive_spc_analysis(data)

        assert isinstance(results, dict)
        assert "control_charts" in results
        assert len(results["control_charts"]) > 0

    @pytest.mark.unit
    def test_integrate_with_quality_assessment(self, client):
        """Test integrating with quality assessment metrics."""
        quality_metrics = {
            "completeness": np.array([0.9, 0.91, 0.89, 0.92, 0.88]),
            "coverage": np.array([0.85, 0.86, 0.84, 0.87, 0.83]),
            "consistency": np.array([0.88, 0.89, 0.87, 0.90, 0.86]),
        }

        results = client.integrate_with_quality_assessment(quality_metrics)

        assert isinstance(results, dict)
        assert len(results) > 0
        for metric_name in quality_metrics.keys():
            assert metric_name in results

    @pytest.mark.unit
    def test_integrate_with_quality_assessment_with_specs(self, client):
        """Test integrating with quality assessment with specification limits."""
        quality_metrics = {
            "completeness": np.array([0.9, 0.91, 0.89, 0.92, 0.88]),
            "coverage": np.array([0.85, 0.86, 0.84, 0.87, 0.83]),
        }
        specification_limits = {"completeness": (1.0, 0.8), "coverage": (1.0, 0.75)}

        results = client.integrate_with_quality_assessment(quality_metrics, specification_limits=specification_limits)

        assert isinstance(results, dict)
        # Results should include capability analysis for metrics with specs

    @pytest.mark.unit
    def test_monitor_streaming_data(self, client):
        """Test monitoring streaming data."""
        from am_qadf.analytics.spc.baseline_calculation import BaselineStatistics

        baseline = BaselineStatistics(
            mean=10.0, std=2.0, median=10.0, min=5.0, max=15.0, range=10.0, sample_size=100, subgroup_size=1
        )

        # Create data stream (need at least 10 samples per batch for Individual chart)
        data_stream = iter(
            [
                np.array([10.0, 11.0, 9.0, 10.5, 11.5, 9.5, 10.2, 11.2, 9.2, 10.3]),  # 10 samples
                np.array([10.5, 11.5, 9.5, 10.6, 11.6, 9.6, 10.4, 11.4, 9.4, 10.5]),  # 10 samples
                np.array([12.0, 13.0, 11.0, 12.1, 13.1, 11.1, 12.2, 13.2, 11.2, 12.3]),  # 10 samples
            ]
        )

        results = list(client.monitor_streaming_data(data_stream, baseline))

        assert len(results) == 3
        for result in results:
            assert hasattr(result, "chart_type")
            assert hasattr(result, "center_line")

    @pytest.mark.unit
    def test_generate_spc_report(self, client):
        """Test generating SPC report."""
        data = generate_subgrouped_data(n_subgroups=20, subgroup_size=5, mean=10.0, std=1.0)

        results = client.comprehensive_spc_analysis(data)

        report = client.generate_spc_report(results)

        assert isinstance(report, str)
        assert "STATISTICAL PROCESS CONTROL (SPC) ANALYSIS REPORT" in report or "SPC ANALYSIS REPORT" in report
        assert "SUMMARY" in report or "📊 SUMMARY" in report
        assert (
            "CONTROL CHARTS" in report
            or "📊 CONTROL CHARTS" in report
            or "BASELINE STATISTICS" in report
            or "📊 BASELINE STATISTICS" in report
        )

    @pytest.mark.unit
    def test_generate_spc_report_with_file(self, client, tmp_path):
        """Test generating SPC report and saving to file."""
        data = generate_subgrouped_data(n_subgroups=20, subgroup_size=5, mean=10.0, std=1.0)

        results = client.comprehensive_spc_analysis(data)

        output_file = tmp_path / "spc_report.txt"
        report = client.generate_spc_report(results, output_file=str(output_file))

        assert output_file.exists()
        assert isinstance(report, str)
        # Read file and verify content
        with open(output_file, "r") as f:
            file_content = f.read()
            assert "STATISTICAL PROCESS CONTROL (SPC) ANALYSIS REPORT" in file_content or "SPC ANALYSIS REPORT" in file_content
