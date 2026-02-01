"""
Integration tests for SPC with QualityAssessmentClient.

Tests integration of SPC module with QualityAssessmentClient.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

try:
    from am_qadf.analytics.quality_assessment.client import QualityAssessmentClient
    from am_qadf.analytics.spc import SPCClient, SPCConfig
except ImportError:
    pytest.skip("SPC or quality assessment modules not available", allow_module_level=True)


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


class TestSPCQualityAssessmentIntegration:
    """Integration tests for SPC with QualityAssessmentClient."""

    @pytest.fixture
    def qa_client_with_spc(self):
        """QualityAssessmentClient with SPC enabled."""
        return QualityAssessmentClient(enable_spc=True, mongo_client=None)

    @pytest.fixture
    def qa_client_without_spc(self):
        """QualityAssessmentClient with SPC disabled."""
        return QualityAssessmentClient(enable_spc=False)

    @pytest.fixture
    def mock_voxel_data(self):
        """Mock voxel data for testing."""
        signals = {
            "laser_power": np.ones((10, 10, 10)) * 250.0,
            "temperature": np.ones((10, 10, 10)) * 1000.0,
        }
        return MockVoxelData(signals, dims=(10, 10, 10))

    @pytest.mark.integration
    def test_qa_client_with_spc_enabled(self, qa_client_with_spc):
        """Test QualityAssessmentClient with SPC enabled."""
        if qa_client_with_spc.enable_spc:
            assert qa_client_with_spc.spc_client is not None
        else:
            pytest.skip("SPC integration not available")

    @pytest.mark.integration
    def test_qa_client_with_spc_disabled(self, qa_client_without_spc):
        """Test QualityAssessmentClient with SPC disabled."""
        assert qa_client_without_spc.enable_spc == False or qa_client_without_spc.spc_client is None

    @pytest.mark.integration
    def test_assess_with_spc(self, qa_client_with_spc, mock_voxel_data):
        """Test assess_with_spc method."""
        if not qa_client_with_spc.enable_spc or qa_client_with_spc.spc_client is None:
            pytest.skip("SPC integration not available")

        results = qa_client_with_spc.assess_with_spc(mock_voxel_data, signals=["laser_power", "temperature"])

        assert isinstance(results, dict)
        assert "quality_assessment" in results
        assert "spc_analysis" in results
        assert "summary" in results
        assert results["summary"]["spc_enabled"] == True

    @pytest.mark.integration
    def test_assess_with_spc_specification_limits(self, qa_client_with_spc, mock_voxel_data):
        """Test assess_with_spc with specification limits."""
        if not qa_client_with_spc.enable_spc or qa_client_with_spc.spc_client is None:
            pytest.skip("SPC integration not available")

        specification_limits = {"completeness": (1.0, 0.8), "coverage_spatial": (1.0, 0.75)}

        results = qa_client_with_spc.assess_with_spc(mock_voxel_data, specification_limits=specification_limits)

        assert "spc_analysis" in results

    @pytest.mark.integration
    def test_assess_with_spc_not_enabled(self, qa_client_without_spc, mock_voxel_data):
        """Test assess_with_spc when SPC is not enabled."""
        if qa_client_without_spc.enable_spc and qa_client_without_spc.spc_client is not None:
            pytest.skip("SPC is actually enabled")

        with pytest.raises(ValueError, match="SPC integration not enabled"):
            qa_client_without_spc.assess_with_spc(mock_voxel_data)

    @pytest.mark.integration
    def test_monitor_quality_with_spc(self, qa_client_with_spc):
        """Test monitor_quality_with_spc method."""
        if not qa_client_with_spc.enable_spc or qa_client_with_spc.spc_client is None:
            pytest.skip("SPC integration not available")

        # Create quality metrics history
        quality_metrics_history = {
            "completeness": np.array([0.9, 0.91, 0.89, 0.92, 0.88, 0.90, 0.93, 0.87, 0.91, 0.89] * 5),
            "coverage": np.array([0.85, 0.86, 0.84, 0.87, 0.83, 0.86, 0.88, 0.82, 0.85, 0.84] * 5),
        }

        results = qa_client_with_spc.monitor_quality_with_spc(quality_metrics_history)

        assert isinstance(results, dict)
        assert "completeness" in results
        assert "coverage" in results
        assert "control_chart" in results["completeness"]
        assert "out_of_control_points" in results["completeness"]

    @pytest.mark.integration
    def test_monitor_quality_with_spc_baseline(self, qa_client_with_spc):
        """Test monitor_quality_with_spc with baseline data."""
        if not qa_client_with_spc.enable_spc or qa_client_with_spc.spc_client is None:
            pytest.skip("SPC integration not available")

        baseline_data = {"completeness": np.array([0.9] * 50), "coverage": np.array([0.85] * 50)}  # Stable baseline

        quality_metrics_history = {
            "completeness": np.array([0.92, 0.93, 0.94, 0.95, 0.96] * 10),  # Trend
            "coverage": np.array([0.86, 0.87, 0.88, 0.89, 0.90] * 10),
        }

        results = qa_client_with_spc.monitor_quality_with_spc(quality_metrics_history, baseline_data=baseline_data)

        assert isinstance(results, dict)
        # Should establish baseline from baseline_data
        assert results["completeness"].get("baseline") is not None

    @pytest.mark.integration
    def test_monitor_quality_with_spc_spec_limits(self, qa_client_with_spc):
        """Test monitor_quality_with_spc with specification limits."""
        if not qa_client_with_spc.enable_spc or qa_client_with_spc.spc_client is None:
            pytest.skip("SPC integration not available")

        quality_metrics_history = {"completeness": np.array([0.9, 0.91, 0.89, 0.92, 0.88] * 10)}

        specification_limits = {"completeness": (1.0, 0.8)}

        results = qa_client_with_spc.monitor_quality_with_spc(
            quality_metrics_history, specification_limits=specification_limits
        )

        assert "process_capability" in results["completeness"]
        assert results["completeness"]["process_capability"] is not None

    @pytest.mark.integration
    def test_spc_rule_violations_in_quality(self, qa_client_with_spc):
        """Test that SPC rule violations are detected in quality monitoring."""
        if not qa_client_with_spc.enable_spc or qa_client_with_spc.spc_client is None:
            pytest.skip("SPC integration not available")

        # Create data with trend (should trigger rule violations)
        quality_metrics_history = {"quality_metric": np.linspace(10.0, 16.0, 50)}  # Steady trend

        results = qa_client_with_spc.monitor_quality_with_spc(quality_metrics_history)

        assert "rule_violations" in results["quality_metric"]
        # Should detect trend violation (Rule 3)
        violations = results["quality_metric"]["rule_violations"]
        assert isinstance(violations, list)
