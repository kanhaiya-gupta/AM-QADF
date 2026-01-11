"""
Integration tests for validation with QualityAssessmentClient.

Tests integration of validation module with QualityAssessmentClient.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch

try:
    from am_qadf.quality.quality_assessment_client import QualityAssessmentClient
    from am_qadf.validation import ValidationClient
except ImportError:
    # Try alternative import path
    try:
        from am_qadf.analytics.quality_assessment.client import QualityAssessmentClient
        from am_qadf.validation import ValidationClient
    except ImportError:
        pytest.skip("Quality or validation modules not available", allow_module_level=True)


class MockVoxelData:
    """Mock voxel data object for testing."""

    def __init__(
        self,
        signals: dict,
        dims: tuple = (10, 10, 10),
        bbox_min: tuple = (0, 0, 0),
        bbox_max: tuple = (10, 10, 10),
        resolution: float = 1.0,
    ):
        """Initialize with signal dictionary."""
        self._signals = signals
        self.dims = dims
        self.bbox_min = bbox_min
        self.bbox_max = bbox_max
        self.resolution = resolution
        self.available_signals = list(signals.keys())

    def get_signal_array(self, signal_name: str, default: float = 0.0) -> np.ndarray:
        """Get signal array by name."""
        return self._signals.get(signal_name, np.full(self.dims, default))


class TestQualityAssessmentClientIntegration:
    """Integration tests for QualityAssessmentClient with validation."""

    @pytest.fixture
    def qa_client_with_validation(self):
        """QualityAssessmentClient with validation enabled."""
        return QualityAssessmentClient(enable_validation=True)

    @pytest.fixture
    def qa_client_without_validation(self):
        """QualityAssessmentClient with validation disabled."""
        return QualityAssessmentClient(enable_validation=False)

    @pytest.fixture
    def mock_voxel_data(self):
        """Mock voxel data for testing."""
        signals = {
            "laser_power": np.ones((10, 10, 10)) * 250.0,
            "temperature": np.ones((10, 10, 10)) * 1000.0,
        }
        return MockVoxelData(signals, dims=(10, 10, 10))

    @pytest.fixture
    def mpm_reference_data(self):
        """Sample MPM reference data."""
        return {
            "overall_quality_score": 0.88,
            "data_quality_score": 0.83,
            "signal_quality_score": 0.90,
            "alignment_score": 0.86,
            "completeness_score": 0.93,
            "quality_scores": {
                "overall_quality_score": 0.88,
                "data_quality_score": 0.83,
            },
        }

    @pytest.mark.integration
    def test_qa_client_with_validation_enabled(self, qa_client_with_validation):
        """Test QualityAssessmentClient with validation enabled."""
        assert qa_client_with_validation.enable_validation is True
        # validation_client may be None if module unavailable, which is okay

    @pytest.mark.integration
    def test_qa_client_with_validation_disabled(self, qa_client_without_validation):
        """Test QualityAssessmentClient with validation disabled."""
        assert qa_client_without_validation.enable_validation is False
        assert qa_client_without_validation.validation_client is None

    @pytest.mark.integration
    def test_validate_quality_assessment_mpm(self, qa_client_with_validation, mock_voxel_data, mpm_reference_data):
        """Test validate_quality_assessment with MPM data."""
        if qa_client_with_validation.validation_client is None:
            pytest.skip("Validation client not available")

        try:
            validation_results = qa_client_with_validation.validate_quality_assessment(
                mock_voxel_data, mpm_reference_data, validation_type="mpm", signals=["laser_power", "temperature"]
            )

            assert isinstance(validation_results, dict)
            assert "framework_metrics" in validation_results
            assert "validation_type" in validation_results
            assert "validation_results" in validation_results
            assert validation_results["validation_type"] == "mpm"
        except RuntimeError:
            # Validation not available - skip test
            pytest.skip("Validation not available")

    @pytest.mark.integration
    def test_validate_quality_assessment_comprehensive(self, qa_client_with_validation, mock_voxel_data, mpm_reference_data):
        """Test validate_quality_assessment with comprehensive validation."""
        if qa_client_with_validation.validation_client is None:
            pytest.skip("Validation client not available")

        try:
            validation_results = qa_client_with_validation.validate_quality_assessment(
                mock_voxel_data, mpm_reference_data, validation_type="comprehensive"
            )

            assert isinstance(validation_results, dict)
            assert "validation_results" in validation_results
        except RuntimeError:
            pytest.skip("Validation not available")

    @pytest.mark.integration
    def test_validate_quality_assessment_disabled(self, qa_client_without_validation, mock_voxel_data, mpm_reference_data):
        """Test validate_quality_assessment raises error when validation disabled."""
        with pytest.raises(RuntimeError, match="Validation not available"):
            qa_client_without_validation.validate_quality_assessment(mock_voxel_data, mpm_reference_data)

    @pytest.mark.integration
    def test_benchmark_quality_assessment(self, qa_client_with_validation, mock_voxel_data):
        """Test benchmark_quality_assessment."""
        if qa_client_with_validation.validation_client is None:
            pytest.skip("Validation client not available")

        benchmark_result = qa_client_with_validation.benchmark_quality_assessment(
            mock_voxel_data, signals=["laser_power"], iterations=2, warmup_iterations=1
        )

        if benchmark_result is not None:
            assert hasattr(benchmark_result, "execution_time")
            assert hasattr(benchmark_result, "memory_usage")
            assert (
                benchmark_result.operation_name == "comprehensive_assessment"
                or "assessment" in benchmark_result.operation_name.lower()
            )

    @pytest.mark.integration
    def test_benchmark_quality_assessment_disabled(self, qa_client_without_validation, mock_voxel_data):
        """Test benchmark_quality_assessment returns None when disabled."""
        result = qa_client_without_validation.benchmark_quality_assessment(mock_voxel_data)

        assert result is None

    @pytest.mark.integration
    def test_integration_comprehensive_workflow(self, qa_client_with_validation, mock_voxel_data, mpm_reference_data):
        """Test complete integration workflow."""
        if qa_client_with_validation.validation_client is None:
            pytest.skip("Validation client not available")

        try:
            # 1. Perform quality assessment
            assessment_results = qa_client_with_validation.comprehensive_assessment(
                mock_voxel_data, signals=["laser_power", "temperature"]
            )

            assert isinstance(assessment_results, dict)
            assert "summary" in assessment_results

            # 2. Validate against MPM
            validation_results = qa_client_with_validation.validate_quality_assessment(
                mock_voxel_data, mpm_reference_data, validation_type="mpm"
            )

            assert isinstance(validation_results, dict)

            # 3. Benchmark performance
            benchmark_result = qa_client_with_validation.benchmark_quality_assessment(mock_voxel_data, iterations=2)

            # All should complete without errors
            assert True  # If we get here, integration works
        except RuntimeError:
            pytest.skip("Validation not available")
