"""
Unit tests for alignment accuracy validation (analytics).

Tests for AlignmentAccuracyMetrics and AlignmentAccuracyAnalyzer.
"""

import pytest
import numpy as np
from am_qadf.analytics.quality_assessment.alignment_accuracy import (
    AlignmentAccuracyMetrics,
    AlignmentAccuracyAnalyzer,
)


class TestAlignmentAccuracyMetrics:
    """Test suite for AlignmentAccuracyMetrics dataclass."""

    @pytest.mark.unit
    def test_metrics_creation(self):
        """Test creating AlignmentAccuracyMetrics."""
        metrics = AlignmentAccuracyMetrics(
            coordinate_alignment_error=0.05,
            temporal_alignment_error=0.1,
            spatial_registration_error=0.08,
            residual_error_mean=0.06,
            residual_error_std=0.02,
            alignment_score=0.9,
            transformation_errors=[0.05, 0.06, 0.04],
            registration_residuals=np.array([0.05, 0.06, 0.04]),
        )

        assert metrics.coordinate_alignment_error == 0.05
        assert metrics.temporal_alignment_error == 0.1
        assert metrics.spatial_registration_error == 0.08
        assert metrics.residual_error_mean == 0.06
        assert metrics.alignment_score == 0.9
        assert len(metrics.transformation_errors) == 3

    @pytest.mark.unit
    def test_metrics_to_dict(self):
        """Test converting AlignmentAccuracyMetrics to dictionary."""
        metrics = AlignmentAccuracyMetrics(
            coordinate_alignment_error=0.05,
            temporal_alignment_error=0.1,
            spatial_registration_error=0.08,
            residual_error_mean=0.06,
            residual_error_std=0.02,
            alignment_score=0.9,
            transformation_errors=[0.05, 0.06],
            registration_residuals=np.array([0.05, 0.06]),
        )

        result = metrics.to_dict()

        assert isinstance(result, dict)
        assert result["coordinate_alignment_error"] == 0.05
        assert result["transformation_errors_count"] == 2
        assert "registration_residuals_shape" in result


class TestAlignmentAccuracyAnalyzer:
    """Test suite for AlignmentAccuracyAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create an AlignmentAccuracyAnalyzer instance."""
        return AlignmentAccuracyAnalyzer()

    @pytest.mark.unit
    def test_analyzer_creation_default(self):
        """Test creating AlignmentAccuracyAnalyzer with default parameters."""
        analyzer = AlignmentAccuracyAnalyzer()

        assert analyzer.max_acceptable_error == 0.1

    @pytest.mark.unit
    def test_analyzer_creation_custom(self):
        """Test creating AlignmentAccuracyAnalyzer with custom parameters."""
        analyzer = AlignmentAccuracyAnalyzer(max_acceptable_error=0.2)

        assert analyzer.max_acceptable_error == 0.2

    @pytest.mark.unit
    def test_validate_coordinate_alignment(self, analyzer):
        """Test validating coordinate alignment."""
        source_points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        target_points = np.array([[1.05, 2.05, 3.05], [4.05, 5.05, 6.05], [7.05, 8.05, 9.05]])

        error = analyzer.validate_coordinate_alignment(source_points, target_points)

        assert isinstance(error, float)
        assert error > 0
        assert error < 0.1

    @pytest.mark.unit
    def test_validate_temporal_alignment(self, analyzer):
        """Test validating temporal alignment."""
        source_times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        target_times = np.array([0.1, 1.1, 2.1, 3.1, 4.1])

        error = analyzer.validate_temporal_alignment(source_times, target_times)

        assert isinstance(error, float)
        assert error == pytest.approx(0.1, rel=1e-9)

    @pytest.mark.unit
    def test_calculate_registration_residuals(self, analyzer):
        """Test calculating registration residuals."""
        reference_points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        aligned_points = np.array([[1.05, 2.05, 3.05], [4.05, 5.05, 6.05], [7.05, 8.05, 9.05]])

        mean_residual, std_residual, residual_map = analyzer.calculate_registration_residuals(reference_points, aligned_points)

        assert isinstance(mean_residual, float)
        assert isinstance(std_residual, float)
        assert len(residual_map) == len(reference_points)
        assert mean_residual > 0

    @pytest.mark.unit
    def test_assess_alignment_accuracy(self, analyzer):
        """Test assessing overall alignment accuracy."""

        class MockVoxelData:
            def __init__(self):
                self.bbox_min = (0, 0, 0)
                self.bbox_max = (10, 10, 10)

        voxel_data = MockVoxelData()
        metrics = analyzer.assess_alignment_accuracy(voxel_data)

        assert isinstance(metrics, AlignmentAccuracyMetrics)
        assert 0.0 <= metrics.alignment_score <= 1.0
