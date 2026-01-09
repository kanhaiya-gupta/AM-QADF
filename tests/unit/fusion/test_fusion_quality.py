"""
Unit tests for fusion quality assessment.

Tests for FusionQualityMetrics and FusionQualityAssessor.
"""

import pytest
import numpy as np
from am_qadf.fusion.fusion_quality import (
    FusionQualityMetrics,
    FusionQualityAssessor,
)


class MockVoxelData:
    """Mock voxel data object for testing."""

    def __init__(self, signals: dict):
        """Initialize with signal dictionary."""
        self._signals = signals

    def get_signal_array(self, signal_name: str, default: float = 0.0) -> np.ndarray:
        """Get signal array by name."""
        return self._signals.get(signal_name, np.array([default]))


class TestFusionQualityMetrics:
    """Test suite for FusionQualityMetrics dataclass."""

    @pytest.mark.unit
    def test_fusion_quality_metrics_creation(self):
        """Test creating FusionQualityMetrics."""
        metrics = FusionQualityMetrics(
            fusion_accuracy=0.9,
            signal_consistency=0.8,
            fusion_completeness=0.95,
            quality_score=0.88,
            per_signal_accuracy={"signal1": 0.9, "signal2": 0.85},
            coverage_ratio=0.95,
            residual_errors=np.array([0.1, 0.2, 0.15]),
        )

        assert metrics.fusion_accuracy == 0.9
        assert metrics.signal_consistency == 0.8
        assert metrics.fusion_completeness == 0.95
        assert metrics.quality_score == 0.88
        assert metrics.per_signal_accuracy["signal1"] == 0.9
        assert metrics.coverage_ratio == 0.95

    @pytest.mark.unit
    def test_fusion_quality_metrics_to_dict(self):
        """Test converting FusionQualityMetrics to dictionary."""
        metrics = FusionQualityMetrics(
            fusion_accuracy=0.9,
            signal_consistency=0.8,
            fusion_completeness=0.95,
            quality_score=0.88,
            per_signal_accuracy={"signal1": 0.9},
            coverage_ratio=0.95,
            residual_errors=np.array([0.1, 0.2]),
        )

        result = metrics.to_dict()

        assert isinstance(result, dict)
        assert result["fusion_accuracy"] == 0.9
        assert result["signal_consistency"] == 0.8
        assert result["fusion_completeness"] == 0.95
        assert result["quality_score"] == 0.88
        assert result["per_signal_accuracy"]["signal1"] == 0.9
        assert result["coverage_ratio"] == 0.95
        assert "residual_errors_shape" in result

    @pytest.mark.unit
    def test_fusion_quality_metrics_to_dict_no_residuals(self):
        """Test converting to dictionary without residual errors."""
        metrics = FusionQualityMetrics(
            fusion_accuracy=0.9,
            signal_consistency=0.8,
            fusion_completeness=0.95,
            quality_score=0.88,
            per_signal_accuracy={},
            coverage_ratio=0.95,
            residual_errors=None,
        )

        result = metrics.to_dict()

        assert "residual_errors_shape" not in result


class TestFusionQualityAssessor:
    """Test suite for FusionQualityAssessor class."""

    @pytest.fixture
    def assessor(self):
        """Create a FusionQualityAssessor instance."""
        return FusionQualityAssessor()

    @pytest.mark.unit
    def test_fusion_quality_assessor_creation(self, assessor):
        """Test creating FusionQualityAssessor."""
        assert assessor is not None

    @pytest.mark.unit
    def test_assess_fusion_quality(self, assessor):
        """Test assessing fusion quality."""
        fused_array = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        source_arrays = {
            "signal1": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            "signal2": np.array([3.0, 4.0, 5.0, 6.0, 7.0]),
        }

        metrics = assessor.assess_fusion_quality(fused_array, source_arrays)

        assert isinstance(metrics, FusionQualityMetrics)
        assert 0.0 <= metrics.fusion_accuracy <= 1.0
        assert 0.0 <= metrics.signal_consistency <= 1.0
        assert 0.0 <= metrics.fusion_completeness <= 1.0
        assert 0.0 <= metrics.quality_score <= 1.0
        assert "signal1" in metrics.per_signal_accuracy
        assert "signal2" in metrics.per_signal_accuracy

    @pytest.mark.unit
    def test_assess_fusion_quality_with_weights(self, assessor):
        """Test assessing fusion quality with fusion weights."""
        fused_array = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        source_arrays = {
            "signal1": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            "signal2": np.array([3.0, 4.0, 5.0, 6.0, 7.0]),
        }
        fusion_weights = {"signal1": 0.7, "signal2": 0.3}

        metrics = assessor.assess_fusion_quality(fused_array, source_arrays, fusion_weights=fusion_weights)

        assert isinstance(metrics, FusionQualityMetrics)
        assert metrics.residual_errors is not None

    @pytest.mark.unit
    def test_assess_fusion_quality_empty_fused(self, assessor):
        """Test assessing quality with empty fused array."""
        fused_array = np.array([])
        source_arrays = {"signal1": np.array([1.0, 2.0, 3.0])}

        metrics = assessor.assess_fusion_quality(fused_array, source_arrays)

        assert metrics.coverage_ratio == 0.0

    @pytest.mark.unit
    def test_assess_fusion_quality_with_nan(self, assessor):
        """Test assessing quality with NaN values."""
        fused_array = np.array([2.0, np.nan, 4.0, 5.0, 6.0])
        source_arrays = {
            "signal1": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            "signal2": np.array([3.0, 4.0, 5.0, 6.0, 7.0]),
        }

        metrics = assessor.assess_fusion_quality(fused_array, source_arrays)

        assert metrics.coverage_ratio < 1.0  # Less than 1.0 due to NaN

    @pytest.mark.unit
    def test_assess_fusion_quality_with_zeros(self, assessor):
        """Test assessing quality with zero values."""
        fused_array = np.array([2.0, 0.0, 4.0, 5.0, 6.0])
        source_arrays = {"signal1": np.array([1.0, 2.0, 3.0, 4.0, 5.0])}

        metrics = assessor.assess_fusion_quality(fused_array, source_arrays)

        assert metrics.coverage_ratio < 1.0  # Less than 1.0 due to zero

    @pytest.mark.unit
    def test_assess_fusion_quality_perfect_match(self, assessor):
        """Test assessing quality with perfect match."""
        source_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        fused_array = source_array.copy()  # Perfect match

        metrics = assessor.assess_fusion_quality(fused_array, {"signal1": source_array})

        # Should have high accuracy
        assert metrics.fusion_accuracy > 0.9

    @pytest.mark.unit
    def test_compare_fusion_strategies(self, assessor):
        """Test comparing different fusion strategies."""
        signals = {
            "signal1": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            "signal2": np.array([2.0, 3.0, 4.0, 5.0, 6.0]),
        }
        voxel_data = MockVoxelData(signals)

        strategies = ["average", "median", "max"]

        results = assessor.compare_fusion_strategies(voxel_data, ["signal1", "signal2"], strategies)

        assert len(results) == 3
        assert "average" in results
        assert "median" in results
        assert "max" in results
        assert all(isinstance(m, FusionQualityMetrics) for m in results.values())

    @pytest.mark.unit
    def test_compare_fusion_strategies_with_quality(self, assessor):
        """Test comparing strategies with quality scores."""
        signals = {
            "signal1": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            "signal2": np.array([2.0, 3.0, 4.0, 5.0, 6.0]),
        }
        voxel_data = MockVoxelData(signals)
        quality_scores = {"signal1": 0.9, "signal2": 0.5}

        strategies = ["weighted_average", "average"]

        results = assessor.compare_fusion_strategies(
            voxel_data,
            ["signal1", "signal2"],
            strategies,
            quality_scores=quality_scores,
        )

        assert len(results) == 2

    @pytest.mark.unit
    def test_compare_fusion_strategies_invalid_strategy(self, assessor):
        """Test comparing with invalid strategy name."""
        signals = {
            "signal1": np.array([1.0, 2.0, 3.0]),
            "signal2": np.array([2.0, 3.0, 4.0]),
        }
        voxel_data = MockVoxelData(signals)

        strategies = ["invalid_strategy", "average"]

        results = assessor.compare_fusion_strategies(voxel_data, ["signal1", "signal2"], strategies)

        # Should only return results for valid strategies
        assert "average" in results
        assert "invalid_strategy" not in results

    @pytest.mark.unit
    def test_compare_fusion_strategies_empty_voxel_data(self, assessor):
        """Test comparing strategies with empty voxel data."""
        empty_voxel_data = MockVoxelData({})

        results = assessor.compare_fusion_strategies(empty_voxel_data, ["signal1", "signal2"], ["average"])

        assert len(results) == 0

    @pytest.mark.unit
    def test_assess_fusion_quality_2d(self, assessor):
        """Test assessing quality for 2D arrays."""
        fused_array = np.array([[2.0, 3.0], [4.0, 5.0]])
        source_arrays = {
            "signal1": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "signal2": np.array([[3.0, 4.0], [5.0, 6.0]]),
        }

        metrics = assessor.assess_fusion_quality(fused_array, source_arrays)

        assert isinstance(metrics, FusionQualityMetrics)
        assert metrics.residual_errors.shape == fused_array.shape

    @pytest.mark.unit
    def test_assess_fusion_quality_3d(self, assessor):
        """Test assessing quality for 3D arrays."""
        fused_array = np.ones((3, 3, 3)) * 2.0
        source_arrays = {
            "signal1": np.ones((3, 3, 3)) * 1.0,
            "signal2": np.ones((3, 3, 3)) * 3.0,
        }

        metrics = assessor.assess_fusion_quality(fused_array, source_arrays)

        assert isinstance(metrics, FusionQualityMetrics)
        assert metrics.residual_errors.shape == fused_array.shape
