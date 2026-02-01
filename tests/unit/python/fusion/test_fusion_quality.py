"""
Unit tests for fusion quality assessment.

Fusion quality is C++ only (no Python fallback). FusionQualityAssessor tests
require am_qadf_native; they are skipped when native is not built.
"""

import pytest
import numpy as np
from unittest.mock import patch
from am_qadf.fusion.fusion_quality import (
    FusionQualityMetrics,
    FusionQualityAssessor,
    CPP_AVAILABLE,
)


class TestFusionQualityMetrics:
    """Test suite for FusionQualityMetrics dataclass (no C++ required)."""

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


@pytest.mark.skipif(not CPP_AVAILABLE, reason="Fusion quality requires am_qadf_native (C++ only, no Python fallback)")
class TestFusionQualityAssessor:
    """Test suite for FusionQualityAssessor (C++ only)."""

    @pytest.fixture
    def assessor(self):
        """Create a FusionQualityAssessor instance (requires C++)."""
        return FusionQualityAssessor()

    @pytest.mark.unit
    def test_assessor_creation(self, assessor):
        """Test creating FusionQualityAssessor when C++ available."""
        assert assessor is not None
        assert assessor._cpp_assessor is not None

    @pytest.mark.unit
    def test_assess_fusion_quality_raises_not_implemented(self, assessor):
        """Numpy input is not supported; must use assess_from_grids."""
        fused_array = np.array([2.0, 3.0, 4.0])
        source_arrays = {"s1": np.array([1.0, 2.0, 3.0])}

        with pytest.raises(NotImplementedError, match="assess_from_grids"):
            assessor.assess_fusion_quality(fused_array, source_arrays)

    @pytest.mark.unit
    def test_compare_fusion_strategies_raises_not_implemented(self, assessor):
        """compare_fusion_strategies is not implemented without Python fallback."""
        with pytest.raises(NotImplementedError, match="compare_fusion_strategies"):
            assessor.compare_fusion_strategies(None, ["s1"], ["average"])

    @pytest.mark.unit
    def test_assess_from_grids_returns_metrics(self, assessor):
        """assess_from_grids with OpenVDB grids returns FusionQualityMetrics."""
        try:
            from am_qadf_native import UniformVoxelGrid
        except ImportError:
            pytest.skip("UniformVoxelGrid not available")

        # Create small grids via native
        g1 = UniformVoxelGrid(1.0, 0.0, 0.0, 0.0)
        g1.add_point_at_voxel(0, 0, 0, 10.0)
        g2 = UniformVoxelGrid(1.0, 0.0, 0.0, 0.0)
        g2.add_point_at_voxel(0, 0, 0, 12.0)

        fused_grid = g1.get_grid()
        source_grids = {"s1": g2.get_grid()}

        metrics = assessor.assess_from_grids(fused_grid, source_grids)

        assert isinstance(metrics, FusionQualityMetrics)
        assert 0.0 <= metrics.fusion_accuracy <= 1.0
        assert 0.0 <= metrics.signal_consistency <= 1.0
        assert 0.0 <= metrics.coverage_ratio <= 1.0
        assert 0.0 <= metrics.quality_score <= 1.0
        assert "s1" in metrics.per_signal_accuracy


class TestFusionQualityAssessorNoCpp:
    """When C++ is not available, FusionQualityAssessor() raises."""

    @pytest.mark.unit
    def test_assessor_raises_import_error_without_cpp(self):
        """FusionQualityAssessor() raises ImportError when C++ not built."""
        import am_qadf.fusion.fusion_quality as mod
        with patch.object(mod, "CPP_AVAILABLE", False):
            with pytest.raises(ImportError, match=r"Fusion quality requires C\+\+|C\+\+ bindings"):
                FusionQualityAssessor()
