"""
Bridge tests: fusion (GridFusion, FusionQualityAssessor) via am_qadf_native.

Aligned with docs/Tests/Test_New_Plans.md (tests/integration/bridge/).
"""

import pytest


@pytest.mark.integration
@pytest.mark.bridge
class TestFusionBridge:
    """Python → C++ fusion API."""

    def test_grid_fusion_fuse_weighted_average(self, native_module):
        """GridFusion.fuse with weighted_average returns single grid."""
        UniformVoxelGrid = native_module.UniformVoxelGrid
        GridFusion = native_module.fusion.GridFusion
        g1 = UniformVoxelGrid(1.0, 0.0, 0.0, 0.0)
        g1.add_point_at_voxel(0, 0, 0, 10.0)
        g2 = UniformVoxelGrid(1.0, 0.0, 0.0, 0.0)
        g2.add_point_at_voxel(0, 0, 0, 30.0)
        fusion = GridFusion()
        fused = fusion.fuse([g1.get_grid(), g2.get_grid()], "weighted_average")
        assert fused is not None
        # Value at (0,0,0) should be (10+30)/2 = 20
        # We can't read FloatGrid from Python directly; just check we got a grid
        assert type(fused).__name__ == "OpenVDBFloatGrid"

    def test_grid_fusion_fuse_weighted(self, native_module):
        """GridFusion.fuse_weighted with custom weights."""
        UniformVoxelGrid = native_module.UniformVoxelGrid
        GridFusion = native_module.fusion.GridFusion
        g1 = UniformVoxelGrid(1.0)
        g1.add_point_at_voxel(0, 0, 0, 0.0)
        g2 = UniformVoxelGrid(1.0)
        g2.add_point_at_voxel(0, 0, 0, 100.0)
        fusion = GridFusion()
        fused = fusion.fuse_weighted([g1.get_grid(), g2.get_grid()], [0.8, 0.2])
        assert fused is not None

    def test_fusion_quality_assess_from_grids(self, native_module):
        """FusionQualityAssessor.assess_from_grids (Python wrapper → C++) returns FusionQualityMetrics."""
        try:
            from am_qadf.fusion.fusion_quality import (
                FusionQualityAssessor,
                FusionQualityMetrics,
            )
        except ImportError:
            pytest.skip("Fusion quality C++ bindings not available (am_qadf_native.fusion)")

        UniformVoxelGrid = native_module.UniformVoxelGrid
        GridFusion = native_module.fusion.GridFusion
        g1 = UniformVoxelGrid(1.0, 0.0, 0.0, 0.0)
        g1.add_point_at_voxel(0, 0, 0, 10.0)
        g2 = UniformVoxelGrid(1.0, 0.0, 0.0, 0.0)
        g2.add_point_at_voxel(0, 0, 0, 30.0)
        fusion = GridFusion()
        fused_grid = fusion.fuse([g1.get_grid(), g2.get_grid()], "weighted_average")
        source_grids = {"s1": g1.get_grid(), "s2": g2.get_grid()}

        assessor = FusionQualityAssessor()
        metrics = assessor.assess_from_grids(fused_grid, source_grids)

        assert isinstance(metrics, FusionQualityMetrics)
        assert 0.0 <= metrics.fusion_accuracy <= 1.0
        assert 0.0 <= metrics.signal_consistency <= 1.0
        assert 0.0 <= metrics.fusion_completeness <= 1.0
        assert 0.0 <= metrics.quality_score <= 1.0
        assert metrics.coverage_ratio == metrics.fusion_completeness
        assert "s1" in metrics.per_signal_accuracy
        assert "s2" in metrics.per_signal_accuracy
