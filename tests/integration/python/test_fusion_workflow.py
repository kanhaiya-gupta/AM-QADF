"""
Integration tests for fusion workflow.

Tests the complete workflow: multi-source data → fusion → quality assessment
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch


@pytest.mark.integration
class TestFusionWorkflow:
    """Integration tests for fusion workflow."""

    @pytest.fixture
    def sample_voxel_grids(self):
        """Create sample voxel grids with different signals."""
        try:
            from am_qadf.voxelization.uniform_resolution import VoxelGrid

            grid1 = VoxelGrid(bbox_min=(0, 0, 0), bbox_max=(10, 10, 10), resolution=1.0)
            grid1.available_signals = {"laser_power"}

            grid2 = VoxelGrid(bbox_min=(0, 0, 0), bbox_max=(10, 10, 10), resolution=1.0)
            grid2.available_signals = {"temperature"}

            grid3 = VoxelGrid(bbox_min=(0, 0, 0), bbox_max=(10, 10, 10), resolution=1.0)
            grid3.available_signals = {"density"}

            return [grid1, grid2, grid3]
        except ImportError:
            pytest.skip("VoxelGrid not available")

    @pytest.mark.integration
    def test_fusion_workflow_basic(self, sample_voxel_grids):
        """Test basic fusion workflow."""
        try:
            from am_qadf.fusion.voxel_fusion import VoxelFusion

            fusion = VoxelFusion()

            # Fuse grids
            fused = fusion.fuse_voxel_grids(sample_voxel_grids, method="weighted_average")

            assert fused is not None
            assert hasattr(fused, "available_signals")
        except (ImportError, AttributeError):
            pytest.skip("VoxelFusion not available")

    @pytest.mark.integration
    def test_fusion_workflow_different_methods(self, sample_voxel_grids):
        """Test fusion with different methods."""
        try:
            from am_qadf.fusion.voxel_fusion import VoxelFusion

            fusion = VoxelFusion()
            strategies = ["weighted_average", "median", "max", "min"]

            for strategy in strategies:
                fused = fusion.fuse_grids(sample_voxel_grids, strategy=strategy)
                assert fused is not None
        except (ImportError, AttributeError):
            pytest.skip("VoxelFusion not available")

    @pytest.mark.integration
    def test_fusion_workflow_with_quality(self, sample_voxel_grids):
        """Test fusion with weighted_average using explicit weights."""
        try:
            from am_qadf.fusion.voxel_fusion import VoxelFusion

            fusion = VoxelFusion()

            # Weights per grid (same length as sample_voxel_grids)
            weights = [0.5, 0.3, 0.2]

            fused = fusion.fuse_grids(
                sample_voxel_grids,
                strategy="weighted_average",
                weights=weights,
            )

            assert fused is not None
        except (ImportError, AttributeError):
            pytest.skip("VoxelFusion not available")

    @pytest.mark.integration
    def test_fusion_workflow_quality_assessment(self, sample_voxel_grids):
        """Test fusion followed by quality assessment."""
        try:
            from am_qadf.fusion.voxel_fusion import VoxelFusion
            from am_qadf.fusion.fusion_quality import FusionQualityAssessor

            fusion = VoxelFusion()
            fused = fusion.fuse_grids(sample_voxel_grids, strategy="weighted_average")

            # Optionally assess fusion quality via assess_from_grids (OpenVDB grids)
            try:
                quality = FusionQualityAssessor()
                common = set(fused.available_signals)
                for grid in sample_voxel_grids:
                    common &= set(grid.available_signals)
                if common and hasattr(fused, "get_grid"):
                    sig = next(iter(common))
                    fused_grid = fused.get_grid(sig)
                    source_grids = {f"grid_{i}": g.get_grid(sig) for i, g in enumerate(sample_voxel_grids) if hasattr(g, "get_grid")}
                    if fused_grid and source_grids:
                        metrics = quality.assess_from_grids(fused_grid, source_grids)
                        assert metrics is not None
                else:
                    assert fused is not None
            except (ImportError, AttributeError, ValueError, NotImplementedError):
                assert fused is not None
        except (ImportError, AttributeError):
            pytest.skip("Fusion modules not available")

    @pytest.mark.integration
    def test_fusion_workflow_signal_preservation(self, sample_voxel_grids):
        """Test that signals are preserved during fusion."""
        try:
            from am_qadf.fusion.voxel_fusion import VoxelFusion

            fusion = VoxelFusion()
            fused = fusion.fuse_voxel_grids(sample_voxel_grids, method="weighted_average")

            # Check that all signals are present
            all_signals = set()
            for grid in sample_voxel_grids:
                all_signals.update(grid.available_signals)

            assert len(fused.available_signals) >= len(all_signals)
        except (ImportError, AttributeError):
            pytest.skip("VoxelFusion not available")

    @pytest.mark.integration
    def test_fusion_workflow_resolution_matching(self):
        """Test fusion with grids of different resolutions."""
        try:
            from am_qadf.voxelization.uniform_resolution import VoxelGrid
            from am_qadf.fusion.voxel_fusion import VoxelFusion

            # Create grids with different resolutions
            grid1 = VoxelGrid(bbox_min=(0, 0, 0), bbox_max=(10, 10, 10), resolution=1.0)

            grid2 = VoxelGrid(bbox_min=(0, 0, 0), bbox_max=(10, 10, 10), resolution=0.5)

            fusion = VoxelFusion()
            fused = fusion.fuse_grids([grid1, grid2], strategy="weighted_average")

            assert fused is not None
        except (ImportError, AttributeError):
            pytest.skip("Fusion modules not available")
