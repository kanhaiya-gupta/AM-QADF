"""
Unit tests for voxel fusion.

Tests for VoxelFusion (thin C++ GridFusion wrapper): fuse_grids() with VoxelGrid.
Requires am_qadf_native; tests skip when C++ not built.
"""

import pytest
import numpy as np
from unittest.mock import patch

pytest.importorskip("am_qadf_native", reason="VoxelFusion and VoxelGrid require am_qadf_native")

from am_qadf.fusion.voxel_fusion import VoxelFusion
from am_qadf.voxelization.uniform_resolution import VoxelGrid


class TestVoxelFusion:
    """Test suite for VoxelFusion (C++ GridFusion wrapper)."""

    @pytest.fixture
    def voxel_fusion(self):
        """Create a VoxelFusion instance (requires C++)."""
        return VoxelFusion()

    @pytest.fixture
    def two_grids_same_shape(self):
        """Two small VoxelGrids with same bbox and resolution, one signal each."""
        bbox_min = (0.0, 0.0, 0.0)
        bbox_max = (2.0, 2.0, 2.0)
        res = 1.0
        g1 = VoxelGrid(bbox_min=bbox_min, bbox_max=bbox_max, resolution=res)
        g2 = VoxelGrid(bbox_min=bbox_min, bbox_max=bbox_max, resolution=res)
        # Fill with values: g1 = 10, g2 = 20 (same voxels)
        arr = np.ones((2, 2, 2), dtype=np.float32) * 10.0
        g1._get_or_create_grid("sig").populate_from_array(arr)
        g1.available_signals.add("sig")
        arr2 = np.ones((2, 2, 2), dtype=np.float32) * 20.0
        g2._get_or_create_grid("sig").populate_from_array(arr2)
        g2.available_signals.add("sig")
        return g1, g2

    @pytest.mark.unit
    def test_voxel_fusion_creation(self, voxel_fusion):
        """VoxelFusion() creates instance when C++ available."""
        assert voxel_fusion is not None
        assert voxel_fusion._fusion is not None

    @pytest.mark.unit
    def test_voxel_fusion_raises_without_cpp(self):
        """VoxelFusion() raises ImportError when am_qadf_native not available."""
        import am_qadf.fusion.voxel_fusion as mod
        with patch.object(mod, "CPP_AVAILABLE", False):
            with pytest.raises(ImportError, match=r"C\+\+ bindings not available|am_qadf_native"):
                VoxelFusion()

    @pytest.mark.unit
    def test_fuse_grids_empty_raises(self, voxel_fusion):
        """fuse_grids with empty list raises ValueError."""
        with pytest.raises(ValueError, match="No grids provided"):
            voxel_fusion.fuse_grids([])

    @pytest.mark.unit
    def test_fuse_grids_single_returns_same(self, voxel_fusion, two_grids_same_shape):
        """fuse_grids with one grid returns that grid unchanged."""
        g1, _ = two_grids_same_shape
        result = voxel_fusion.fuse_grids([g1])
        assert result is g1

    @pytest.mark.unit
    def test_fuse_grids_two_weighted_average(self, voxel_fusion, two_grids_same_shape):
        """fuse_grids with two grids and weighted_average returns grid with averaged signal."""
        g1, g2 = two_grids_same_shape
        fused = voxel_fusion.fuse_grids([g1, g2], strategy="weighted_average")
        assert fused is not None
        assert "sig" in fused.available_signals
        out = fused.get_signal_array("sig", default=0.0)
        # (10 + 20) / 2 = 15
        assert out.shape == (2, 2, 2)
        assert np.allclose(out, 15.0, atol=0.01)

    @pytest.mark.unit
    def test_fuse_grids_two_with_weights(self, voxel_fusion, two_grids_same_shape):
        """fuse_grids with custom weights."""
        g1, g2 = two_grids_same_shape
        fused = voxel_fusion.fuse_grids(
            [g1, g2],
            strategy="weighted_average",
            weights=[0.8, 0.2],
        )
        assert fused is not None
        out = fused.get_signal_array("sig", default=0.0)
        # 0.8*10 + 0.2*20 = 12
        assert np.allclose(out, 12.0, atol=0.01)

    @pytest.mark.unit
    def test_fuse_grids_strategy_max(self, voxel_fusion, two_grids_same_shape):
        """fuse_grids with strategy max takes max per voxel."""
        g1, g2 = two_grids_same_shape
        fused = voxel_fusion.fuse_grids([g1, g2], strategy="max")
        out = fused.get_signal_array("sig", default=0.0)
        assert np.allclose(out, 20.0, atol=0.01)

    @pytest.mark.unit
    def test_fuse_grids_strategy_min(self, voxel_fusion, two_grids_same_shape):
        """fuse_grids with strategy min takes min per voxel."""
        g1, g2 = two_grids_same_shape
        fused = voxel_fusion.fuse_grids([g1, g2], strategy="min")
        out = fused.get_signal_array("sig", default=0.0)
        assert np.allclose(out, 10.0, atol=0.01)
