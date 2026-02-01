"""
Bridge tests: synchronization (GridSynchronizer, SpatialAlignment) via am_qadf_native.

Aligned with docs/Tests/Test_New_Plans.md (tests/integration/bridge/).
"""

import pytest


@pytest.mark.integration
@pytest.mark.bridge
class TestSynchronizationBridge:
    """Python → C++ synchronization API."""

    def test_grid_synchronizer_synchronize_spatial(self, native_module):
        """GridSynchronizer.synchronize_spatial returns list of grids."""
        UniformVoxelGrid = native_module.UniformVoxelGrid
        GridSynchronizer = native_module.GridSynchronizer
        ref = UniformVoxelGrid(1.0, 0.0, 0.0, 0.0)
        ref.add_point_at_voxel(0, 0, 0, 0.0)
        src = UniformVoxelGrid(1.0, 0.0, 0.0, 0.0)
        src.add_point_at_voxel(0, 0, 0, 5.0)
        sync = GridSynchronizer()
        out = sync.synchronize_spatial([src.get_grid()], ref.get_grid(), "trilinear")
        assert out is not None
        assert len(out) == 1
        assert out[0] is not None

    def test_spatial_alignment_transforms_match(self, native_module):
        """SpatialAlignment.transforms_match returns bool for two grids."""
        UniformVoxelGrid = native_module.UniformVoxelGrid
        SpatialAlignment = native_module.SpatialAlignment
        g1 = UniformVoxelGrid(1.0)
        g2 = UniformVoxelGrid(1.0)
        aligner = SpatialAlignment()
        match = aligner.transforms_match(g1.get_grid(), g2.get_grid())
        assert match is True

    def test_spatial_alignment_get_world_bounding_box(self, native_module):
        """SpatialAlignment.get_world_bounding_box returns 6-tuple."""
        UniformVoxelGrid = native_module.UniformVoxelGrid
        SpatialAlignment = native_module.SpatialAlignment
        grid = UniformVoxelGrid(1.0, 0.0, 0.0, 0.0)
        grid.add_point_at_voxel(0, 0, 0, 1.0)
        aligner = SpatialAlignment()
        bbox = aligner.get_world_bounding_box(grid.get_grid())
        assert len(bbox) == 6
        assert bbox[0] <= bbox[3]
        assert bbox[1] <= bbox[4]
        assert bbox[2] <= bbox[5]
