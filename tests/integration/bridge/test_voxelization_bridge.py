"""
Bridge tests: voxelization (UniformVoxelGrid, VoxelGridFactory) via am_qadf_native.

Aligned with docs/Tests/Test_New_Plans.md (tests/integration/bridge/).
"""

import pytest


@pytest.mark.integration
@pytest.mark.bridge
class TestVoxelizationBridge:
    """Python → C++ voxelization API."""

    def test_uniform_voxel_grid_create_and_add_point(self, native_module):
        """UniformVoxelGrid: create, add_point_at_voxel, get_value."""
        UniformVoxelGrid = native_module.UniformVoxelGrid
        grid = UniformVoxelGrid(1.0, 0.0, 0.0, 0.0)
        grid.set_signal_name("signal")
        grid.add_point_at_voxel(0, 0, 0, 5.0)
        grid.add_point_at_voxel(1, 0, 0, 10.0)
        assert grid.get_value(0, 0, 0) == pytest.approx(5.0)
        assert grid.get_value(1, 0, 0) == pytest.approx(10.0)

    def test_uniform_voxel_grid_add_point_world(self, native_module):
        """UniformVoxelGrid: add_point (world coords), get_value_at_world."""
        UniformVoxelGrid = native_module.UniformVoxelGrid
        grid = UniformVoxelGrid(1.0, 0.0, 0.0, 0.0)
        grid.add_point(0.5, 0.5, 0.5, 7.0)
        assert grid.get_value(0, 0, 0) == pytest.approx(7.0)
        assert grid.get_value_at_world(0.5, 0.5, 0.5) == pytest.approx(7.0)

    def test_uniform_voxel_grid_get_grid(self, native_module):
        """UniformVoxelGrid: get_grid returns OpenVDB grid (shared_ptr)."""
        UniformVoxelGrid = native_module.UniformVoxelGrid
        grid = UniformVoxelGrid(1.0)
        grid.add_point_at_voxel(0, 0, 0, 1.0)
        g = grid.get_grid()
        assert g is not None
        assert type(g).__name__ == "OpenVDBFloatGrid"

    def test_voxel_grid_factory_create_uniform(self, native_module):
        """VoxelGridFactory.create_uniform(resolution, signal_name) returns UniformVoxelGrid."""
        VoxelGridFactory = native_module.VoxelGridFactory
        grid = VoxelGridFactory.create_uniform(1.0, "mean")
        assert grid is not None
        assert grid.get_voxel_size() == pytest.approx(1.0)

    def test_uniform_voxel_grid_statistics(self, native_module):
        """UniformVoxelGrid: get_statistics returns filled_voxels, mean, etc."""
        UniformVoxelGrid = native_module.UniformVoxelGrid
        grid = UniformVoxelGrid(1.0)
        grid.add_point_at_voxel(0, 0, 0, 1.0)
        grid.add_point_at_voxel(1, 0, 0, 3.0)
        stats = grid.get_statistics()
        assert hasattr(stats, "filled_voxels")
        assert hasattr(stats, "mean")
        assert stats.filled_voxels >= 2
        assert stats.mean == pytest.approx(2.0)
