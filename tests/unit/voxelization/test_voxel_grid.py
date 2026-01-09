"""
Unit tests for VoxelGrid.

Tests for voxel grid creation, point addition, signal retrieval, and statistics.
"""

import pytest
import numpy as np
from am_qadf.voxelization.voxel_grid import VoxelGrid
from am_qadf.core.entities import VoxelData


class TestVoxelGrid:
    """Test suite for VoxelGrid class."""

    @pytest.mark.unit
    def test_voxel_grid_creation(self):
        """Test creating a VoxelGrid."""
        grid = VoxelGrid(bbox_min=(0.0, 0.0, 0.0), bbox_max=(10.0, 10.0, 10.0), resolution=1.0)

        assert np.array_equal(grid.bbox_min, [0.0, 0.0, 0.0])
        assert np.array_equal(grid.bbox_max, [10.0, 10.0, 10.0])
        assert grid.resolution == 1.0
        assert grid.aggregation == "mean"
        assert len(grid.voxels) == 0
        assert len(grid.available_signals) == 0

    @pytest.mark.unit
    def test_voxel_grid_dimensions_calculation(self):
        """Test voxel grid dimensions calculation."""
        grid = VoxelGrid(bbox_min=(0.0, 0.0, 0.0), bbox_max=(10.0, 10.0, 10.0), resolution=1.0)

        # Should be 10x10x10 voxels
        assert np.array_equal(grid.dims, [10, 10, 10])

    @pytest.mark.unit
    def test_voxel_grid_dimensions_rounding(self):
        """Test voxel grid dimensions with non-integer division."""
        grid = VoxelGrid(bbox_min=(0.0, 0.0, 0.0), bbox_max=(10.0, 10.0, 10.0), resolution=3.0)

        # Should round up: ceil(10/3) = 4
        assert np.array_equal(grid.dims, [4, 4, 4])

    @pytest.mark.unit
    def test_voxel_grid_minimum_dimensions(self):
        """Test that grid has at least 1 voxel in each dimension."""
        grid = VoxelGrid(bbox_min=(0.0, 0.0, 0.0), bbox_max=(0.1, 0.1, 0.1), resolution=1.0)

        # Should be at least 1x1x1
        assert np.all(grid.dims >= 1)

    @pytest.mark.unit
    def test_voxel_grid_world_to_voxel(self):
        """Test world to voxel coordinate conversion."""
        grid = VoxelGrid(bbox_min=(0.0, 0.0, 0.0), bbox_max=(10.0, 10.0, 10.0), resolution=1.0)

        # Test point at origin
        idx = grid._world_to_voxel(0.0, 0.0, 0.0)
        assert idx == (0, 0, 0)

        # Test point at center
        idx = grid._world_to_voxel(5.0, 5.0, 5.0)
        assert idx == (5, 5, 5)

        # Test point near boundary
        idx = grid._world_to_voxel(9.9, 9.9, 9.9)
        assert idx == (9, 9, 9)

    @pytest.mark.unit
    def test_voxel_grid_world_to_voxel_out_of_bounds(self):
        """Test world to voxel conversion with out-of-bounds points."""
        grid = VoxelGrid(bbox_min=(0.0, 0.0, 0.0), bbox_max=(10.0, 10.0, 10.0), resolution=1.0)

        # Point below bbox_min should clamp to (0, 0, 0)
        idx = grid._world_to_voxel(-5.0, -5.0, -5.0)
        assert idx == (0, 0, 0)

        # Point above bbox_max should clamp to (9, 9, 9)
        idx = grid._world_to_voxel(15.0, 15.0, 15.0)
        assert idx == (9, 9, 9)

    @pytest.mark.unit
    def test_voxel_grid_world_to_voxel_batch(self):
        """Test batch world to voxel coordinate conversion."""
        grid = VoxelGrid(bbox_min=(0.0, 0.0, 0.0), bbox_max=(10.0, 10.0, 10.0), resolution=1.0)

        points = np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0], [9.9, 9.9, 9.9]])

        indices = grid._world_to_voxel_batch(points)

        assert indices.shape == (3, 3)
        assert np.array_equal(indices[0], [0, 0, 0])
        assert np.array_equal(indices[1], [5, 5, 5])
        assert np.array_equal(indices[2], [9, 9, 9])

    @pytest.mark.unit
    def test_voxel_grid_world_to_voxel_batch_invalid_shape(self):
        """Test batch conversion with invalid point shape."""
        grid = VoxelGrid(bbox_min=(0.0, 0.0, 0.0), bbox_max=(10.0, 10.0, 10.0), resolution=1.0)

        points = np.array([[0.0, 0.0]])  # Wrong shape

        with pytest.raises(ValueError, match="Points must be shape"):
            grid._world_to_voxel_batch(points)

    @pytest.mark.unit
    def test_voxel_grid_voxel_to_world(self):
        """Test voxel to world coordinate conversion."""
        grid = VoxelGrid(bbox_min=(0.0, 0.0, 0.0), bbox_max=(10.0, 10.0, 10.0), resolution=1.0)

        # Voxel at origin should be at center of first voxel
        world = grid._voxel_to_world(0, 0, 0)
        assert np.allclose(world, (0.5, 0.5, 0.5))

        # Voxel at (5, 5, 5) should be at center
        world = grid._voxel_to_world(5, 5, 5)
        assert np.allclose(world, (5.5, 5.5, 5.5))

    @pytest.mark.unit
    def test_voxel_grid_add_point(self):
        """Test adding a point to the voxel grid."""
        grid = VoxelGrid(bbox_min=(0.0, 0.0, 0.0), bbox_max=(10.0, 10.0, 10.0), resolution=1.0)

        grid.add_point(5.0, 5.0, 5.0, {"power": 200.0, "speed": 100.0})

        assert len(grid.voxels) == 1
        assert (5, 5, 5) in grid.voxels
        assert "power" in grid.available_signals
        assert "speed" in grid.available_signals

    @pytest.mark.unit
    def test_voxel_grid_add_point_multiple_to_same_voxel(self):
        """Test adding multiple points to the same voxel."""
        grid = VoxelGrid(bbox_min=(0.0, 0.0, 0.0), bbox_max=(10.0, 10.0, 10.0), resolution=1.0)

        grid.add_point(5.0, 5.0, 5.0, {"power": 200.0})
        grid.add_point(5.1, 5.1, 5.1, {"power": 250.0})
        grid.add_point(5.2, 5.2, 5.2, {"power": 300.0})

        # Should still be one voxel
        assert len(grid.voxels) == 1
        voxel = grid.voxels[(5, 5, 5)]
        assert voxel.count == 3
        # Values should be in list before finalize
        assert isinstance(voxel.signals["power"], list)
        assert len(voxel.signals["power"]) == 3

    @pytest.mark.unit
    def test_voxel_grid_finalize(self):
        """Test finalizing voxel grid."""
        grid = VoxelGrid(
            bbox_min=(0.0, 0.0, 0.0),
            bbox_max=(10.0, 10.0, 10.0),
            resolution=1.0,
            aggregation="mean",
        )

        grid.add_point(5.0, 5.0, 5.0, {"power": 200.0})
        grid.add_point(5.1, 5.1, 5.1, {"power": 250.0})
        grid.add_point(5.2, 5.2, 5.2, {"power": 300.0})

        grid.finalize()

        voxel = grid.voxels[(5, 5, 5)]
        # After finalize, should be aggregated value
        assert isinstance(voxel.signals["power"], (int, float, np.floating))
        assert voxel.signals["power"] == 250.0  # Mean of [200, 250, 300]

    @pytest.mark.unit
    def test_voxel_grid_finalize_max_aggregation(self):
        """Test finalizing with max aggregation."""
        grid = VoxelGrid(
            bbox_min=(0.0, 0.0, 0.0),
            bbox_max=(10.0, 10.0, 10.0),
            resolution=1.0,
            aggregation="max",
        )

        grid.add_point(5.0, 5.0, 5.0, {"power": 200.0})
        grid.add_point(5.1, 5.1, 5.1, {"power": 250.0})
        grid.add_point(5.2, 5.2, 5.2, {"power": 300.0})

        grid.finalize()

        voxel = grid.voxels[(5, 5, 5)]
        assert voxel.signals["power"] == 300.0  # Max

    @pytest.mark.unit
    def test_voxel_grid_get_voxel(self):
        """Test getting voxel data."""
        grid = VoxelGrid(bbox_min=(0.0, 0.0, 0.0), bbox_max=(10.0, 10.0, 10.0), resolution=1.0)

        grid.add_point(5.0, 5.0, 5.0, {"power": 200.0})

        voxel = grid.get_voxel(5, 5, 5)
        assert voxel is not None
        assert isinstance(voxel, VoxelData)

    @pytest.mark.unit
    def test_voxel_grid_get_voxel_nonexistent(self):
        """Test getting nonexistent voxel."""
        grid = VoxelGrid(bbox_min=(0.0, 0.0, 0.0), bbox_max=(10.0, 10.0, 10.0), resolution=1.0)

        voxel = grid.get_voxel(5, 5, 5)
        assert voxel is None

    @pytest.mark.unit
    def test_voxel_grid_get_signal_array(self):
        """Test getting signal array."""
        grid = VoxelGrid(bbox_min=(0.0, 0.0, 0.0), bbox_max=(10.0, 10.0, 10.0), resolution=1.0)

        grid.add_point(5.0, 5.0, 5.0, {"power": 200.0})
        grid.finalize()

        signal_array = grid.get_signal_array("power")

        assert signal_array.shape == tuple(grid.dims)
        assert signal_array[5, 5, 5] == 200.0
        assert np.all(signal_array[signal_array != 200.0] == 0.0)

    @pytest.mark.unit
    def test_voxel_grid_get_signal_array_default_value(self):
        """Test getting signal array with custom default value."""
        grid = VoxelGrid(bbox_min=(0.0, 0.0, 0.0), bbox_max=(10.0, 10.0, 10.0), resolution=1.0)

        grid.add_point(5.0, 5.0, 5.0, {"power": 200.0})
        grid.finalize()

        signal_array = grid.get_signal_array("power", default=-1.0)

        assert signal_array[5, 5, 5] == 200.0
        assert np.all(signal_array[signal_array != 200.0] == -1.0)

    @pytest.mark.unit
    def test_voxel_grid_get_signal_array_nonexistent_signal(self):
        """Test getting signal array for nonexistent signal."""
        grid = VoxelGrid(bbox_min=(0.0, 0.0, 0.0), bbox_max=(10.0, 10.0, 10.0), resolution=1.0)

        signal_array = grid.get_signal_array("nonexistent", default=0.0)

        assert signal_array.shape == tuple(grid.dims)
        assert np.all(signal_array == 0.0)

    @pytest.mark.unit
    def test_voxel_grid_get_bounding_box(self):
        """Test getting bounding box."""
        grid = VoxelGrid(bbox_min=(0.0, 0.0, 0.0), bbox_max=(10.0, 10.0, 10.0), resolution=1.0)

        bbox_min, bbox_max = grid.get_bounding_box()

        assert np.array_equal(bbox_min, [0.0, 0.0, 0.0])
        assert np.array_equal(bbox_max, [10.0, 10.0, 10.0])

    @pytest.mark.unit
    def test_voxel_grid_get_statistics(self):
        """Test getting grid statistics."""
        grid = VoxelGrid(bbox_min=(0.0, 0.0, 0.0), bbox_max=(10.0, 10.0, 10.0), resolution=1.0)

        grid.add_point(5.0, 5.0, 5.0, {"power": 200.0})
        grid.add_point(6.0, 6.0, 6.0, {"power": 250.0})
        grid.finalize()

        stats = grid.get_statistics()

        assert stats["dimensions"] == (10, 10, 10)
        assert stats["resolution_mm"] == 1.0
        assert stats["total_voxels"] == 1000
        assert stats["filled_voxels"] == 2
        assert "power" in stats["available_signals"]
        assert "power_mean" in stats
        assert "power_min" in stats
        assert "power_max" in stats

    @pytest.mark.unit
    def test_voxel_grid_build_voxel_grid_batch(self):
        """Test building voxel grid from pre-aggregated data."""
        grid = VoxelGrid(bbox_min=(0.0, 0.0, 0.0), bbox_max=(10.0, 10.0, 10.0), resolution=1.0)

        voxel_data = {
            (5, 5, 5): {"signals": {"power": 200.0, "speed": 100.0}, "count": 3},
            (6, 6, 6): {"signals": {"power": 250.0}, "count": 1},
        }

        grid._build_voxel_grid_batch(voxel_data)

        assert len(grid.voxels) == 2
        assert (5, 5, 5) in grid.voxels
        assert (6, 6, 6) in grid.voxels
        assert grid.voxels[(5, 5, 5)].signals["power"] == 200.0
        assert grid.voxels[(5, 5, 5)].count == 3
        assert "power" in grid.available_signals
        assert "speed" in grid.available_signals
