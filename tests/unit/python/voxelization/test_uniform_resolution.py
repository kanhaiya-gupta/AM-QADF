"""
Unit tests for am_qadf.voxelization.uniform_resolution.

Tests VoxelGrid (uniform resolution voxel grid) from uniform_resolution.py.
Requires am_qadf_native; tests are skipped when native bindings are not available.
"""

import numpy as np
import pytest

pytest.importorskip("am_qadf_native")

from am_qadf.voxelization.uniform_resolution import VoxelGrid


class TestVoxelGridInit:
    """Tests for VoxelGrid.__init__."""

    def test_init_valid(self):
        grid = VoxelGrid(
            bbox_min=(0.0, 0.0, 0.0),
            bbox_max=(10.0, 10.0, 10.0),
            resolution=1.0,
        )
        assert grid.resolution == 1.0
        assert grid.aggregation == "mean"
        np.testing.assert_array_almost_equal(grid.bbox_min, [0, 0, 0])
        np.testing.assert_array_almost_equal(grid.bbox_max, [10, 10, 10])
        assert grid.dims is not None
        assert len(grid.available_signals) == 0

    def test_init_with_aggregation(self):
        grid = VoxelGrid(
            bbox_min=(0, 0, 0),
            bbox_max=(5, 5, 5),
            resolution=0.5,
            aggregation="max",
        )
        assert grid.aggregation == "max"

    def test_init_resolution_zero_raises(self):
        with pytest.raises(ValueError, match="Resolution must be greater than 0"):
            VoxelGrid(
                bbox_min=(0, 0, 0),
                bbox_max=(10, 10, 10),
                resolution=0.0,
            )

    def test_init_resolution_negative_raises(self):
        with pytest.raises(ValueError, match="Resolution must be greater than 0"):
            VoxelGrid(
                bbox_min=(0, 0, 0),
                bbox_max=(10, 10, 10),
                resolution=-1.0,
            )

    def test_init_bbox_max_not_greater_raises(self):
        with pytest.raises(ValueError, match="bbox_max must be greater than bbox_min"):
            VoxelGrid(
                bbox_min=(10, 10, 10),
                bbox_max=(10, 10, 10),
                resolution=1.0,
            )


class TestVoxelGridWorldToVoxel:
    """Tests for _world_to_voxel and _world_to_voxel_batch."""

    def test_world_to_voxel_origin(self):
        grid = VoxelGrid(
            bbox_min=(0.0, 0.0, 0.0),
            bbox_max=(10.0, 10.0, 10.0),
            resolution=1.0,
        )
        i, j, k = grid._world_to_voxel(0.0, 0.0, 0.0)
        assert (i, j, k) == (0, 0, 0)

    def test_world_to_voxel_inside(self):
        grid = VoxelGrid(
            bbox_min=(0.0, 0.0, 0.0),
            bbox_max=(10.0, 10.0, 10.0),
            resolution=1.0,
        )
        i, j, k = grid._world_to_voxel(3.5, 4.2, 1.8)
        assert (i, j, k) == (3, 4, 1)

    def test_world_to_voxel_batch(self):
        grid = VoxelGrid(
            bbox_min=(0.0, 0.0, 0.0),
            bbox_max=(10.0, 10.0, 10.0),
            resolution=1.0,
        )
        points = np.array([[0, 0, 0], [1, 1, 1], [9, 9, 9]], dtype=np.float64)
        indices = grid._world_to_voxel_batch(points)
        assert indices.shape == (3, 3)
        np.testing.assert_array_equal(indices[0], [0, 0, 0])
        np.testing.assert_array_equal(indices[1], [1, 1, 1])
        np.testing.assert_array_equal(indices[2], [9, 9, 9])

    def test_world_to_voxel_batch_wrong_shape_raises(self):
        grid = VoxelGrid(
            bbox_min=(0, 0, 0),
            bbox_max=(10, 10, 10),
            resolution=1.0,
        )
        points = np.array([[0, 0], [1, 1]])
        with pytest.raises(ValueError, match="Points must be shape"):
            grid._world_to_voxel_batch(points)


class TestVoxelGridAddPointAndFinalize:
    """Tests for add_point and finalize."""

    def test_add_point_single(self):
        grid = VoxelGrid(
            bbox_min=(0.0, 0.0, 0.0),
            bbox_max=(10.0, 10.0, 10.0),
            resolution=1.0,
        )
        grid.add_point(1.0, 1.0, 1.0, {"temperature": 100.0})
        assert "temperature" in grid.available_signals

    def test_add_point_multiple_signals(self):
        grid = VoxelGrid(
            bbox_min=(0.0, 0.0, 0.0),
            bbox_max=(10.0, 10.0, 10.0),
            resolution=1.0,
        )
        grid.add_point(2.0, 2.0, 2.0, {"temp": 50.0, "power": 200.0})
        assert "temp" in grid.available_signals
        assert "power" in grid.available_signals

    def test_finalize_after_add_point(self):
        grid = VoxelGrid(
            bbox_min=(0.0, 0.0, 0.0),
            bbox_max=(10.0, 10.0, 10.0),
            resolution=1.0,
        )
        grid.add_point(0.5, 0.5, 0.5, {"s": 1.0})
        grid.finalize()
        voxel = grid.get_voxel(0, 0, 0)
        assert voxel is not None
        assert "s" in voxel.signals
        assert voxel.signals["s"] == 1.0


class TestVoxelGridGetVoxel:
    """Tests for get_voxel."""

    def test_get_voxel_empty_returns_none(self):
        grid = VoxelGrid(
            bbox_min=(0, 0, 0),
            bbox_max=(10, 10, 10),
            resolution=1.0,
        )
        grid.finalize()
        v = grid.get_voxel(0, 0, 0)
        assert v is None

    def test_get_voxel_after_add_and_finalize(self):
        grid = VoxelGrid(
            bbox_min=(0, 0, 0),
            bbox_max=(10, 10, 10),
            resolution=1.0,
        )
        grid.add_point(0.5, 0.5, 0.5, {"a": 42.0})
        grid.finalize()
        v = grid.get_voxel(0, 0, 0)
        assert v is not None
        assert v.signals["a"] == 42.0


class TestVoxelGridGetGridAndSignalArray:
    """Tests for get_grid and get_signal_array."""

    def test_get_grid_none_for_unknown_signal(self):
        grid = VoxelGrid(
            bbox_min=(0, 0, 0),
            bbox_max=(10, 10, 10),
            resolution=1.0,
        )
        assert grid.get_grid("nonexistent") is None

    def test_get_grid_after_add_finalize(self):
        grid = VoxelGrid(
            bbox_min=(0, 0, 0),
            bbox_max=(10, 10, 10),
            resolution=1.0,
        )
        grid.add_point(0.5, 0.5, 0.5, {"s": 1.0})
        grid.finalize()
        g = grid.get_grid("s")
        assert g is not None

    def test_get_signal_array_unknown_returns_dense_default(self):
        grid = VoxelGrid(
            bbox_min=(0, 0, 0),
            bbox_max=(10, 10, 10),
            resolution=1.0,
        )
        arr = grid.get_signal_array("unknown", default=0.0)
        assert arr.shape == tuple(grid.dims)
        assert arr.dtype == np.float32
        assert np.all(arr == 0.0)


class TestVoxelGridBoundingBoxAndStatistics:
    """Tests for get_bounding_box and get_statistics."""

    def test_get_bounding_box(self):
        grid = VoxelGrid(
            bbox_min=(1.0, 2.0, 3.0),
            bbox_max=(11.0, 12.0, 13.0),
            resolution=1.0,
        )
        bmin, bmax = grid.get_bounding_box()
        np.testing.assert_array_almost_equal(bmin, [1, 2, 3])
        np.testing.assert_array_almost_equal(bmax, [11, 12, 13])

    def test_get_statistics_empty_grid(self):
        grid = VoxelGrid(
            bbox_min=(0, 0, 0),
            bbox_max=(10, 10, 10),
            resolution=1.0,
        )
        grid.finalize()
        stats = grid.get_statistics()
        assert stats["dimensions"] == (10, 10, 10)
        assert stats["resolution_mm"] == 1.0
        assert stats["available_signals"] == []
        assert stats["filled_voxels"] == 0

    def test_get_statistics_after_add_finalize(self):
        grid = VoxelGrid(
            bbox_min=(0, 0, 0),
            bbox_max=(10, 10, 10),
            resolution=1.0,
        )
        grid.add_point(0.5, 0.5, 0.5, {"t": 100.0})
        grid.finalize()
        stats = grid.get_statistics()
        assert "t" in stats["available_signals"]
        assert stats["filled_voxels"] >= 1
