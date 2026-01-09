"""
Unit tests for NearestNeighborInterpolation.

Tests for nearest neighbor interpolation method.
"""

import pytest
import numpy as np
from am_qadf.signal_mapping.methods.nearest_neighbor import NearestNeighborInterpolation
from am_qadf.voxelization.voxel_grid import VoxelGrid


class TestNearestNeighborInterpolation:
    """Test suite for NearestNeighborInterpolation class."""

    @pytest.fixture
    def voxel_grid(self):
        """Create a test voxel grid."""
        return VoxelGrid(
            bbox_min=(0.0, 0.0, 0.0),
            bbox_max=(10.0, 10.0, 10.0),
            resolution=1.0,
            aggregation="mean",
        )

    @pytest.fixture
    def interpolation_method(self):
        """Create NearestNeighborInterpolation instance."""
        return NearestNeighborInterpolation()

    @pytest.mark.unit
    def test_nearest_neighbor_interpolation_creation(self):
        """Test creating NearestNeighborInterpolation."""
        method = NearestNeighborInterpolation()

        assert method is not None
        assert isinstance(method, NearestNeighborInterpolation)

    @pytest.mark.unit
    def test_interpolate_empty_points(self, interpolation_method, voxel_grid):
        """Test interpolation with empty points."""
        points = np.array([]).reshape(0, 3)
        signals = {}

        result = interpolation_method.interpolate(points, signals, voxel_grid)

        assert result is voxel_grid
        assert len(voxel_grid.voxels) == 0

    @pytest.mark.unit
    def test_interpolate_single_point(self, interpolation_method, voxel_grid):
        """Test interpolation with single point."""
        points = np.array([[5.0, 5.0, 5.0]])
        signals = {"power": np.array([200.0])}

        result = interpolation_method.interpolate(points, signals, voxel_grid)

        assert result is voxel_grid
        assert len(voxel_grid.voxels) == 1
        assert (5, 5, 5) in voxel_grid.voxels
        assert voxel_grid.voxels[(5, 5, 5)].signals["power"] == 200.0

    @pytest.mark.unit
    def test_interpolate_multiple_points_same_voxel(self, interpolation_method, voxel_grid):
        """Test interpolation with multiple points in same voxel."""
        points = np.array([[5.0, 5.0, 5.0], [5.1, 5.1, 5.1], [5.2, 5.2, 5.2]])
        signals = {"power": np.array([200.0, 250.0, 300.0])}

        result = interpolation_method.interpolate(points, signals, voxel_grid)

        assert result is voxel_grid
        assert len(voxel_grid.voxels) == 1
        voxel = voxel_grid.voxels[(5, 5, 5)]
        # Should be mean of [200, 250, 300] = 250.0
        assert voxel.signals["power"] == 250.0
        assert voxel.count == 3

    @pytest.mark.unit
    def test_interpolate_multiple_points_different_voxels(self, interpolation_method, voxel_grid):
        """Test interpolation with points in different voxels."""
        points = np.array([[1.0, 1.0, 1.0], [5.0, 5.0, 5.0], [9.0, 9.0, 9.0]])
        signals = {"power": np.array([100.0, 200.0, 300.0])}

        result = interpolation_method.interpolate(points, signals, voxel_grid)

        assert result is voxel_grid
        assert len(voxel_grid.voxels) == 3
        assert (1, 1, 1) in voxel_grid.voxels
        assert (5, 5, 5) in voxel_grid.voxels
        assert (9, 9, 9) in voxel_grid.voxels
        assert voxel_grid.voxels[(1, 1, 1)].signals["power"] == 100.0
        assert voxel_grid.voxels[(5, 5, 5)].signals["power"] == 200.0
        assert voxel_grid.voxels[(9, 9, 9)].signals["power"] == 300.0

    @pytest.mark.unit
    def test_interpolate_multiple_signals(self, interpolation_method, voxel_grid):
        """Test interpolation with multiple signals."""
        points = np.array([[5.0, 5.0, 5.0], [6.0, 6.0, 6.0]])
        signals = {"power": np.array([200.0, 250.0]), "speed": np.array([100.0, 150.0])}

        result = interpolation_method.interpolate(points, signals, voxel_grid)

        assert result is voxel_grid
        assert len(voxel_grid.voxels) == 2
        assert "power" in voxel_grid.available_signals
        assert "speed" in voxel_grid.available_signals

    @pytest.mark.unit
    def test_interpolate_aggregation_max(self, interpolation_method):
        """Test interpolation with max aggregation."""
        voxel_grid = VoxelGrid(
            bbox_min=(0.0, 0.0, 0.0),
            bbox_max=(10.0, 10.0, 10.0),
            resolution=1.0,
            aggregation="max",
        )

        points = np.array([[5.0, 5.0, 5.0], [5.1, 5.1, 5.1], [5.2, 5.2, 5.2]])
        signals = {"power": np.array([200.0, 250.0, 300.0])}

        result = interpolation_method.interpolate(points, signals, voxel_grid)

        voxel = voxel_grid.voxels[(5, 5, 5)]
        assert voxel.signals["power"] == 300.0  # Max

    @pytest.mark.unit
    def test_interpolate_aggregation_min(self, interpolation_method):
        """Test interpolation with min aggregation."""
        voxel_grid = VoxelGrid(
            bbox_min=(0.0, 0.0, 0.0),
            bbox_max=(10.0, 10.0, 10.0),
            resolution=1.0,
            aggregation="min",
        )

        points = np.array([[5.0, 5.0, 5.0], [5.1, 5.1, 5.1], [5.2, 5.2, 5.2]])
        signals = {"power": np.array([200.0, 250.0, 300.0])}

        result = interpolation_method.interpolate(points, signals, voxel_grid)

        voxel = voxel_grid.voxels[(5, 5, 5)]
        assert voxel.signals["power"] == 200.0  # Min

    @pytest.mark.unit
    def test_interpolate_aggregation_sum(self, interpolation_method):
        """Test interpolation with sum aggregation."""
        voxel_grid = VoxelGrid(
            bbox_min=(0.0, 0.0, 0.0),
            bbox_max=(10.0, 10.0, 10.0),
            resolution=1.0,
            aggregation="sum",
        )

        points = np.array([[5.0, 5.0, 5.0], [5.1, 5.1, 5.1], [5.2, 5.2, 5.2]])
        signals = {"power": np.array([200.0, 250.0, 300.0])}

        result = interpolation_method.interpolate(points, signals, voxel_grid)

        voxel = voxel_grid.voxels[(5, 5, 5)]
        assert voxel.signals["power"] == 750.0  # Sum

    @pytest.mark.unit
    def test_interpolate_points_out_of_bounds(self, interpolation_method, voxel_grid):
        """Test interpolation with points outside bounding box."""
        points = np.array([[-5.0, -5.0, -5.0], [15.0, 15.0, 15.0]])  # Below bbox_min  # Above bbox_max
        signals = {"power": np.array([100.0, 200.0])}

        result = interpolation_method.interpolate(points, signals, voxel_grid)

        # Points should be clamped to valid voxels
        assert len(voxel_grid.voxels) > 0

    @pytest.mark.unit
    def test_interpolate_signal_length_mismatch(self, interpolation_method, voxel_grid):
        """Test interpolation with signal length mismatch."""
        points = np.array([[5.0, 5.0, 5.0], [6.0, 6.0, 6.0]])
        signals = {
            "power": np.array([200.0]),  # Wrong length
            "speed": np.array([100.0, 150.0]),  # Correct length
        }

        result = interpolation_method.interpolate(points, signals, voxel_grid)

        # Should only process signals with correct length
        assert "speed" in voxel_grid.available_signals
        # 'power' might not be added due to length mismatch
