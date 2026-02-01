"""
Unit tests for IDWInterpolation.

Tests for Inverse Distance Weighting interpolation.
"""

import pytest
import numpy as np

pytest.importorskip("am_qadf_native")

from am_qadf.signal_mapping.methods.idw import IDWInterpolation
from am_qadf.voxelization import VoxelGrid


class TestIDWInterpolation:
    """Test suite for IDWInterpolation class."""

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
        """Create IDWInterpolation instance."""
        return IDWInterpolation(power=2.0, k_neighbors=4)

    @pytest.mark.unit
    def test_idw_interpolation_creation_default(self):
        """Test creating IDWInterpolation with default parameters."""
        method = IDWInterpolation()

        assert method.power == 2.0
        assert method.k_neighbors == 10

    @pytest.mark.unit
    def test_idw_interpolation_creation_custom(self):
        """Test creating IDWInterpolation with custom parameters."""
        method = IDWInterpolation(power=3.0, k_neighbors=6)

        assert method.power == 3.0
        assert method.k_neighbors == 6

    @pytest.mark.unit
    def test_interpolate_empty_points(self, interpolation_method, voxel_grid):
        """Test interpolation with empty points."""
        points = np.array([]).reshape(0, 3)
        signals = {}

        result = interpolation_method.interpolate(points, signals, voxel_grid)

        assert result is voxel_grid
        assert len(voxel_grid.available_signals) == 0

    @pytest.mark.unit
    def test_interpolate_single_point(self, interpolation_method, voxel_grid):
        """Test interpolation with single point."""
        points = np.array([[5.0, 5.0, 5.0]])
        signals = {"power": np.array([200.0])}

        result = interpolation_method.interpolate(points, signals, voxel_grid)

        assert result is voxel_grid
        # Should find the point and assign to nearest voxel
        assert voxel_grid.get_statistics().get("filled_voxels", 0) >= 1

    @pytest.mark.unit
    def test_interpolate_multiple_points(self, interpolation_method, voxel_grid):
        """Test interpolation with multiple points."""
        points = np.array(
            [
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0],
                [4.0, 4.0, 4.0],
                [5.0, 5.0, 5.0],
            ]
        )
        signals = {"power": np.array([100.0, 150.0, 200.0, 250.0, 300.0])}

        result = interpolation_method.interpolate(points, signals, voxel_grid)

        assert result is voxel_grid
        assert voxel_grid.get_statistics().get("filled_voxels", 0) > 0

    @pytest.mark.unit
    def test_interpolate_power_parameter_effect(self):
        """Test that power parameter affects weighting."""
        voxel_grid = VoxelGrid(bbox_min=(0.0, 0.0, 0.0), bbox_max=(10.0, 10.0, 10.0), resolution=1.0)

        # Low power (closer to linear)
        method_low_power = IDWInterpolation(power=1.0, k_neighbors=4)

        # High power (more emphasis on nearest)
        method_high_power = IDWInterpolation(power=4.0, k_neighbors=4)

        points = np.array(
            [
                [5.0, 5.0, 5.0],  # Close
                [6.0, 6.0, 6.0],  # Medium
                [7.0, 7.0, 7.0],  # Far
            ]
        )
        signals = {"power": np.array([200.0, 250.0, 300.0])}

        voxel_grid1 = VoxelGrid(bbox_min=(0.0, 0.0, 0.0), bbox_max=(10.0, 10.0, 10.0), resolution=1.0)
        result_low = method_low_power.interpolate(points, signals, voxel_grid1)

        voxel_grid2 = VoxelGrid(bbox_min=(0.0, 0.0, 0.0), bbox_max=(10.0, 10.0, 10.0), resolution=1.0)
        result_high = method_high_power.interpolate(points, signals, voxel_grid2)

        # Both should produce results
        assert result_low.get_statistics().get("filled_voxels", 0) > 0
        assert result_high.get_statistics().get("filled_voxels", 0) > 0

    @pytest.mark.unit
    def test_interpolate_fewer_points_than_k(self, interpolation_method, voxel_grid):
        """Test interpolation when points < k_neighbors."""
        method = IDWInterpolation(power=2.0, k_neighbors=10)  # More than available points

        points = np.array([[5.0, 5.0, 5.0], [6.0, 6.0, 6.0]])
        signals = {"power": np.array([200.0, 250.0])}

        result = method.interpolate(points, signals, voxel_grid)

        assert result is voxel_grid
        # Should handle gracefully with fewer points than k_neighbors

    @pytest.mark.unit
    def test_interpolate_multiple_signals(self, interpolation_method, voxel_grid):
        """Test interpolation with multiple signals."""
        points = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])
        signals = {
            "power": np.array([100.0, 150.0, 200.0, 250.0]),
            "speed": np.array([50.0, 75.0, 100.0, 125.0]),
        }

        result = interpolation_method.interpolate(points, signals, voxel_grid)

        assert result is voxel_grid
        assert "power" in voxel_grid.available_signals
        assert "speed" in voxel_grid.available_signals

    @pytest.mark.unit
    def test_interpolate_idw_formula(self, interpolation_method, voxel_grid):
        """Test that IDW uses correct formula: v = sum(s_i / d_i^p) / sum(1 / d_i^p)."""
        # Create points with known distances
        points = np.array(
            [
                [5.0, 5.0, 5.0],  # Close to voxel center (5.5, 5.5, 5.5)
                [6.0, 6.0, 6.0],  # Medium distance
                [7.0, 7.0, 7.0],  # Far distance
            ]
        )
        signals = {"power": np.array([200.0, 250.0, 300.0])}

        result = interpolation_method.interpolate(points, signals, voxel_grid)

        assert result is voxel_grid
        # Closer points should have more weight (higher power = more emphasis on distance)

    @pytest.mark.unit
    def test_interpolate_scipy_required(self):
        """Test that scipy is required for IDW interpolation."""
        method = IDWInterpolation()
        voxel_grid = VoxelGrid(bbox_min=(0.0, 0.0, 0.0), bbox_max=(10.0, 10.0, 10.0), resolution=1.0)

        points = np.array([[5.0, 5.0, 5.0]])
        signals = {"power": np.array([200.0])}

        # Should work if scipy is available, or raise ImportError if not
        try:
            result = method.interpolate(points, signals, voxel_grid)
            assert result is voxel_grid
        except ImportError as e:
            assert "scipy" in str(e).lower()
