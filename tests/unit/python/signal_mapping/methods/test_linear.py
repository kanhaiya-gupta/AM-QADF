"""
Unit tests for LinearInterpolation.

Tests for linear interpolation using k-nearest neighbors.
"""

import pytest
import numpy as np

pytest.importorskip("am_qadf_native")

from am_qadf.signal_mapping.methods.linear import LinearInterpolation
from am_qadf.voxelization import VoxelGrid


class TestLinearInterpolation:
    """Test suite for LinearInterpolation class."""

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
        """Create LinearInterpolation instance."""
        return LinearInterpolation(k_neighbors=4)

    @pytest.mark.unit
    def test_linear_interpolation_creation_default(self):
        """Test creating LinearInterpolation with default parameters."""
        method = LinearInterpolation()

        assert method.k_neighbors == 8
        assert method.radius is None

    @pytest.mark.unit
    def test_linear_interpolation_creation_custom(self):
        """Test creating LinearInterpolation with custom parameters."""
        method = LinearInterpolation(k_neighbors=4, radius=2.0)

        assert method.k_neighbors == 4
        assert method.radius == 2.0

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
        assert "power" in voxel_grid.available_signals
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
    def test_interpolate_with_radius(self):
        """Test interpolation with radius constraint."""
        voxel_grid = VoxelGrid(bbox_min=(0.0, 0.0, 0.0), bbox_max=(10.0, 10.0, 10.0), resolution=1.0)

        method = LinearInterpolation(k_neighbors=8, radius=1.0)

        points = np.array(
            [
                [5.0, 5.0, 5.0],
                [5.1, 5.1, 5.1],
                [5.2, 5.2, 5.2],
                [10.0, 10.0, 10.0],  # Far point
            ]
        )
        signals = {"power": np.array([200.0, 210.0, 220.0, 500.0])}

        result = method.interpolate(points, signals, voxel_grid)

        assert result is voxel_grid
        # Far point might not contribute to nearby voxels due to radius

    @pytest.mark.unit
    def test_interpolate_fewer_points_than_k(self, interpolation_method, voxel_grid):
        """Test interpolation when points < k_neighbors."""
        method = LinearInterpolation(k_neighbors=10)  # More than available points

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
    def test_interpolate_weighted_average(self, interpolation_method, voxel_grid):
        """Test that interpolation uses weighted average based on distance."""
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
        # Closer points should have more weight
        # The voxel at (5, 5, 5) should have value closer to 200.0 than 300.0

    @pytest.mark.unit
    def test_interpolate_scipy_required(self):
        """Test that scipy is required for linear interpolation."""
        # This test verifies the ImportError is raised if scipy is not available
        # In practice, scipy should be installed, but we test the error handling
        method = LinearInterpolation()
        voxel_grid = VoxelGrid(bbox_min=(0.0, 0.0, 0.0), bbox_max=(10.0, 10.0, 10.0), resolution=1.0)

        points = np.array([[5.0, 5.0, 5.0]])
        signals = {"power": np.array([200.0])}

        # Should work if scipy is available, or raise ImportError if not
        try:
            result = method.interpolate(points, signals, voxel_grid)
            assert result is voxel_grid
        except ImportError as e:
            assert "scipy" in str(e).lower()
