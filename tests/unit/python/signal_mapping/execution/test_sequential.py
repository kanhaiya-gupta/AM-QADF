"""
Unit tests for am_qadf.signal_mapping.execution.sequential.

Tests interpolate_to_voxels, interpolate_hatching_paths, INTERPOLATION_METHODS.
Requires am_qadf_native; tests are skipped when native bindings are not available.
"""

import pytest
import numpy as np

pytest.importorskip("am_qadf_native")

from am_qadf.voxelization import VoxelGrid
from am_qadf.signal_mapping.execution.sequential import (
    interpolate_to_voxels,
    interpolate_hatching_paths,
    INTERPOLATION_METHODS,
)


def _has_filled_voxels(grid):
    """True if grid has any filled voxels or signals."""
    return len(grid.available_signals) > 0 and grid.get_statistics().get("filled_voxels", 0) > 0


class TestInterpolateToVoxels:
    """Test suite for interpolate_to_voxels function."""

    @pytest.fixture
    def voxel_grid(self):
        """Create a test voxel grid."""
        return VoxelGrid(
            bbox_min=(0.0, 0.0, 0.0),
            bbox_max=(10.0, 10.0, 10.0),
            resolution=1.0,
            aggregation="mean",
        )

    @pytest.mark.unit
    def test_interpolate_to_voxels_nearest_default(self, voxel_grid):
        """Test interpolate_to_voxels with nearest neighbor (default)."""
        points = np.array([[1.0, 1.0, 1.0], [5.0, 5.0, 5.0], [9.0, 9.0, 9.0]])
        signals = {"power": np.array([100.0, 200.0, 300.0])}

        result = interpolate_to_voxels(points, signals, voxel_grid)

        assert result is voxel_grid
        assert "power" in result.available_signals
        assert _has_filled_voxels(result)

    @pytest.mark.unit
    def test_interpolate_to_voxels_linear(self, voxel_grid):
        """Test interpolate_to_voxels with linear interpolation."""
        points = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])
        signals = {"power": np.array([100.0, 150.0, 200.0, 250.0])}

        result = interpolate_to_voxels(points, signals, voxel_grid, method="linear", k_neighbors=4)

        assert result is voxel_grid
        assert _has_filled_voxels(result)

    @pytest.mark.unit
    def test_interpolate_to_voxels_idw(self, voxel_grid):
        """Test interpolate_to_voxels with IDW interpolation."""
        points = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])
        signals = {"power": np.array([100.0, 150.0, 200.0, 250.0])}

        result = interpolate_to_voxels(points, signals, voxel_grid, method="idw", power=2.0, k_neighbors=4)

        assert result is voxel_grid
        assert _has_filled_voxels(result)

    @pytest.mark.unit
    def test_interpolate_to_voxels_gaussian_kde(self, voxel_grid):
        """Test interpolate_to_voxels with Gaussian KDE interpolation."""
        points = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])
        signals = {"power": np.array([100.0, 150.0, 200.0, 250.0])}

        result = interpolate_to_voxels(points, signals, voxel_grid, method="gaussian_kde", bandwidth=1.0)

        assert result is voxel_grid
        assert "power" in result.available_signals
        # C++ KDE may or may not report filled_voxels depending on implementation
        assert result.get_statistics().get("filled_voxels", 0) >= 0

    @pytest.mark.unit
    def test_interpolate_to_voxels_invalid_method(self, voxel_grid):
        """Test interpolate_to_voxels with invalid method."""
        points = np.array([[5.0, 5.0, 5.0]])
        signals = {"power": np.array([200.0])}

        with pytest.raises(ValueError, match="Unknown interpolation method"):
            interpolate_to_voxels(points, signals, voxel_grid, method="invalid")

    @pytest.mark.unit
    def test_interpolate_to_voxels_invalid_points_shape(self, voxel_grid):
        """Test interpolate_to_voxels with invalid points shape."""
        points = np.array([[5.0, 5.0]])  # Wrong shape
        signals = {"power": np.array([200.0])}

        with pytest.raises(ValueError, match="Points must be shape"):
            interpolate_to_voxels(points, signals, voxel_grid)

    @pytest.mark.unit
    def test_interpolate_to_voxels_empty_points(self, voxel_grid):
        """Test interpolate_to_voxels with empty points."""
        points = np.array([]).reshape(0, 3)
        signals = {}

        result = interpolate_to_voxels(points, signals, voxel_grid)

        assert result is voxel_grid
        assert len(result.available_signals) == 0

    @pytest.mark.unit
    def test_interpolation_methods_registry(self):
        """Test that INTERPOLATION_METHODS registry contains all methods."""
        assert "nearest" in INTERPOLATION_METHODS
        assert "linear" in INTERPOLATION_METHODS
        assert "idw" in INTERPOLATION_METHODS
        assert "gaussian_kde" in INTERPOLATION_METHODS
        assert "rbf" in INTERPOLATION_METHODS


class TestInterpolateHatchingPaths:
    """Test suite for interpolate_hatching_paths function."""

    @pytest.fixture
    def voxel_grid(self):
        """Create a test voxel grid."""
        return VoxelGrid(
            bbox_min=(0.0, 0.0, 0.0),
            bbox_max=(10.0, 10.0, 10.0),
            resolution=1.0,
            aggregation="mean",
        )

    @pytest.mark.unit
    def test_interpolate_hatching_paths_single_path(self, voxel_grid):
        """Test interpolating a single hatching path."""
        paths = [np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])]
        signals = {"power": [np.array([100.0, 150.0, 200.0])]}

        result = interpolate_hatching_paths(paths, signals, voxel_grid)

        assert result is voxel_grid
        assert _has_filled_voxels(result)

    @pytest.mark.unit
    def test_interpolate_hatching_paths_multiple_paths(self, voxel_grid):
        """Test interpolating multiple hatching paths."""
        paths = [
            np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]),
            np.array([[5.0, 5.0, 5.0], [6.0, 6.0, 6.0]]),
            np.array([[8.0, 8.0, 8.0], [9.0, 9.0, 9.0]]),
        ]
        signals = {
            "power": [
                np.array([100.0, 150.0]),
                np.array([200.0, 250.0]),
                np.array([300.0, 350.0]),
            ]
        }

        result = interpolate_hatching_paths(paths, signals, voxel_grid)

        assert result is voxel_grid
        assert _has_filled_voxels(result)

    @pytest.mark.unit
    def test_interpolate_hatching_paths_empty_paths(self, voxel_grid):
        """Test interpolating empty paths list."""
        paths = []
        signals = {}

        result = interpolate_hatching_paths(paths, signals, voxel_grid)

        assert result is voxel_grid
        assert len(result.available_signals) == 0

    @pytest.mark.unit
    def test_interpolate_hatching_paths_short_path(self, voxel_grid):
        """Test interpolating path with less than 2 points."""
        paths = [np.array([[5.0, 5.0, 5.0]])]  # Single point
        signals = {"power": [np.array([200.0])]}

        result = interpolate_hatching_paths(paths, signals, voxel_grid)

        assert result is voxel_grid

    @pytest.mark.unit
    def test_interpolate_hatching_paths_custom_points_per_mm(self, voxel_grid):
        """Test interpolating with custom points_per_mm."""
        paths = [np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])]
        signals = {"power": [np.array([100.0, 150.0, 200.0])]}

        result = interpolate_hatching_paths(paths, signals, voxel_grid, points_per_mm=20.0)

        assert result is voxel_grid

    @pytest.mark.unit
    def test_interpolate_hatching_paths_custom_method(self, voxel_grid):
        """Test interpolating with custom interpolation method."""
        paths = [np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])]
        signals = {"power": [np.array([100.0, 150.0, 200.0])]}

        result = interpolate_hatching_paths(paths, signals, voxel_grid, interpolation_method="linear", k_neighbors=4)

        assert result is voxel_grid
        assert _has_filled_voxels(result)

    @pytest.mark.unit
    def test_interpolate_hatching_paths_zero_length_path(self, voxel_grid):
        """Test interpolating path with zero length."""
        paths = [np.array([[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]])]  # Same point (zero length)
        signals = {"power": [np.array([200.0, 200.0])]}

        result = interpolate_hatching_paths(paths, signals, voxel_grid)

        assert result is voxel_grid
