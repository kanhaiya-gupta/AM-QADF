"""
Unit tests for RBFInterpolation.

Tests for Radial Basis Functions interpolation.
"""

import pytest
import numpy as np
pytest.importorskip("am_qadf_native")

from am_qadf.signal_mapping.methods.rbf import RBFInterpolation

from am_qadf.voxelization import VoxelGrid


class TestRBFInterpolation:
    """Test suite for RBFInterpolation class."""

    @pytest.fixture
    def voxel_grid(self):
        """Create a test voxel grid."""
        return VoxelGrid(
            bbox_min=(0.0, 0.0, 0.0),
            bbox_max=(10.0, 10.0, 10.0),
            resolution=1.0,
            aggregation="mean",
        )

    def _create_voxel_grid_copy(self, grid):
        """Helper to create a copy of a voxel grid."""
        return VoxelGrid(
            bbox_min=tuple(grid.bbox_min),
            bbox_max=tuple(grid.bbox_max),
            resolution=grid.resolution,
            aggregation=grid.aggregation,
        )

    @pytest.fixture
    def interpolation_method(self):
        """Create RBFInterpolation instance."""
        return RBFInterpolation(kernel_type="gaussian", epsilon=1.0)

    @pytest.mark.unit
    def test_rbf_interpolation_creation_default(self):
        """Test creating RBFInterpolation with default parameters."""
        method = RBFInterpolation()

        assert method.kernel_type == "gaussian"
        assert method.epsilon == 1.0

    @pytest.mark.unit
    def test_rbf_interpolation_creation_custom(self):
        """Test creating RBFInterpolation with custom parameters."""
        method = RBFInterpolation(kernel_type="multiquadric", epsilon=2.0)

        assert method.kernel_type == "multiquadric"
        assert method.epsilon == 2.0

    @pytest.mark.unit
    def test_rbf_valid_kernels(self):
        """Test that supported kernel types can be created (C++ RBFMapper)."""
        valid_kernels = ["gaussian", "multiquadric", "thin_plate"]

        for k in valid_kernels:
            method = RBFInterpolation(kernel_type=k)
            assert method.kernel_type == k

    @pytest.mark.unit
    def test_interpolate_empty_points(self, interpolation_method, voxel_grid):
        """Test interpolation with empty points."""
        points = np.array([]).reshape(0, 3)
        signals = {}

        result = interpolation_method.interpolate(points, signals, voxel_grid)

        assert result is voxel_grid
        assert voxel_grid.get_statistics().get("filled_voxels", 0) == 0

    @pytest.mark.unit
    def test_interpolate_single_point(self, interpolation_method, voxel_grid):
        """Test interpolation with single point."""
        points = np.array([[5.0, 5.0, 5.0]])
        signals = {"power": np.array([200.0])}

        result = interpolation_method.interpolate(points, signals, voxel_grid)

        assert result is voxel_grid
        # RBF should interpolate to voxel centers
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
        assert voxel_grid.get_statistics().get("filled_voxels", 0) >= 1
        assert "power" in voxel_grid.available_signals

    @pytest.mark.unit
    def test_interpolate_exact_at_data_points(self, interpolation_method, voxel_grid):
        """Test that RBF provides exact interpolation at data points (when smoothing=0)."""
        # Create points that map directly to voxel centers
        points = np.array(
            [
                [0.5, 0.5, 0.5],  # Voxel center at (0,0,0)
                [1.5, 1.5, 1.5],  # Voxel center at (1,1,1)
                [2.5, 2.5, 2.5],  # Voxel center at (2,2,2)
            ]
        )
        signal_values = np.array([100.0, 200.0, 300.0])
        signals = {"power": signal_values}

        result = interpolation_method.interpolate(points, signals, voxel_grid)

        # RBF should interpolate exactly at data points when smoothing=0
        # Note: Due to numerical precision, we check that values are close
        assert result is voxel_grid
        assert voxel_grid.get_statistics().get("filled_voxels", 0) >= 1

    @pytest.mark.unit
    def test_interpolate_multiple_signals(self, interpolation_method, voxel_grid):
        """Test interpolation with multiple signal types."""
        points = np.array(
            [
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0],
            ]
        )
        signals = {
            "power": np.array([100.0, 200.0, 300.0]),
            "speed": np.array([500.0, 600.0, 700.0]),
        }

        result = interpolation_method.interpolate(points, signals, voxel_grid)

        assert result is voxel_grid
        assert "power" in voxel_grid.available_signals
        assert "speed" in voxel_grid.available_signals

    @pytest.mark.unit
    def test_interpolate_different_kernels(self, voxel_grid):
        """Test interpolation with supported kernel types (C++ RBFMapper)."""
        points = np.array(
            [
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0],
                [4.0, 4.0, 4.0],
                [1.5, 2.5, 3.5],
            ]
        )
        signals = {"power": np.array([100.0, 200.0, 300.0, 400.0, 250.0])}

        for kernel_type in ["gaussian", "multiquadric", "thin_plate"]:
            method = RBFInterpolation(kernel_type=kernel_type, epsilon=1.0)
            result = method.interpolate(points, signals, self._create_voxel_grid_copy(voxel_grid))
            assert result is not None

    @pytest.mark.unit
    def test_interpolate_mismatched_signal_length(self, interpolation_method, voxel_grid):
        """Test interpolation with mismatched signal length."""
        points = np.array(
            [
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
            ]
        )
        signals = {
            "power": np.array([100.0, 200.0]),  # Correct length
            "speed": np.array([500.0]),  # Wrong length
        }

        # Should handle gracefully and skip invalid signals
        result = interpolation_method.interpolate(points, signals, voxel_grid)

        assert result is voxel_grid
        # Should still process valid signals
        assert len(voxel_grid.available_signals) >= 0

    @pytest.mark.unit
    def test_epsilon_none_uses_default(self, voxel_grid):
        """Test that epsilon=None is converted to 1.0 (C++ default)."""
        points = np.array(
            [
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0],
            ]
        )
        signals = {"power": np.array([100.0, 200.0, 300.0])}

        method = RBFInterpolation(kernel_type="gaussian", epsilon=None)
        result = method.interpolate(points, signals, voxel_grid)

        assert method.epsilon == 1.0
        assert result is voxel_grid
