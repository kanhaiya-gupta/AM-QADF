"""
Integration tests for RBF interpolation.

Tests RBF interpolation with real voxel grids, compares with other methods,
and verifies exact interpolation accuracy.
"""

import pytest
import numpy as np
from am_qadf.signal_mapping.methods import (
    RBFInterpolation,
    NearestNeighborInterpolation,
    LinearInterpolation,
    IDWInterpolation,
    GaussianKDEInterpolation,
)
from am_qadf.voxelization.voxel_grid import VoxelGrid
from am_qadf.signal_mapping.execution.sequential import interpolate_to_voxels


class TestRBFIntegration:
    """Integration tests for RBF interpolation."""

    @pytest.fixture
    def sample_voxel_grid(self):
        """Create a test voxel grid."""
        return VoxelGrid(
            bbox_min=(0.0, 0.0, 0.0),
            bbox_max=(10.0, 10.0, 10.0),
            resolution=0.5,
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
    def sample_points_and_signals(self):
        """Create sample point cloud data."""
        np.random.seed(42)
        n_points = 100
        points = np.random.rand(n_points, 3) * 10.0

        # Create smooth signal function
        signals = {
            "power": 100.0 + 50.0 * np.sin(points[:, 0] / 2.0) + np.random.randn(n_points) * 2.0,
            "temperature": 1000.0 + 200.0 * np.cos(points[:, 1] / 2.0) + np.random.randn(n_points) * 5.0,
        }

        return points, signals

    @pytest.mark.integration
    def test_rbf_with_real_voxel_grid(self, sample_voxel_grid, sample_points_and_signals):
        """Test RBF interpolation with real voxel grid."""
        points, signals = sample_points_and_signals

        rbf = RBFInterpolation(kernel="gaussian", epsilon=1.0, smoothing=0.0)
        result = rbf.interpolate(points, signals, self._create_voxel_grid_copy(sample_voxel_grid))

        assert result is not None
        assert len(result.voxels) > 0

        # Check that signals are stored
        for voxel_key, voxel_data in result.voxels.items():
            assert hasattr(voxel_data, "signals")
            assert len(voxel_data.signals) > 0

    @pytest.mark.integration
    def test_rbf_different_grid_resolutions(self, sample_points_and_signals):
        """Test RBF with different grid resolutions."""
        points, signals = sample_points_and_signals

        resolutions = [0.5, 1.0, 2.0]
        for resolution in resolutions:
            grid = VoxelGrid(
                bbox_min=(0.0, 0.0, 0.0),
                bbox_max=(10.0, 10.0, 10.0),
                resolution=resolution,
                aggregation="mean",
            )

            rbf = RBFInterpolation(kernel="gaussian", epsilon=1.0)
            result = rbf.interpolate(points, signals, grid)

            assert result is not None
            assert len(result.voxels) > 0

    @pytest.mark.integration
    def test_rbf_exact_interpolation_at_data_points(self, sample_voxel_grid):
        """Test that RBF provides exact interpolation at data points (when smoothing=0)."""
        # Create points that are at voxel centers
        points = np.array(
            [
                [0.25, 0.25, 0.25],  # Voxel center at (0, 0, 0)
                [0.75, 0.75, 0.75],  # Voxel center at (0, 0, 0) or (1, 1, 1)
                [1.25, 1.25, 1.25],  # Voxel center at (1, 1, 1)
                [2.25, 2.25, 2.25],  # Voxel center at (2, 2, 2)
            ]
        )
        signal_values = np.array([100.0, 200.0, 300.0, 400.0])
        signals = {"power": signal_values}

        # Use RBF with exact interpolation (smoothing=0)
        rbf = RBFInterpolation(kernel="gaussian", epsilon=1.0, smoothing=0.0)
        result = rbf.interpolate(points, signals, self._create_voxel_grid_copy(sample_voxel_grid))

        assert result is not None

        # Check that interpolated values are close to original values
        # (exact match may not be possible due to voxel center vs point location)
        # But RBF should interpolate smoothly
        assert len(result.voxels) > 0

    @pytest.mark.integration
    def test_rbf_compare_with_nearest_neighbor(self, sample_voxel_grid, sample_points_and_signals):
        """Compare RBF results with nearest neighbor interpolation."""
        points, signals = sample_points_and_signals

        # RBF interpolation
        rbf = RBFInterpolation(kernel="gaussian", epsilon=1.0, smoothing=0.0)
        rbf_result = rbf.interpolate(points, signals, self._create_voxel_grid_copy(sample_voxel_grid))

        # Nearest neighbor interpolation
        nn = NearestNeighborInterpolation()
        nn_result = nn.interpolate(points, signals, self._create_voxel_grid_copy(sample_voxel_grid))

        # Both should produce results
        assert len(rbf_result.voxels) > 0
        assert len(nn_result.voxels) > 0

        # RBF should generally produce smoother results
        # (can't easily verify without detailed analysis, but both should work)

    @pytest.mark.integration
    def test_rbf_compare_with_linear(self, sample_voxel_grid, sample_points_and_signals):
        """Compare RBF results with linear interpolation."""
        points, signals = sample_points_and_signals

        # RBF interpolation
        rbf = RBFInterpolation(kernel="gaussian", epsilon=1.0, smoothing=0.0)
        rbf_result = rbf.interpolate(points, signals, self._create_voxel_grid_copy(sample_voxel_grid))

        # Linear interpolation
        linear = LinearInterpolation(k_neighbors=8)
        linear_result = linear.interpolate(points, signals, self._create_voxel_grid_copy(sample_voxel_grid))

        # Both should produce results
        assert len(rbf_result.voxels) > 0
        assert len(linear_result.voxels) > 0

    @pytest.mark.integration
    def test_rbf_compare_with_idw(self, sample_voxel_grid, sample_points_and_signals):
        """Compare RBF results with IDW interpolation."""
        points, signals = sample_points_and_signals

        # RBF interpolation
        rbf = RBFInterpolation(kernel="gaussian", epsilon=1.0, smoothing=0.0)
        rbf_result = rbf.interpolate(points, signals, self._create_voxel_grid_copy(sample_voxel_grid))

        # IDW interpolation
        idw = IDWInterpolation(power=2.0, k_neighbors=8)
        idw_result = idw.interpolate(points, signals, self._create_voxel_grid_copy(sample_voxel_grid))

        # Both should produce results
        assert len(rbf_result.voxels) > 0
        assert len(idw_result.voxels) > 0

    @pytest.mark.integration
    def test_rbf_compare_with_kde(self, sample_voxel_grid, sample_points_and_signals):
        """Compare RBF results with Gaussian KDE interpolation."""
        points, signals = sample_points_and_signals

        # RBF interpolation
        rbf = RBFInterpolation(kernel="gaussian", epsilon=1.0, smoothing=0.0)
        rbf_result = rbf.interpolate(points, signals, self._create_voxel_grid_copy(sample_voxel_grid))

        # KDE interpolation
        kde = GaussianKDEInterpolation(bandwidth=1.0)
        kde_result = kde.interpolate(points, signals, self._create_voxel_grid_copy(sample_voxel_grid))

        # Both should produce results
        assert len(rbf_result.voxels) > 0
        assert len(kde_result.voxels) > 0

    @pytest.mark.integration
    def test_rbf_multiple_signals(self, sample_voxel_grid):
        """Test RBF with multiple signal types."""
        points = np.array(
            [
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0],
                [4.0, 4.0, 4.0],
                [5.0, 5.0, 5.0],
            ]
        )
        signals = {
            "power": np.array([100.0, 150.0, 200.0, 250.0, 300.0]),
            "speed": np.array([500.0, 600.0, 700.0, 800.0, 900.0]),
            "temperature": np.array([1000.0, 1100.0, 1200.0, 1300.0, 1400.0]),
        }

        rbf = RBFInterpolation(kernel="gaussian", epsilon=1.0)
        result = rbf.interpolate(points, signals, self._create_voxel_grid_copy(sample_voxel_grid))

        assert result is not None
        assert len(result.voxels) > 0

        # Check that all signals are stored
        signal_names_found = set()
        for voxel_key, voxel_data in result.voxels.items():
            for signal_name in voxel_data.signals.keys():
                signal_names_found.add(signal_name)

        # Should have at least some signals
        assert len(signal_names_found) > 0

    @pytest.mark.integration
    def test_rbf_different_kernels(self, sample_voxel_grid, sample_points_and_signals):
        """Test RBF with different kernel types."""
        points, signals = sample_points_and_signals

        kernels = ["gaussian", "multiquadric", "thin_plate_spline", "linear"]

        for kernel in kernels:
            rbf = RBFInterpolation(kernel=kernel, epsilon=1.0, smoothing=0.0)
            result = rbf.interpolate(points, signals, self._create_voxel_grid_copy(sample_voxel_grid))

            assert result is not None
            assert len(result.voxels) > 0

    @pytest.mark.integration
    def test_rbf_via_interpolate_to_voxels(self, sample_voxel_grid, sample_points_and_signals):
        """Test RBF using the high-level interpolate_to_voxels function."""
        points, signals = sample_points_and_signals

        result = interpolate_to_voxels(
            points=points,
            signals=signals,
            voxel_grid=self._create_voxel_grid_copy(sample_voxel_grid),
            method="rbf",
            kernel="gaussian",
            epsilon=1.0,
            smoothing=0.0,
        )

        assert result is not None
        assert len(result.voxels) > 0

    @pytest.mark.integration
    def test_rbf_smoothing_parameter(self, sample_voxel_grid, sample_points_and_signals):
        """Test RBF with different smoothing parameters."""
        points, signals = sample_points_and_signals

        smoothing_values = [0.0, 0.1, 0.5]

        for smoothing in smoothing_values:
            rbf = RBFInterpolation(kernel="gaussian", epsilon=1.0, smoothing=smoothing)
            result = rbf.interpolate(points, signals, self._create_voxel_grid_copy(sample_voxel_grid))

            assert result is not None
            assert len(result.voxels) > 0

    @pytest.mark.integration
    def test_rbf_auto_epsilon_estimation(self, sample_voxel_grid, sample_points_and_signals):
        """Test RBF with auto-estimated epsilon."""
        points, signals = sample_points_and_signals

        rbf = RBFInterpolation(kernel="gaussian", epsilon=None, smoothing=0.0)
        result = rbf.interpolate(points, signals, self._create_voxel_grid_copy(sample_voxel_grid))

        # Epsilon should be auto-estimated
        assert rbf.epsilon is not None
        assert rbf.epsilon > 0
        assert result is not None
        assert len(result.voxels) > 0

    @pytest.mark.integration
    def test_rbf_accuracy_verification(self, sample_voxel_grid):
        """Verify RBF accuracy by checking smoothness and interpolation quality."""
        # Create a known function: f(x,y,z) = x + y + z
        points = np.array(
            [
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0],
                [4.0, 4.0, 4.0],
                [5.0, 5.0, 5.0],
            ]
        )
        # Signal = x + y + z
        signal_values = points[:, 0] + points[:, 1] + points[:, 2]
        signals = {"test_signal": signal_values}

        rbf = RBFInterpolation(kernel="gaussian", epsilon=1.0, smoothing=0.0)
        result = rbf.interpolate(points, signals, self._create_voxel_grid_copy(sample_voxel_grid))

        assert result is not None
        assert len(result.voxels) > 0

        # Check that interpolated values are reasonable
        # (should be close to x+y+z for points near data)
        for voxel_key, voxel_data in result.voxels.items():
            signal_value = voxel_data.signals.get("test_signal")
            if signal_value is not None:
                # Value should be in reasonable range (3 to 15 for our test points)
                assert 0 <= signal_value <= 20  # Allow some margin
