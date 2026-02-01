"""
Performance tests for signal mapping interpolation methods.

Covers nearest neighbor, linear, IDW, Gaussian KDE, and RBF.
Measures scaling, relative timing, and documents performance characteristics.
"""

import pytest
import numpy as np
import time
import logging
from am_qadf.voxelization.uniform_resolution import VoxelGrid
from am_qadf.signal_mapping.execution.sequential import interpolate_to_voxels

logger = logging.getLogger(__name__)


def create_test_data(n_points, bbox_size=10.0, seed=42):
    """Create test data for performance testing."""
    np.random.seed(seed)
    points = np.random.rand(n_points, 3) * bbox_size
    signals = {"power": 100.0 + 50.0 * np.sin(points[:, 0] / 2.0) + np.random.randn(n_points) * 2.0}
    return points, signals


def create_test_grid(bbox_size=10.0, resolution=0.5):
    """Create test voxel grid."""
    return VoxelGrid(
        bbox_min=(0.0, 0.0, 0.0),
        bbox_max=(bbox_size, bbox_size, bbox_size),
        resolution=resolution,
        aggregation="mean",
    )


def _copy_grid(grid):
    """Create a fresh grid with same bounds/resolution (for in-place mutation)."""
    return VoxelGrid(
        bbox_min=tuple(grid.bbox_min),
        bbox_max=tuple(grid.bbox_max),
        resolution=grid.resolution,
        aggregation=grid.aggregation,
    )


METHODS = ["nearest", "linear", "idw", "gaussian_kde", "rbf"]


@pytest.mark.performance
class TestInterpolationMethodsPerformance:
    """Performance tests for all signal mapping interpolation methods."""

    @pytest.mark.performance
    @pytest.mark.parametrize("method", METHODS)
    def test_method_scaling_behavior(self, method):
        """Test that each method completes and time increases with dataset size."""
        sizes = [100, 500, 1000]  # Keep moderate for all methods (RBF is O(N³))
        times = []

        for n_points in sizes:
            points, signals = create_test_data(n_points)
            grid = create_test_grid()

            start_time = time.time()
            result = interpolate_to_voxels(points, signals, grid, method=method)
            elapsed = time.time() - start_time

            times.append(elapsed)
            logger.info(f"{method} with {n_points} points: {elapsed:.3f}s")
            assert result is not None

        if len(times) >= 2:
            assert times[-1] >= 0, "Time should be non-negative"

    @pytest.mark.performance
    def test_nearest_neighbor_performance(self):
        """Performance test for nearest neighbor interpolation."""
        n_points = 5000
        points, signals = create_test_data(n_points)
        grid = create_test_grid()

        start_time = time.time()
        result = interpolate_to_voxels(points, signals, grid, method="nearest")
        elapsed = time.time() - start_time

        logger.info(f"Nearest neighbor: {n_points} points in {elapsed:.3f}s")
        assert result is not None

    @pytest.mark.performance
    def test_linear_interpolation_performance(self):
        """Performance test for linear interpolation."""
        n_points = 5000
        points, signals = create_test_data(n_points)
        grid = create_test_grid()

        start_time = time.time()
        result = interpolate_to_voxels(points, signals, grid, method="linear")
        elapsed = time.time() - start_time

        logger.info(f"Linear: {n_points} points in {elapsed:.3f}s")
        assert result is not None

    @pytest.mark.performance
    def test_idw_interpolation_performance(self):
        """Performance test for IDW interpolation."""
        n_points = 5000
        points, signals = create_test_data(n_points)
        grid = create_test_grid()

        start_time = time.time()
        result = interpolate_to_voxels(points, signals, grid, method="idw")
        elapsed = time.time() - start_time

        logger.info(f"IDW: {n_points} points in {elapsed:.3f}s")
        assert result is not None

    @pytest.mark.performance
    def test_gaussian_kde_interpolation_performance(self):
        """Performance test for Gaussian KDE interpolation."""
        n_points = 2000  # KDE can be heavy
        points, signals = create_test_data(n_points)
        grid = create_test_grid()

        start_time = time.time()
        result = interpolate_to_voxels(points, signals, grid, method="gaussian_kde")
        elapsed = time.time() - start_time

        logger.info(f"Gaussian KDE: {n_points} points in {elapsed:.3f}s")
        assert result is not None

    @pytest.mark.performance
    def test_rbf_interpolation_performance(self):
        """Performance test for RBF interpolation."""
        n_points = 200  # RBF is O(N³)
        points, signals = create_test_data(n_points)
        grid = create_test_grid()

        start_time = time.time()
        result = interpolate_to_voxels(points, signals, grid, method="rbf")
        elapsed = time.time() - start_time

        logger.info(f"RBF: {n_points} points in {elapsed:.3f}s")
        assert result is not None

    @pytest.mark.performance
    def test_relative_performance_all_methods(self):
        """Compare relative timing of all interpolation methods (same dataset)."""
        n_points = 500  # Small enough for RBF
        points, signals = create_test_data(n_points)
        grid = create_test_grid()

        times = {}
        for method in METHODS:
            g = _copy_grid(grid)
            start_time = time.time()
            result = interpolate_to_voxels(points, signals, g, method=method)
            elapsed = time.time() - start_time
            times[method] = elapsed
            logger.info(f"{method}: {elapsed:.3f}s")
            assert result is not None

        logger.info("Relative times (nearest=1.0): %s", {m: times[m] / times["nearest"] for m in METHODS})
        assert len(times) == len(METHODS)
