"""
Performance benchmarks for interpolation methods.

Compares speed and accuracy of different interpolation methods.
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock


def create_test_data(n_points=10000, bbox_size=100.0):
    """Create test data for interpolation benchmarking."""
    np.random.seed(42)
    points = np.random.rand(n_points, 3) * bbox_size
    values = np.random.rand(n_points) * 100.0
    signals = {"test_signal": values}
    return points, signals


def create_voxel_grid_for_interpolation(bbox_size=100.0, resolution=1.0):
    """Create a voxel grid for interpolation benchmarking."""
    try:
        from am_qadf.voxelization.voxel_grid import VoxelGrid

        return VoxelGrid(
            bbox_min=(0.0, 0.0, 0.0),
            bbox_max=(bbox_size, bbox_size, bbox_size),
            resolution=resolution,
        )
    except ImportError:
        pytest.skip("VoxelGrid not available")


@pytest.mark.benchmark
@pytest.mark.performance
class TestInterpolationMethodBenchmarks:
    """Performance benchmarks for interpolation methods."""

    @pytest.mark.benchmark
    def test_nearest_neighbor_performance(self, benchmark):
        """Benchmark nearest neighbor interpolation."""
        try:
            from am_qadf.signal_mapping.execution.sequential import (
                interpolate_to_voxels,
            )

            points, signals = create_test_data(n_points=50000)
            grid = create_voxel_grid_for_interpolation(bbox_size=100.0, resolution=1.0)

            def run_interpolation():
                interpolate_to_voxels(points, signals, grid, method="nearest")
                return grid

            result = benchmark(run_interpolation)
            assert result is not None
        except ImportError:
            pytest.skip("Interpolation module not available")

    @pytest.mark.benchmark
    def test_linear_interpolation_performance(self, benchmark):
        """Benchmark linear interpolation."""
        try:
            from am_qadf.signal_mapping.execution.sequential import (
                interpolate_to_voxels,
            )

            points, signals = create_test_data(n_points=50000)
            grid = create_voxel_grid_for_interpolation(bbox_size=100.0, resolution=1.0)

            def run_interpolation():
                interpolate_to_voxels(points, signals, grid, method="linear")
                return grid

            result = benchmark(run_interpolation)
            assert result is not None
        except ImportError:
            pytest.skip("Interpolation module not available")

    @pytest.mark.benchmark
    def test_idw_interpolation_performance(self, benchmark):
        """Benchmark IDW (Inverse Distance Weighting) interpolation."""
        try:
            from am_qadf.signal_mapping.execution.sequential import (
                interpolate_to_voxels,
            )

            points, signals = create_test_data(n_points=50000)
            grid = create_voxel_grid_for_interpolation(bbox_size=100.0, resolution=1.0)

            def run_interpolation():
                interpolate_to_voxels(points, signals, grid, method="idw")
                return grid

            result = benchmark(run_interpolation)
            assert result is not None
        except ImportError:
            pytest.skip("Interpolation module not available")

    @pytest.mark.benchmark
    def test_kde_interpolation_performance(self, benchmark):
        """Benchmark KDE (Kernel Density Estimation) interpolation."""
        try:
            from am_qadf.signal_mapping.execution.sequential import (
                interpolate_to_voxels,
            )

            points, signals = create_test_data(n_points=50000)
            grid = create_voxel_grid_for_interpolation(bbox_size=100.0, resolution=1.0)

            def run_interpolation():
                interpolate_to_voxels(points, signals, grid, method="gaussian_kde")
                return grid

            result = benchmark(run_interpolation)
            assert result is not None
        except ImportError:
            pytest.skip("Interpolation module not available")

    @pytest.mark.benchmark
    def test_interpolation_method_comparison(self, benchmark):
        """Compare performance of all interpolation methods."""
        try:
            from am_qadf.signal_mapping.execution.sequential import (
                interpolate_to_voxels,
            )

            points, signals = create_test_data(n_points=50000)
            methods = ["nearest", "linear", "idw", "gaussian_kde"]
            results = {}

            for method in methods:
                grid = create_voxel_grid_for_interpolation(bbox_size=100.0, resolution=1.0)

                def run_interpolation():
                    interpolate_to_voxels(points, signals, grid, method=method)
                    return grid

                result = benchmark(run_interpolation)
                results[method] = result
                assert result is not None

            # All methods should complete
            assert len(results) == len(methods)
        except ImportError:
            pytest.skip("Interpolation module not available")

    @pytest.mark.benchmark
    def test_interpolation_scalability(self, benchmark):
        """Benchmark interpolation scalability with different data sizes."""
        try:
            from am_qadf.signal_mapping.execution.sequential import (
                interpolate_to_voxels,
            )

            sizes = [10000, 50000, 100000]
            results = {}

            for size in sizes:
                points, signals = create_test_data(n_points=size)
                grid = create_voxel_grid_for_interpolation(bbox_size=100.0, resolution=1.0)

                def run_interpolation():
                    interpolate_to_voxels(points, signals, grid, method="nearest")
                    return grid

                result = benchmark(run_interpolation)
                results[size] = result
                assert result is not None

            assert len(results) == len(sizes)
        except ImportError:
            pytest.skip("Interpolation module not available")

    @pytest.mark.benchmark
    def test_interpolation_different_resolutions(self, benchmark):
        """Benchmark interpolation with different grid resolutions."""
        try:
            from am_qadf.signal_mapping.execution.sequential import (
                interpolate_to_voxels,
            )

            points, signals = create_test_data(n_points=50000)
            resolutions = [0.5, 1.0, 2.0]
            results = {}

            for resolution in resolutions:
                grid = create_voxel_grid_for_interpolation(bbox_size=100.0, resolution=resolution)

                def run_interpolation():
                    interpolate_to_voxels(points, signals, grid, method="nearest")
                    return grid

                result = benchmark(run_interpolation)
                results[resolution] = result
                assert result is not None

            assert len(results) == len(resolutions)
        except ImportError:
            pytest.skip("Interpolation module not available")

    @pytest.mark.benchmark
    def test_interpolation_accuracy_vs_speed(self, benchmark):
        """Benchmark to compare accuracy vs speed trade-offs."""
        try:
            from am_qadf.signal_mapping.execution.sequential import (
                interpolate_to_voxels,
            )

            # Create known test data
            points = np.array(
                [
                    [10.0, 10.0, 10.0],
                    [20.0, 20.0, 20.0],
                    [30.0, 30.0, 30.0],
                    [40.0, 40.0, 40.0],
                    [50.0, 50.0, 50.0],
                ]
            )
            signals = {"test_signal": np.array([1.0, 2.0, 3.0, 4.0, 5.0])}

            methods = ["nearest", "linear", "idw"]
            results = {}

            for method in methods:
                grid = create_voxel_grid_for_interpolation(bbox_size=100.0, resolution=1.0)

                def run_interpolation():
                    interpolate_to_voxels(points, signals, grid, method=method)
                    return grid

                result = benchmark(run_interpolation)
                results[method] = result
                assert result is not None

            assert len(results) == len(methods)
        except ImportError:
            pytest.skip("Interpolation module not available")

    @pytest.mark.benchmark
    def test_interpolation_memory_efficiency(self, benchmark):
        """Benchmark memory efficiency of interpolation methods."""
        import tracemalloc

        try:
            from am_qadf.signal_mapping.execution.sequential import (
                interpolate_to_voxels,
            )

            points, signals = create_test_data(n_points=100000)
            grid = create_voxel_grid_for_interpolation(bbox_size=100.0, resolution=1.0)

            tracemalloc.start()

            def run_interpolation():
                interpolate_to_voxels(points, signals, grid, method="nearest")
                return grid

            result = benchmark(run_interpolation)

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            assert result is not None
            # Memory should be reasonable
            assert peak < 2 * 1024 * 1024 * 1024  # 2GB limit
        except ImportError:
            pytest.skip("Interpolation module not available")
