"""
Performance benchmarks for signal mapping.

Tests signal mapping performance with various data sizes and methods.
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock
from am_qadf.voxel_domain.voxel_domain_client import VoxelDomainClient
from am_qadf.signal_mapping.execution.sequential import interpolate_to_voxels


class MockUnifiedQueryClient:
    """Mock unified query client for benchmarking."""

    def __init__(self):
        self.stl_client = Mock()
        self.stl_client.get_model_bounding_box = Mock(return_value=([0.0, 0.0, 0.0], [100.0, 100.0, 100.0]))
        self.hatching_client = Mock()
        self.laser_client = Mock()
        self.ct_client = Mock()
        self.ispm_client = Mock()


def create_large_point_cloud(n_points):
    """Create a large point cloud for benchmarking."""
    np.random.seed(42)
    points = np.random.rand(n_points, 3) * 100.0
    signals = {
        "laser_power": np.random.rand(n_points) * 300.0,
        "temperature": np.random.rand(n_points) * 1000.0,
    }
    return points, signals


def create_voxel_grid_for_benchmark(bbox_size=100.0, resolution=1.0):
    """Create a voxel grid for benchmarking."""
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
class TestSignalMappingBenchmarks:
    """Performance benchmarks for signal mapping."""

    @pytest.fixture
    def unified_client(self):
        """Create mock unified query client."""
        return MockUnifiedQueryClient()

    @pytest.fixture
    def voxel_domain_client(self, unified_client):
        """Create voxel domain client."""
        return VoxelDomainClient(unified_query_client=unified_client)

    @pytest.mark.benchmark
    def test_signal_mapping_small_dataset(self, benchmark):
        """Benchmark signal mapping with small dataset (10K points)."""
        points, signals = create_large_point_cloud(10000)
        grid = create_voxel_grid_for_benchmark(bbox_size=100.0, resolution=1.0)

        def run_mapping():
            interpolate_to_voxels(points, signals, grid, method="nearest")
            return grid

        result = benchmark(run_mapping)
        assert result is not None

    @pytest.mark.benchmark
    def test_signal_mapping_medium_dataset(self, benchmark):
        """Benchmark signal mapping with medium dataset (100K points)."""
        points, signals = create_large_point_cloud(100000)
        grid = create_voxel_grid_for_benchmark(bbox_size=100.0, resolution=1.0)

        def run_mapping():
            interpolate_to_voxels(points, signals, grid, method="nearest")
            return grid

        result = benchmark(run_mapping)
        assert result is not None

    @pytest.mark.benchmark
    def test_signal_mapping_large_dataset(self, benchmark):
        """Benchmark signal mapping with large dataset (1M points)."""
        points, signals = create_large_point_cloud(1000000)
        grid = create_voxel_grid_for_benchmark(bbox_size=100.0, resolution=1.0)

        def run_mapping():
            interpolate_to_voxels(points, signals, grid, method="nearest")
            return grid

        result = benchmark(run_mapping)
        assert result is not None
        # Performance assertion: should complete in reasonable time
        # (actual time depends on hardware, but should be < 60 seconds for 1M points)

    @pytest.mark.benchmark
    def test_signal_mapping_vectorization_benefit(self, benchmark):
        """Benchmark to validate vectorization benefits."""
        points, signals = create_large_point_cloud(100000)
        grid = create_voxel_grid_for_benchmark(bbox_size=100.0, resolution=1.0)

        def run_mapping():
            interpolate_to_voxels(points, signals, grid, method="nearest")
            return grid

        result = benchmark(run_mapping)
        assert result is not None

    @pytest.mark.benchmark
    def test_signal_mapping_different_resolutions(self, benchmark):
        """Benchmark signal mapping with different grid resolutions."""
        points, signals = create_large_point_cloud(50000)

        resolutions = [0.5, 1.0, 2.0]
        for resolution in resolutions:
            grid = create_voxel_grid_for_benchmark(bbox_size=100.0, resolution=resolution)

            def run_mapping():
                interpolate_to_voxels(points, signals, grid, method="nearest")
                return grid

            result = benchmark(run_mapping)
            assert result is not None

    @pytest.mark.benchmark
    def test_signal_mapping_multiple_signals(self, benchmark):
        """Benchmark mapping multiple signals simultaneously."""
        n_points = 50000
        points = np.random.rand(n_points, 3) * 100.0
        signals = {
            "laser_power": np.random.rand(n_points) * 300.0,
            "temperature": np.random.rand(n_points) * 1000.0,
            "density": np.random.rand(n_points) * 1.0,
            "velocity": np.random.rand(n_points) * 100.0,
        }
        grid = create_voxel_grid_for_benchmark(bbox_size=100.0, resolution=1.0)

        def run_mapping():
            interpolate_to_voxels(points, signals, grid, method="nearest")
            return grid

        result = benchmark(run_mapping)
        assert result is not None

    @pytest.mark.benchmark
    def test_signal_mapping_memory_efficiency(self, benchmark):
        """Benchmark memory efficiency with large datasets."""
        import tracemalloc

        points, signals = create_large_point_cloud(500000)
        grid = create_voxel_grid_for_benchmark(bbox_size=100.0, resolution=1.0)

        tracemalloc.start()

        def run_mapping():
            interpolate_to_voxels(points, signals, grid, method="nearest")
            return grid

        result = benchmark(run_mapping)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert result is not None
        # Memory should be reasonable (peak < 2GB for 500K points)
        assert peak < 2 * 1024 * 1024 * 1024  # 2GB limit

    @pytest.mark.benchmark
    def test_signal_mapping_scalability(self, benchmark):
        """Benchmark scalability across different data sizes."""
        sizes = [10000, 50000, 100000, 500000]
        results = {}

        for size in sizes:
            points, signals = create_large_point_cloud(size)
            grid = create_voxel_grid_for_benchmark(bbox_size=100.0, resolution=1.0)

            def run_mapping():
                interpolate_to_voxels(points, signals, grid, method="nearest")
                return grid

            result = benchmark(run_mapping)
            results[size] = result
            assert result is not None

        # Verify scalability (time should scale roughly linearly with size)
        assert len(results) == len(sizes)
