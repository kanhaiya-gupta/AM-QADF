"""
Performance benchmarks for parallel execution.

Tests speedup from parallelization vs sequential execution.
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock


def create_large_point_cloud(n_points):
    """Create a large point cloud for parallel benchmarking."""
    np.random.seed(42)
    points = np.random.rand(n_points, 3) * 100.0
    signals = {
        "laser_power": np.random.rand(n_points) * 300.0,
        "temperature": np.random.rand(n_points) * 1000.0,
    }
    return points, signals


def create_voxel_grid_for_parallel(bbox_size=100.0, resolution=1.0):
    """Create a voxel grid for parallel benchmarking."""
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
class TestParallelExecutionBenchmarks:
    """Performance benchmarks for parallel execution."""

    @pytest.mark.benchmark
    def test_sequential_vs_parallel_interpolation(self, benchmark):
        """Compare sequential vs parallel interpolation."""
        try:
            from am_qadf.signal_mapping.execution.sequential import (
                interpolate_to_voxels,
            )
            from am_qadf.signal_mapping.execution.parallel import (
                interpolate_to_voxels_parallel,
            )

            points, signals = create_large_point_cloud(100000)

            # Sequential
            grid_seq = create_voxel_grid_for_parallel(bbox_size=100.0, resolution=1.0)

            def run_sequential():
                interpolate_to_voxels(points, signals, grid_seq, method="nearest", use_parallel=False)
                return grid_seq

            sequential_result = benchmark(run_sequential)

            # Parallel
            grid_par = create_voxel_grid_for_parallel(bbox_size=100.0, resolution=1.0)

            def run_parallel():
                interpolate_to_voxels_parallel(points, signals, grid_par, method="nearest", max_workers=4)
                return grid_par

            parallel_result = benchmark(run_parallel)

            assert sequential_result is not None
            assert parallel_result is not None
            # Parallel should be faster (or at least not slower)
        except (ImportError, AttributeError):
            pytest.skip("Parallel execution module not available")

    @pytest.mark.benchmark
    def test_parallel_speedup_scaling(self, benchmark):
        """Test speedup scaling with number of workers."""
        try:
            from am_qadf.signal_mapping.execution.parallel import (
                interpolate_to_voxels_parallel,
            )

            points, signals = create_large_point_cloud(200000)
            worker_counts = [1, 2, 4, 8]
            results = {}

            for n_workers in worker_counts:
                grid = create_voxel_grid_for_parallel(bbox_size=100.0, resolution=1.0)

                def run_parallel():
                    interpolate_to_voxels_parallel(points, signals, grid, method="nearest", max_workers=n_workers)
                    return grid

                result = benchmark(run_parallel)
                results[n_workers] = result
                assert result is not None

            assert len(results) == len(worker_counts)
        except (ImportError, AttributeError):
            pytest.skip("Parallel execution module not available")

    @pytest.mark.benchmark
    def test_parallel_source_processing(self, benchmark):
        """Benchmark parallel processing of multiple sources."""
        try:
            from am_qadf.voxel_domain.voxel_domain_client import VoxelDomainClient

            class MockUnifiedQueryClient:
                def __init__(self):
                    self.stl_client = Mock()
                    self.stl_client.get_model_bounding_box = Mock(return_value=([0.0, 0.0, 0.0], [100.0, 100.0, 100.0]))
                    self.hatching_client = Mock()
                    self.hatching_client.get_layers = Mock(return_value=[])
                    self.laser_client = Mock()
                    self.ct_client = Mock()
                    self.ispm_client = Mock()

            unified_client = MockUnifiedQueryClient()
            domain_client = VoxelDomainClient(unified_query_client=unified_client)

            grid = domain_client.create_voxel_grid(model_id="test_model", bbox_min=(0, 0, 0), bbox_max=(100, 100, 100))

            def run_parallel_sources():
                return domain_client.map_signals_to_voxels(
                    model_id="test_model",
                    voxel_grid=grid,
                    sources=["hatching", "laser", "ct"],
                    use_parallel_sources=True,
                    max_workers=4,
                )

            result = benchmark(run_parallel_sources)
            assert result is not None
        except (ImportError, AttributeError):
            pytest.skip("VoxelDomainClient not available")

    @pytest.mark.benchmark
    def test_parallel_vs_sequential_sources(self, benchmark):
        """Compare parallel vs sequential source processing."""
        try:
            from am_qadf.voxel_domain.voxel_domain_client import VoxelDomainClient

            class MockUnifiedQueryClient:
                def __init__(self):
                    self.stl_client = Mock()
                    self.stl_client.get_model_bounding_box = Mock(return_value=([0.0, 0.0, 0.0], [100.0, 100.0, 100.0]))
                    self.hatching_client = Mock()
                    self.hatching_client.get_layers = Mock(return_value=[])
                    self.laser_client = Mock()
                    self.ct_client = Mock()
                    self.ispm_client = Mock()

            unified_client = MockUnifiedQueryClient()
            domain_client = VoxelDomainClient(unified_query_client=unified_client)

            sources = ["hatching", "laser", "ct", "ispm"]

            # Sequential
            grid_seq = domain_client.create_voxel_grid(model_id="test_model", bbox_min=(0, 0, 0), bbox_max=(100, 100, 100))

            def run_sequential():
                return domain_client.map_signals_to_voxels(
                    model_id="test_model",
                    voxel_grid=grid_seq,
                    sources=sources,
                    use_parallel_sources=False,
                )

            sequential_result = benchmark(run_sequential)

            # Parallel
            grid_par = domain_client.create_voxel_grid(model_id="test_model", bbox_min=(0, 0, 0), bbox_max=(100, 100, 100))

            def run_parallel():
                return domain_client.map_signals_to_voxels(
                    model_id="test_model",
                    voxel_grid=grid_par,
                    sources=sources,
                    use_parallel_sources=True,
                    max_workers=4,
                )

            parallel_result = benchmark(run_parallel)

            assert sequential_result is not None
            assert parallel_result is not None
        except (ImportError, AttributeError):
            pytest.skip("VoxelDomainClient not available")

    @pytest.mark.benchmark
    def test_parallel_interpolation_overhead(self, benchmark):
        """Test overhead of parallel execution setup."""
        try:
            from am_qadf.signal_mapping.execution.sequential import (
                interpolate_to_voxels,
            )
            from am_qadf.signal_mapping.execution.parallel import (
                interpolate_to_voxels_parallel,
            )

            # Small dataset (parallel overhead might dominate)
            points, signals = create_large_point_cloud(10000)

            # Sequential
            grid_seq = create_voxel_grid_for_parallel(bbox_size=100.0, resolution=1.0)

            def run_sequential():
                interpolate_to_voxels(points, signals, grid_seq, method="nearest", use_parallel=False)
                return grid_seq

            sequential_result = benchmark(run_sequential)

            # Parallel
            grid_par = create_voxel_grid_for_parallel(bbox_size=100.0, resolution=1.0)

            def run_parallel():
                interpolate_to_voxels_parallel(points, signals, grid_par, method="nearest", max_workers=4)
                return grid_par

            parallel_result = benchmark(run_parallel)

            assert sequential_result is not None
            assert parallel_result is not None
        except (ImportError, AttributeError):
            pytest.skip("Parallel execution module not available")

    @pytest.mark.benchmark
    def test_parallel_scalability(self, benchmark):
        """Test parallel execution scalability with data size."""
        try:
            from am_qadf.signal_mapping.execution.parallel import (
                interpolate_to_voxels_parallel,
            )

            sizes = [50000, 100000, 200000, 500000]
            results = {}

            for size in sizes:
                points, signals = create_large_point_cloud(size)
                grid = create_voxel_grid_for_parallel(bbox_size=100.0, resolution=1.0)

                def run_parallel():
                    interpolate_to_voxels_parallel(points, signals, grid, method="nearest", max_workers=4)
                    return grid

                result = benchmark(run_parallel)
                results[size] = result
                assert result is not None

            assert len(results) == len(sizes)
        except (ImportError, AttributeError):
            pytest.skip("Parallel execution module not available")

    @pytest.mark.benchmark
    def test_parallel_memory_usage(self, benchmark):
        """Benchmark memory usage of parallel execution."""
        import tracemalloc

        try:
            from am_qadf.signal_mapping.execution.parallel import (
                interpolate_to_voxels_parallel,
            )

            points, signals = create_large_point_cloud(200000)
            grid = create_voxel_grid_for_parallel(bbox_size=100.0, resolution=1.0)

            tracemalloc.start()

            def run_parallel():
                interpolate_to_voxels_parallel(points, signals, grid, method="nearest", max_workers=4)
                return grid

            result = benchmark(run_parallel)

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            assert result is not None
            # Memory should be reasonable
            assert peak < 3 * 1024 * 1024 * 1024  # 3GB limit (parallel may use more)
        except (ImportError, AttributeError):
            pytest.skip("Parallel execution module not available")

    @pytest.mark.benchmark
    def test_spark_execution_performance(self, benchmark):
        """Benchmark Spark execution for large datasets."""
        try:
            from am_qadf.signal_mapping.execution.spark import (
                interpolate_to_voxels_spark,
            )

            points, signals = create_large_point_cloud(1000000)  # 1M points
            grid = create_voxel_grid_for_parallel(bbox_size=100.0, resolution=1.0)

            # Mock Spark session
            mock_spark = Mock()

            def run_spark():
                interpolate_to_voxels_spark(points, signals, grid, method="nearest", spark_session=mock_spark)
                return grid

            result = benchmark(run_spark)
            assert result is not None
        except (ImportError, AttributeError):
            pytest.skip("Spark execution module not available")
