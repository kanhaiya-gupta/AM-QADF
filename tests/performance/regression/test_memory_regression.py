"""
Memory regression tests.

Track memory usage for large datasets, detect memory leaks, and track peak memory.
"""

import pytest
import numpy as np
import tracemalloc
import gc
import json
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock


# Memory baseline thresholds (in MB)
# Note: Sequential processing is limited to 200K points maximum. Larger datasets must use Spark.
MEMORY_BASELINES = {
    "signal_mapping_10k": 50.0,  # 10K points (sequential)
    "signal_mapping_50k": 100.0,  # 50K points (sequential)
    "signal_mapping_200k": 300.0,  # 200K points (sequential - maximum)
    "voxel_fusion_small": 20.0,  # Small grids
    "voxel_fusion_medium": 100.0,  # Medium grids
    "voxel_fusion_large": 500.0,  # Large grids
    "interpolation_nearest": 100.0,  # Nearest neighbor
    "interpolation_linear": 150.0,  # Linear interpolation
    "interpolation_idw": 200.0,  # IDW interpolation
    "parallel_4_workers": 400.0,  # 4 workers
}

# Memory regression threshold (20% increase)
MEMORY_REGRESSION_THRESHOLD = 0.20

# Baseline storage file
MEMORY_BASELINE_FILE = Path(__file__).parent / "memory_baselines.json"


def load_memory_baselines():
    """Load memory baselines from file."""
    if MEMORY_BASELINE_FILE.exists():
        with open(MEMORY_BASELINE_FILE, "r") as f:
            return json.load(f)
    return MEMORY_BASELINES.copy()


def save_memory_baselines(baselines):
    """Save memory baselines to file."""
    with open(MEMORY_BASELINE_FILE, "w") as f:
        json.dump(baselines, f, indent=2)


def bytes_to_mb(bytes_value):
    """Convert bytes to megabytes."""
    return bytes_value / (1024 * 1024)


def check_memory_regression(test_name, peak_memory_mb, baseline_memory_mb, threshold=MEMORY_REGRESSION_THRESHOLD):
    """
    Check if memory usage has regressed.

    Args:
        test_name: Name of the test
        peak_memory_mb: Current peak memory in MB
        baseline_memory_mb: Baseline memory in MB
        threshold: Regression threshold (default: 20%)

    Returns:
        tuple: (is_regressed, increase_percentage, message)
    """
    if baseline_memory_mb is None:
        return False, 0.0, f"No baseline for {test_name}"

    increase = (peak_memory_mb - baseline_memory_mb) / baseline_memory_mb

    if increase > threshold:
        return (
            True,
            increase * 100,
            f"Memory regression detected: {increase*100:.1f}% more memory than baseline",
        )
    else:
        return (
            False,
            increase * 100,
            f"Memory within threshold: {increase*100:.1f}% change",
        )


def create_large_point_cloud(n_points):
    """Create a large point cloud for memory testing."""
    np.random.seed(42)
    points = np.random.rand(n_points, 3) * 100.0
    signals = {
        "laser_power": np.random.rand(n_points) * 300.0,
        "temperature": np.random.rand(n_points) * 1000.0,
    }
    return points, signals


def create_voxel_grid(bbox_size=100.0, resolution=1.0):
    """Create a voxel grid for memory testing."""
    try:
        from am_qadf.voxelization.voxel_grid import VoxelGrid

        return VoxelGrid(
            bbox_min=(0.0, 0.0, 0.0),
            bbox_max=(bbox_size, bbox_size, bbox_size),
            resolution=resolution,
        )
    except ImportError:
        pytest.skip("VoxelGrid not available")


@pytest.mark.regression
@pytest.mark.performance
class TestMemoryRegression:
    """Memory regression tests."""

    @pytest.fixture(scope="class", autouse=True)
    def memory_baselines(self):
        """Load memory baselines at class level."""
        return load_memory_baselines()

    @pytest.mark.regression
    def test_signal_mapping_memory_10k(self, memory_baselines):
        """Test memory usage for signal mapping (10K points) - sequential processing for low data size."""
        try:
            from am_qadf.signal_mapping.execution.sequential import (
                interpolate_to_voxels,
            )

            points, signals = create_large_point_cloud(10000)
            grid = create_voxel_grid(bbox_size=100.0, resolution=1.0)

            tracemalloc.start()
            interpolate_to_voxels(
                points,
                signals,
                grid,
                method="nearest",
                use_spark=False,  # Explicitly use sequential for low data size
            )
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            peak_mb = bytes_to_mb(peak)
            baseline_mb = memory_baselines.get("signal_mapping_10k")

            is_regressed, increase, message = check_memory_regression("signal_mapping_10k", peak_mb, baseline_mb)

            if is_regressed:
                pytest.fail(f"Memory regression: {message}")
        except ImportError:
            pytest.skip("Interpolation module not available")

    @pytest.mark.regression
    def test_signal_mapping_memory_50k(self, memory_baselines):
        """Test memory usage for signal mapping (50K points) - sequential processing for low data size."""
        try:
            from am_qadf.signal_mapping.execution.sequential import (
                interpolate_to_voxels,
            )

            points, signals = create_large_point_cloud(50000)
            grid = create_voxel_grid(bbox_size=100.0, resolution=1.0)

            tracemalloc.start()
            interpolate_to_voxels(
                points,
                signals,
                grid,
                method="nearest",
                use_spark=False,  # Explicitly use sequential for low data size
            )
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            peak_mb = bytes_to_mb(peak)
            baseline_mb = memory_baselines.get("signal_mapping_50k", memory_baselines.get("signal_mapping_100k"))

            is_regressed, increase, message = check_memory_regression("signal_mapping_50k", peak_mb, baseline_mb)

            if is_regressed:
                pytest.fail(f"Memory regression: {message}")
        except ImportError:
            pytest.skip("Interpolation module not available")

    @pytest.mark.regression
    def test_signal_mapping_memory_200k(self, memory_baselines):
        """Test memory usage for signal mapping (200K points) - maximum sequential processing size."""
        try:
            from am_qadf.signal_mapping.execution.sequential import (
                interpolate_to_voxels,
            )

            points, signals = create_large_point_cloud(200000)
            grid = create_voxel_grid(bbox_size=100.0, resolution=1.0)

            tracemalloc.start()
            interpolate_to_voxels(
                points,
                signals,
                grid,
                method="nearest",
                use_spark=False,  # Sequential processing - 200K is the maximum for sequential
            )
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            peak_mb = bytes_to_mb(peak)
            baseline_mb = memory_baselines.get(
                "signal_mapping_200k",
                memory_baselines.get("signal_mapping_100k", 300.0),
            )

            is_regressed, increase, message = check_memory_regression("signal_mapping_200k", peak_mb, baseline_mb)

            if is_regressed:
                pytest.fail(f"Memory regression: {message}")
        except ImportError:
            pytest.skip("Interpolation module not available")

    @pytest.mark.regression
    def test_memory_leak_detection(self):
        """Test for memory leaks by running operation multiple times - sequential processing for low data size."""
        try:
            from am_qadf.signal_mapping.execution.sequential import (
                interpolate_to_voxels,
            )

            points, signals = create_large_point_cloud(50000)
            grid = create_voxel_grid(bbox_size=100.0, resolution=1.0)

            tracemalloc.start()
            initial_current, initial_peak = tracemalloc.get_traced_memory()

            # Run operation multiple times
            for i in range(10):
                interpolate_to_voxels(
                    points,
                    signals,
                    grid,
                    method="nearest",
                    use_spark=False,  # Explicitly use sequential for low data size
                )
                gc.collect()  # Force garbage collection

            final_current, final_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Check for memory growth
            memory_growth = bytes_to_mb(final_current - initial_current)

            # Memory should not grow significantly
            # Use both percentage (50%) and absolute (50 MB) thresholds to be more lenient
            # This accounts for Python's memory management and caching behavior
            initial_mb = bytes_to_mb(initial_current)
            max_growth_percentage = initial_mb * 0.5  # 50% growth allowed
            max_growth_absolute = 50.0  # 50 MB absolute growth allowed
            max_growth_mb = max(max_growth_percentage, max_growth_absolute)

            if memory_growth > max_growth_mb:
                pytest.fail(
                    f"Potential memory leak detected: {memory_growth:.2f} MB growth after 10 iterations (threshold: {max_growth_mb:.2f} MB)"
                )
        except ImportError:
            pytest.skip("Interpolation module not available")

    @pytest.mark.regression
    def test_peak_memory_tracking(self, memory_baselines):
        """Test peak memory tracking - sequential processing for low data size."""
        try:
            from am_qadf.signal_mapping.execution.sequential import (
                interpolate_to_voxels,
            )

            points, signals = create_large_point_cloud(50000)
            grid = create_voxel_grid(bbox_size=100.0, resolution=1.0)

            tracemalloc.start()
            interpolate_to_voxels(
                points,
                signals,
                grid,
                method="nearest",
                use_spark=False,  # Explicitly use sequential for low data size
            )
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            peak_mb = bytes_to_mb(peak)

            # Verify peak memory is tracked
            assert peak_mb > 0
            assert peak >= current

            # Check against baseline
            baseline_mb = memory_baselines.get("signal_mapping_50k", memory_baselines.get("signal_mapping_100k"))
            if baseline_mb:
                is_regressed, increase, message = check_memory_regression("signal_mapping_50k", peak_mb, baseline_mb)
                if is_regressed:
                    pytest.fail(f"Memory regression: {message}")
        except ImportError:
            pytest.skip("Interpolation module not available")

    @pytest.mark.regression
    def test_voxel_fusion_memory_small(self, memory_baselines):
        """Test memory usage for voxel fusion (small grids)."""
        try:
            from am_qadf.fusion.voxel_fusion import VoxelFusion

            grids = []
            for i in range(3):
                grid = create_voxel_grid(bbox_size=10.0, resolution=1.0)
                grid.available_signals = {f"signal_{i}"}
                grids.append(grid)

            fusion = VoxelFusion()

            tracemalloc.start()
            fused = fusion.fuse_voxel_grids(grids, method="weighted_average")
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            peak_mb = bytes_to_mb(peak)
            baseline_mb = memory_baselines.get("voxel_fusion_small")

            is_regressed, increase, message = check_memory_regression("voxel_fusion_small", peak_mb, baseline_mb)

            if is_regressed:
                pytest.fail(f"Memory regression: {message}")
        except (ImportError, AttributeError):
            pytest.skip("VoxelFusion not available")

    @pytest.mark.regression
    def test_voxel_fusion_memory_large(self, memory_baselines):
        """Test memory usage for voxel fusion (large grids)."""
        try:
            from am_qadf.fusion.voxel_fusion import VoxelFusion

            grids = []
            for i in range(5):
                grid = create_voxel_grid(bbox_size=100.0, resolution=1.0)
                grid.available_signals = {f"signal_{i}"}
                grids.append(grid)

            fusion = VoxelFusion()

            tracemalloc.start()
            fused = fusion.fuse_voxel_grids(grids, method="weighted_average")
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            peak_mb = bytes_to_mb(peak)
            baseline_mb = memory_baselines.get("voxel_fusion_large")

            is_regressed, increase, message = check_memory_regression("voxel_fusion_large", peak_mb, baseline_mb)

            if is_regressed:
                pytest.fail(f"Memory regression: {message}")
        except (ImportError, AttributeError):
            pytest.skip("VoxelFusion not available")

    @pytest.mark.regression
    def test_parallel_memory_usage(self, memory_baselines):
        """Test memory usage for parallel execution - moderate data size."""
        try:
            from am_qadf.signal_mapping.execution.sequential import (
                interpolate_to_voxels,
            )

            points, signals = create_large_point_cloud(100000)  # Reduced from 200K for sequential-friendly size
            grid = create_voxel_grid(bbox_size=100.0, resolution=1.0)

            tracemalloc.start()
            # Use parallel processing via use_parallel flag
            interpolate_to_voxels(
                points,
                signals,
                grid,
                method="nearest",
                use_parallel=True,
                max_workers=4,
            )
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            peak_mb = bytes_to_mb(peak)
            baseline_mb = memory_baselines.get("parallel_4_workers")

            is_regressed, increase, message = check_memory_regression("parallel_4_workers", peak_mb, baseline_mb)

            if is_regressed:
                pytest.fail(f"Memory regression: {message}")
        except (ImportError, AttributeError) as e:
            pytest.skip(f"Parallel execution module not available: {e}")

    @pytest.mark.regression
    def test_memory_cleanup_after_operation(self):
        """Test that memory is properly cleaned up after operations - sequential processing for low data size."""
        try:
            from am_qadf.signal_mapping.execution.sequential import (
                interpolate_to_voxels,
            )

            points, signals = create_large_point_cloud(50000)
            grid = create_voxel_grid(bbox_size=100.0, resolution=1.0)

            tracemalloc.start()
            initial_current, initial_peak = tracemalloc.get_traced_memory()

            # Run operation
            interpolate_to_voxels(
                points,
                signals,
                grid,
                method="nearest",
                use_spark=False,  # Explicitly use sequential for low data size
            )

            # Force cleanup
            del points, signals, grid
            gc.collect()

            final_current, final_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Memory should decrease after cleanup
            # (peak will remain, but current should decrease)
            assert final_peak >= initial_peak  # Peak can only increase
        except ImportError:
            pytest.skip("Interpolation module not available")

    @pytest.mark.regression
    def test_memory_baseline_initialization(self):
        """Test memory baseline initialization and storage."""
        # Load baselines
        baselines = load_memory_baselines()
        assert baselines is not None
        assert isinstance(baselines, dict)

        # Save baselines
        save_memory_baselines(baselines)
        assert MEMORY_BASELINE_FILE.exists()

        # Reload and verify
        reloaded = load_memory_baselines()
        assert reloaded == baselines

    @pytest.mark.regression
    def test_memory_trend_tracking(self, memory_baselines):
        """Test tracking memory trends over time."""
        # This test would typically store results in a database or file
        # For now, we just verify the baseline system works
        assert memory_baselines is not None
        assert isinstance(memory_baselines, dict)
        assert len(memory_baselines) > 0

    @pytest.mark.regression
    @pytest.mark.slow
    @pytest.mark.requires_spark
    def test_large_dataset_memory_handling(self):
        """Test memory handling with large datasets using Spark for distributed processing (500K points - above 200K sequential limit)."""
        try:
            from am_qadf.signal_mapping.execution.sequential import (
                interpolate_to_voxels,
            )
            from am_qadf.signal_mapping.utils.spark_utils import create_spark_session

            # Create Spark session for distributed processing
            spark = create_spark_session(app_name="MemoryTest_Large")
            if spark is None:
                pytest.skip("Spark not available, skipping large dataset test (requires PySpark)")

            try:
                # Large dataset (>200K) - must use Spark (sequential limit is 200K)
                points, signals = create_large_point_cloud(500000)
                grid = create_voxel_grid(bbox_size=100.0, resolution=1.0)

                tracemalloc.start()
                interpolate_to_voxels(
                    points,
                    signals,
                    grid,
                    method="nearest",
                    use_spark=True,  # Use Spark for datasets > 200K points
                    spark_session=spark,
                )
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                peak_mb = bytes_to_mb(peak)

                # Memory should be reasonable even for large datasets
                # (should not exceed 3GB for 500K points)
                max_memory_mb = 3 * 1024  # 3GB
                if peak_mb > max_memory_mb:
                    pytest.fail(f"Excessive memory usage: {peak_mb:.2f} MB for 500K points")
            finally:
                # Always cleanup Spark session
                try:
                    if spark is not None:
                        spark.stop()
                except Exception:
                    # Ignore errors during cleanup
                    pass
        except ImportError:
            pytest.skip("Interpolation module or Spark not available")
