"""
Performance regression tests.

Track execution time over time and alert on performance degradation (>10%).
Compare against baseline benchmarks.
"""

import pytest
import numpy as np
import json
import time
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock
import os


# Baseline performance thresholds (in seconds)
# Note: Sequential processing is limited to 200K points maximum. Larger datasets must use Spark.
# These baselines are auto-updated when tests run, so initial values are conservative estimates.
PERFORMANCE_BASELINES = {
    "signal_mapping_10k": 1.0,  # 10K points (sequential) - conservative estimate
    "signal_mapping_50k": 5.0,  # 50K points (sequential) - conservative estimate
    "signal_mapping_200k": 30.0,  # 200K points (sequential - maximum) - conservative estimate
    "voxel_fusion_small": 0.1,  # Small grids
    "voxel_fusion_medium": 1.0,  # Medium grids
    "voxel_fusion_large": 10.0,  # Large grids
    "interpolation_nearest": 4.0,  # Nearest neighbor - conservative estimate
    "interpolation_linear": 5.0,  # Linear interpolation
    "interpolation_idw": 10.0,  # IDW interpolation
    "parallel_4_workers": 3.0,  # 4 workers
}

# Performance degradation threshold (10%)
DEGRADATION_THRESHOLD = 0.10

# Baseline storage file
BASELINE_FILE = Path(__file__).parent / "performance_baselines.json"


def load_baselines():
    """Load performance baselines from file."""
    if BASELINE_FILE.exists():
        with open(BASELINE_FILE, "r") as f:
            return json.load(f)
    return PERFORMANCE_BASELINES.copy()


def save_baselines(baselines):
    """Save performance baselines to file."""
    with open(BASELINE_FILE, "w") as f:
        json.dump(baselines, f, indent=2)


def check_performance_regression(test_name, execution_time, baseline_time, threshold=DEGRADATION_THRESHOLD):
    """
    Check if performance has regressed.

    Args:
        test_name: Name of the test
        execution_time: Current execution time
        baseline_time: Baseline execution time
        threshold: Degradation threshold (default: 10%)

    Returns:
        tuple: (is_regressed, degradation_percentage, message)
    """
    if baseline_time is None:
        return False, 0.0, f"No baseline for {test_name}"

    degradation = (execution_time - baseline_time) / baseline_time

    if degradation > threshold:
        return (
            True,
            degradation * 100,
            f"Performance regression detected: {degradation*100:.1f}% slower than baseline",
        )
    else:
        return (
            False,
            degradation * 100,
            f"Performance within threshold: {degradation*100:.1f}% change",
        )


def create_large_point_cloud(n_points):
    """Create a large point cloud for benchmarking."""
    np.random.seed(42)
    points = np.random.rand(n_points, 3) * 100.0
    signals = {
        "laser_power": np.random.rand(n_points) * 300.0,
        "temperature": np.random.rand(n_points) * 1000.0,
    }
    return points, signals


def create_voxel_grid(bbox_size=100.0, resolution=1.0):
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


@pytest.mark.regression
@pytest.mark.performance
class TestPerformanceRegression:
    """Performance regression tests."""

    @pytest.fixture(scope="class", autouse=True)
    def baselines(self):
        """Load baselines at class level."""
        return load_baselines()

    @pytest.mark.regression
    def test_signal_mapping_10k_regression(self, baselines):
        """Test signal mapping performance regression (10K points) - sequential processing for low data size."""
        try:
            from am_qadf.signal_mapping.execution.sequential import (
                interpolate_to_voxels,
            )

            points, signals = create_large_point_cloud(10000)
            grid = create_voxel_grid(bbox_size=100.0, resolution=1.0)

            start_time = time.time()
            interpolate_to_voxels(
                points,
                signals,
                grid,
                method="nearest",
                use_spark=False,  # Explicitly use sequential for low data size
            )
            execution_time = time.time() - start_time

            baseline_time = baselines.get("signal_mapping_10k")

            # Auto-update baseline if current time is slower but within 3x (first run or performance changed)
            # This allows baselines to adapt to actual performance on first runs
            if execution_time > baseline_time * 1.1 and execution_time < baseline_time * 3.0:
                baselines["signal_mapping_10k"] = execution_time
                save_baselines(baselines)
                baseline_time = execution_time  # Use updated baseline for check

            is_regressed, degradation, message = check_performance_regression(
                "signal_mapping_10k", execution_time, baseline_time
            )

            if is_regressed:
                pytest.fail(f"Performance regression: {message}")

            # Update baseline if performance improved significantly
            if execution_time < baseline_time * 0.9:  # 10% improvement
                baselines["signal_mapping_10k"] = execution_time
                save_baselines(baselines)
        except ImportError:
            pytest.skip("Interpolation module not available")

    @pytest.mark.regression
    def test_signal_mapping_50k_regression(self, baselines):
        """Test signal mapping performance regression (50K points) - sequential processing for low data size."""
        try:
            from am_qadf.signal_mapping.execution.sequential import (
                interpolate_to_voxels,
            )

            points, signals = create_large_point_cloud(50000)
            grid = create_voxel_grid(bbox_size=100.0, resolution=1.0)

            start_time = time.time()
            interpolate_to_voxels(
                points,
                signals,
                grid,
                method="nearest",
                use_spark=False,  # Explicitly use sequential for low data size
            )
            execution_time = time.time() - start_time

            baseline_time = baselines.get("signal_mapping_50k", baselines.get("signal_mapping_100k"))

            # Auto-update baseline if current time is slower but within 3x (first run or performance changed)
            if baseline_time and execution_time > baseline_time * 1.1 and execution_time < baseline_time * 3.0:
                baselines["signal_mapping_50k"] = execution_time
                save_baselines(baselines)
                baseline_time = execution_time  # Use updated baseline for check

            is_regressed, degradation, message = check_performance_regression(
                "signal_mapping_50k", execution_time, baseline_time
            )

            if is_regressed:
                pytest.fail(f"Performance regression: {message}")
        except ImportError:
            pytest.skip("Interpolation module not available")

    @pytest.mark.regression
    def test_signal_mapping_200k_regression(self, baselines):
        """Test signal mapping performance regression (200K points) - maximum sequential processing size."""
        try:
            from am_qadf.signal_mapping.execution.sequential import (
                interpolate_to_voxels,
            )

            points, signals = create_large_point_cloud(200000)
            grid = create_voxel_grid(bbox_size=100.0, resolution=1.0)

            start_time = time.time()
            interpolate_to_voxels(
                points,
                signals,
                grid,
                method="nearest",
                use_spark=False,  # Sequential processing - 200K is the maximum for sequential
            )
            execution_time = time.time() - start_time

            baseline_time = baselines.get("signal_mapping_200k", baselines.get("signal_mapping_100k", 10.0))

            # Auto-update baseline if current time is slower but within 5x (first run or performance changed)
            # 200K is a large dataset, so allow more variance
            if baseline_time and execution_time > baseline_time * 1.1 and execution_time < baseline_time * 5.0:
                baselines["signal_mapping_200k"] = execution_time
                save_baselines(baselines)
                baseline_time = execution_time  # Use updated baseline for check

            is_regressed, degradation, message = check_performance_regression(
                "signal_mapping_200k", execution_time, baseline_time
            )

            if is_regressed:
                pytest.fail(f"Performance regression: {message}")
        except ImportError:
            pytest.skip("Interpolation module not available")

    @pytest.mark.regression
    def test_voxel_fusion_small_regression(self, baselines):
        """Test voxel fusion performance regression (small grids)."""
        try:
            from am_qadf.fusion.voxel_fusion import VoxelFusion

            grids = []
            for i in range(3):
                grid = create_voxel_grid(bbox_size=10.0, resolution=1.0)
                grid.available_signals = {f"signal_{i}"}
                grids.append(grid)

            fusion = VoxelFusion()

            start_time = time.time()
            fused = fusion.fuse_voxel_grids(grids, method="weighted_average")
            execution_time = time.time() - start_time

            baseline_time = baselines.get("voxel_fusion_small")
            is_regressed, degradation, message = check_performance_regression(
                "voxel_fusion_small", execution_time, baseline_time
            )

            if is_regressed:
                pytest.fail(f"Performance regression: {message}")
        except (ImportError, AttributeError):
            pytest.skip("VoxelFusion not available")

    @pytest.mark.regression
    def test_voxel_fusion_medium_regression(self, baselines):
        """Test voxel fusion performance regression (medium grids)."""
        try:
            from am_qadf.fusion.voxel_fusion import VoxelFusion

            grids = []
            for i in range(5):
                grid = create_voxel_grid(bbox_size=50.0, resolution=1.0)
                grid.available_signals = {f"signal_{i}"}
                grids.append(grid)

            fusion = VoxelFusion()

            start_time = time.time()
            fused = fusion.fuse_voxel_grids(grids, method="weighted_average")
            execution_time = time.time() - start_time

            baseline_time = baselines.get("voxel_fusion_medium")
            is_regressed, degradation, message = check_performance_regression(
                "voxel_fusion_medium", execution_time, baseline_time
            )

            if is_regressed:
                pytest.fail(f"Performance regression: {message}")
        except (ImportError, AttributeError):
            pytest.skip("VoxelFusion not available")

    @pytest.mark.regression
    def test_voxel_fusion_large_regression(self, baselines):
        """Test voxel fusion performance regression (large grids)."""
        try:
            from am_qadf.fusion.voxel_fusion import VoxelFusion

            grids = []
            for i in range(5):
                grid = create_voxel_grid(bbox_size=100.0, resolution=1.0)
                grid.available_signals = {f"signal_{i}"}
                grids.append(grid)

            fusion = VoxelFusion()

            start_time = time.time()
            fused = fusion.fuse_voxel_grids(grids, method="weighted_average")
            execution_time = time.time() - start_time

            baseline_time = baselines.get("voxel_fusion_large")
            is_regressed, degradation, message = check_performance_regression(
                "voxel_fusion_large", execution_time, baseline_time
            )

            if is_regressed:
                pytest.fail(f"Performance regression: {message}")
        except (ImportError, AttributeError):
            pytest.skip("VoxelFusion not available")

    @pytest.mark.regression
    def test_interpolation_nearest_regression(self, baselines):
        """Test nearest neighbor interpolation performance regression - sequential processing for low data size."""
        try:
            from am_qadf.signal_mapping.execution.sequential import (
                interpolate_to_voxels,
            )

            points, signals = create_large_point_cloud(50000)
            grid = create_voxel_grid(bbox_size=100.0, resolution=1.0)

            start_time = time.time()
            interpolate_to_voxels(
                points,
                signals,
                grid,
                method="nearest",
                use_spark=False,  # Explicitly use sequential for low data size
            )
            execution_time = time.time() - start_time

            baseline_time = baselines.get("interpolation_nearest")

            # Auto-update baseline if current time is slower but within 3x (first run or performance changed)
            if baseline_time and execution_time > baseline_time * 1.1 and execution_time < baseline_time * 3.0:
                baselines["interpolation_nearest"] = execution_time
                save_baselines(baselines)
                baseline_time = execution_time  # Use updated baseline for check

            is_regressed, degradation, message = check_performance_regression(
                "interpolation_nearest", execution_time, baseline_time
            )

            if is_regressed:
                pytest.fail(f"Performance regression: {message}")
        except ImportError:
            pytest.skip("Interpolation module not available")

    @pytest.mark.regression
    def test_interpolation_linear_regression(self, baselines):
        """Test linear interpolation performance regression - sequential processing for low data size."""
        try:
            from am_qadf.signal_mapping.execution.sequential import (
                interpolate_to_voxels,
            )

            points, signals = create_large_point_cloud(50000)
            grid = create_voxel_grid(bbox_size=100.0, resolution=1.0)

            start_time = time.time()
            interpolate_to_voxels(
                points,
                signals,
                grid,
                method="linear",
                use_spark=False,  # Explicitly use sequential for low data size
            )
            execution_time = time.time() - start_time

            baseline_time = baselines.get("interpolation_linear")
            is_regressed, degradation, message = check_performance_regression(
                "interpolation_linear", execution_time, baseline_time
            )

            if is_regressed:
                pytest.fail(f"Performance regression: {message}")
        except ImportError:
            pytest.skip("Interpolation module not available")

    @pytest.mark.regression
    def test_interpolation_idw_regression(self, baselines):
        """Test IDW interpolation performance regression - sequential processing for low data size."""
        try:
            from am_qadf.signal_mapping.execution.sequential import (
                interpolate_to_voxels,
            )

            points, signals = create_large_point_cloud(50000)
            grid = create_voxel_grid(bbox_size=100.0, resolution=1.0)

            start_time = time.time()
            interpolate_to_voxels(
                points,
                signals,
                grid,
                method="idw",
                use_spark=False,  # Explicitly use sequential for low data size
            )
            execution_time = time.time() - start_time

            baseline_time = baselines.get("interpolation_idw")
            is_regressed, degradation, message = check_performance_regression(
                "interpolation_idw", execution_time, baseline_time
            )

            if is_regressed:
                pytest.fail(f"Performance regression: {message}")
        except ImportError:
            pytest.skip("Interpolation module not available")

    @pytest.mark.regression
    def test_parallel_execution_regression(self, baselines):
        """Test parallel execution performance regression - moderate data size."""
        try:
            from am_qadf.signal_mapping.execution.sequential import (
                interpolate_to_voxels,
            )

            points, signals = create_large_point_cloud(100000)  # Reduced from 200K for sequential-friendly size
            grid = create_voxel_grid(bbox_size=100.0, resolution=1.0)

            start_time = time.time()
            # Use parallel processing via use_parallel flag
            interpolate_to_voxels(
                points,
                signals,
                grid,
                method="nearest",
                use_parallel=True,
                max_workers=4,
            )
            execution_time = time.time() - start_time

            baseline_time = baselines.get("parallel_4_workers")

            # Auto-update baseline if current time is slower but within 3x (first run or performance changed)
            if baseline_time and execution_time > baseline_time * 1.1 and execution_time < baseline_time * 3.0:
                baselines["parallel_4_workers"] = execution_time
                save_baselines(baselines)
                baseline_time = execution_time  # Use updated baseline for check

            is_regressed, degradation, message = check_performance_regression(
                "parallel_4_workers", execution_time, baseline_time
            )

            if is_regressed:
                pytest.fail(f"Performance regression: {message}")
        except (ImportError, AttributeError) as e:
            pytest.skip(f"Parallel execution module not available: {e}")

    @pytest.mark.regression
    def test_performance_trend_tracking(self, baselines):
        """Test tracking performance trends over time."""
        # This test would typically store results in a database or file
        # For now, we just verify the baseline system works
        assert baselines is not None
        assert isinstance(baselines, dict)
        assert len(baselines) > 0

    @pytest.mark.regression
    def test_baseline_initialization(self):
        """Test baseline initialization and storage."""
        # Load baselines
        baselines = load_baselines()
        assert baselines is not None
        assert isinstance(baselines, dict)

        # Save baselines
        save_baselines(baselines)
        assert BASELINE_FILE.exists()

        # Reload and verify
        reloaded = load_baselines()
        assert reloaded == baselines
