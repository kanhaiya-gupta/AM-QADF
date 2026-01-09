"""
Unit tests for parallel execution.

Tests for ParallelInterpolationExecutor and parallel interpolation strategies.
"""

import pytest
import numpy as np
from am_qadf.signal_mapping.execution.parallel import (
    ParallelInterpolationExecutor,
    INTERPOLATION_METHODS,
)
from am_qadf.voxelization.voxel_grid import VoxelGrid


class TestParallelInterpolationExecutor:
    """Test suite for ParallelInterpolationExecutor class."""

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
    def executor(self):
        """Create ParallelInterpolationExecutor instance."""
        return ParallelInterpolationExecutor(max_workers=2, use_processes=False)

    @pytest.mark.unit
    def test_executor_creation_default(self):
        """Test creating ParallelInterpolationExecutor with default parameters."""
        executor = ParallelInterpolationExecutor()

        assert executor.max_workers > 0
        assert executor.chunk_size is None
        assert executor.use_processes is True

    @pytest.mark.unit
    def test_executor_creation_custom(self):
        """Test creating ParallelInterpolationExecutor with custom parameters."""
        executor = ParallelInterpolationExecutor(max_workers=4, chunk_size=1000, use_processes=False)

        assert executor.max_workers == 4
        assert executor.chunk_size == 1000
        assert executor.use_processes is False

    @pytest.mark.unit
    def test_execute_parallel_nearest_neighbor_small(self, executor, voxel_grid):
        """Test parallel execution with nearest neighbor for small dataset."""
        # Small dataset should not use parallelization
        points = np.random.rand(100, 3) * 10.0
        signals = {"power": np.random.rand(100) * 300.0}

        result = executor.execute_parallel("nearest", points, signals, voxel_grid)

        assert result is voxel_grid
        assert len(voxel_grid.voxels) > 0

    @pytest.mark.unit
    def test_execute_parallel_nearest_neighbor_large(self, executor, voxel_grid):
        """Test parallel execution with nearest neighbor for large dataset."""
        # Large dataset should use parallelization
        points = np.random.rand(50000, 3) * 10.0
        signals = {"power": np.random.rand(50000) * 300.0}

        result = executor.execute_parallel("nearest", points, signals, voxel_grid)

        assert result is voxel_grid
        assert len(voxel_grid.voxels) > 0

    @pytest.mark.unit
    def test_execute_parallel_linear(self, executor, voxel_grid):
        """Test parallel execution with linear interpolation."""
        points = np.random.rand(20000, 3) * 10.0
        signals = {"power": np.random.rand(20000) * 300.0}

        result = executor.execute_parallel("linear", points, signals, voxel_grid, method_kwargs={"k_neighbors": 4})

        assert result is voxel_grid
        assert len(voxel_grid.voxels) > 0

    @pytest.mark.unit
    def test_execute_parallel_idw(self, executor, voxel_grid):
        """Test parallel execution with IDW interpolation."""
        points = np.random.rand(20000, 3) * 10.0
        signals = {"power": np.random.rand(20000) * 300.0}

        result = executor.execute_parallel(
            "idw",
            points,
            signals,
            voxel_grid,
            method_kwargs={"power": 2.0, "k_neighbors": 4},
        )

        assert result is voxel_grid
        assert len(voxel_grid.voxels) > 0

    @pytest.mark.unit
    def test_execute_parallel_gaussian_kde(self, executor, voxel_grid):
        """Test parallel execution with Gaussian KDE interpolation."""
        points = np.random.rand(10000, 3) * 10.0
        signals = {"power": np.random.rand(10000) * 300.0}

        result = executor.execute_parallel(
            "gaussian_kde",
            points,
            signals,
            voxel_grid,
            method_kwargs={"bandwidth": 1.0},
        )

        assert result is voxel_grid
        assert len(voxel_grid.voxels) > 0

    @pytest.mark.unit
    def test_execute_parallel_invalid_method(self, executor, voxel_grid):
        """Test parallel execution with invalid method."""
        points = np.array([[5.0, 5.0, 5.0]])
        signals = {"power": np.array([200.0])}

        with pytest.raises(ValueError, match="Unknown interpolation method"):
            executor.execute_parallel("invalid", points, signals, voxel_grid)

    @pytest.mark.unit
    def test_calculate_chunk_size(self, executor):
        """Test chunk size calculation."""
        chunk_size = executor._calculate_chunk_size(100000, 4)

        assert chunk_size >= 1000  # Minimum chunk size
        assert chunk_size <= 100000  # Should not exceed total points

    @pytest.mark.unit
    def test_calculate_chunk_size_custom(self):
        """Test chunk size calculation with custom chunk_size."""
        executor = ParallelInterpolationExecutor(chunk_size=5000)

        chunk_size = executor._calculate_chunk_size(100000, 4)

        assert chunk_size == 5000  # Should use custom value

    @pytest.mark.unit
    def test_execute_parallel_multiple_signals(self, executor, voxel_grid):
        """Test parallel execution with multiple signals."""
        points = np.random.rand(20000, 3) * 10.0
        signals = {
            "power": np.random.rand(20000) * 300.0,
            "speed": np.random.rand(20000) * 200.0,
        }

        result = executor.execute_parallel("nearest", points, signals, voxel_grid)

        assert result is voxel_grid
        assert "power" in voxel_grid.available_signals
        assert "speed" in voxel_grid.available_signals

    @pytest.mark.unit
    def test_execute_parallel_threads_vs_processes(self, voxel_grid):
        """Test parallel execution with threads vs processes."""
        points = np.random.rand(20000, 3) * 10.0
        signals = {"power": np.random.rand(20000) * 300.0}

        # Test with threads
        executor_threads = ParallelInterpolationExecutor(max_workers=2, use_processes=False)
        result_threads = executor_threads.execute_parallel(
            "nearest",
            points,
            signals,
            VoxelGrid(bbox_min=(0.0, 0.0, 0.0), bbox_max=(10.0, 10.0, 10.0), resolution=1.0),
        )

        # Test with processes
        executor_processes = ParallelInterpolationExecutor(max_workers=2, use_processes=True)
        result_processes = executor_processes.execute_parallel(
            "nearest",
            points,
            signals,
            VoxelGrid(bbox_min=(0.0, 0.0, 0.0), bbox_max=(10.0, 10.0, 10.0), resolution=1.0),
        )

        # Both should produce results
        assert len(result_threads.voxels) > 0
        assert len(result_processes.voxels) > 0
