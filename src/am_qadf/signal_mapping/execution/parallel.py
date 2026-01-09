"""
Parallel Interpolation Module

Handles parallel execution of interpolation operations for improved performance.
Supports both parallel source processing and chunked point processing.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import Manager, cpu_count
import logging
import time
import copy

from ..methods import (
    InterpolationMethod,
    NearestNeighborInterpolation,
    LinearInterpolation,
    IDWInterpolation,
    GaussianKDEInterpolation,
)
from .sequential import INTERPOLATION_METHODS

logger = logging.getLogger(__name__)


class ParallelInterpolationExecutor:
    """
    Handles parallel execution for different interpolation methods.

    Provides method-aware parallelization strategies optimized for each
    interpolation method's computational characteristics.
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        chunk_size: Optional[int] = None,
        use_processes: bool = True,
    ):
        """
        Initialize parallel interpolation executor.

        Args:
            max_workers: Maximum number of worker threads/processes
                        If None, uses cpu_count()
            chunk_size: Number of points per chunk for parallel processing
                      If None, auto-calculated based on data size
            use_processes: Whether to use ProcessPoolExecutor (True) or
                          ThreadPoolExecutor (False)
        """
        self.max_workers = max_workers or cpu_count()
        self.chunk_size = chunk_size
        self.use_processes = use_processes

    def _calculate_chunk_size(self, num_points: int, num_workers: int) -> int:
        """
        Calculate optimal chunk size for parallel processing.

        Args:
            num_points: Total number of points
            num_workers: Number of workers

        Returns:
            Optimal chunk size
        """
        if self.chunk_size is not None:
            return self.chunk_size

        # Aim for 10-50 chunks per worker for good load balancing
        target_chunks_per_worker = 20
        total_chunks = num_workers * target_chunks_per_worker
        chunk_size = max(1000, num_points // total_chunks)

        return chunk_size

    def execute_parallel(
        self,
        method: str,
        points: np.ndarray,
        signals: Dict[str, np.ndarray],
        voxel_grid: Any,
        method_kwargs: Optional[Dict] = None,
    ) -> Any:
        """
        Execute interpolation in parallel based on method characteristics.

        Args:
            method: Interpolation method name ('nearest', 'linear', 'idw', 'gaussian_kde')
            points: Array of points (N, 3)
            signals: Dictionary of signal arrays
            voxel_grid: Target voxel grid
            method_kwargs: Additional arguments for interpolation method

        Returns:
            VoxelGrid with interpolated data
        """
        if method not in INTERPOLATION_METHODS:
            raise ValueError(f"Unknown interpolation method: {method}")

        method_kwargs = method_kwargs or {}
        method_class = INTERPOLATION_METHODS[method]
        method_instance = method_class(**method_kwargs)

        # Choose parallelization strategy based on method
        if isinstance(method_instance, NearestNeighborInterpolation):
            return self._parallel_nearest_neighbor(method_instance, points, signals, voxel_grid)
        elif isinstance(method_instance, (LinearInterpolation, IDWInterpolation)):
            return self._parallel_neighbor_based(method_instance, points, signals, voxel_grid)
        elif isinstance(method_instance, GaussianKDEInterpolation):
            return self._parallel_kde(method_instance, points, signals, voxel_grid)
        else:
            # Default: chunked processing
            return self._parallel_chunked(method_instance, points, signals, voxel_grid)

    def _parallel_nearest_neighbor(
        self,
        method: NearestNeighborInterpolation,
        points: np.ndarray,
        signals: Dict[str, np.ndarray],
        voxel_grid: Any,
    ) -> Any:
        """
        Parallel nearest neighbor interpolation.

        Strategy: Chunk points, process in parallel, merge results.
        """
        if len(points) < 10000:
            # Too small for parallelization overhead
            return method.interpolate(points, signals, voxel_grid)

        chunk_size = self._calculate_chunk_size(len(points), self.max_workers)
        chunks = self._create_chunks(points, signals, chunk_size)

        # Process chunks in parallel
        if self.use_processes:
            executor = ProcessPoolExecutor(max_workers=self.max_workers)
        else:
            executor = ThreadPoolExecutor(max_workers=self.max_workers)

        partial_results = []
        with executor as exec:
            futures = []
            for chunk_points, chunk_signals in chunks:
                # Create a copy of voxel grid config for each worker
                future = exec.submit(
                    _process_chunk_nearest,
                    chunk_points,
                    chunk_signals,
                    voxel_grid.bbox_min.copy(),
                    voxel_grid.bbox_max.copy(),
                    voxel_grid.resolution,
                    voxel_grid.aggregation,
                )
                futures.append(future)

            # Collect results
            for future in as_completed(futures):
                try:
                    result = future.result()
                    partial_results.append(result)
                except Exception as e:
                    logger.error(f"Error in parallel chunk processing: {e}", exc_info=True)

        # Merge partial results
        return self._merge_voxel_results(partial_results, voxel_grid)

    def _parallel_neighbor_based(
        self,
        method: Union[LinearInterpolation, IDWInterpolation],
        points: np.ndarray,
        signals: Dict[str, np.ndarray],
        voxel_grid: Any,
    ) -> Any:
        """
        Parallel neighbor-based interpolation (Linear, IDW).

        Strategy: Chunk points and process in parallel (simpler than voxel-based).
        """
        if len(points) < 10000:
            return method.interpolate(points, signals, voxel_grid)

        # Use chunked approach (simpler and more efficient)
        chunk_size = self._calculate_chunk_size(len(points), self.max_workers)
        chunks = self._create_chunks(points, signals, chunk_size)

        # Determine method name and kwargs
        if isinstance(method, LinearInterpolation):
            method_name = "linear"
            method_kwargs = {"k_neighbors": method.k_neighbors, "radius": method.radius}
        else:  # IDWInterpolation
            method_name = "idw"
            method_kwargs = {
                "power": method.power,
                "k_neighbors": method.k_neighbors,
                "radius": method.radius,
            }

        # Process chunks in parallel
        if self.use_processes:
            executor = ProcessPoolExecutor(max_workers=self.max_workers)
        else:
            executor = ThreadPoolExecutor(max_workers=self.max_workers)

        partial_results = []
        with executor as exec:
            futures = []
            for chunk_points, chunk_signals in chunks:
                future = exec.submit(
                    _process_chunk_generic,
                    method_name,
                    method_kwargs,
                    chunk_points,
                    chunk_signals,
                    voxel_grid.bbox_min.copy(),
                    voxel_grid.bbox_max.copy(),
                    voxel_grid.resolution,
                    voxel_grid.aggregation,
                )
                futures.append(future)

            for future in as_completed(futures):
                try:
                    result = future.result()
                    partial_results.append(result)
                except Exception as e:
                    logger.error(f"Error in parallel chunk processing: {e}", exc_info=True)

        return self._merge_voxel_results(partial_results, voxel_grid)

    def _parallel_kde(
        self,
        method: GaussianKDEInterpolation,
        points: np.ndarray,
        signals: Dict[str, np.ndarray],
        voxel_grid: Any,
    ) -> Any:
        """
        Parallel Gaussian KDE interpolation.

        Strategy: Chunk points and process in parallel.
        """
        if len(points) < 5000:
            return method.interpolate(points, signals, voxel_grid)

        # Use chunked approach
        chunk_size = self._calculate_chunk_size(len(points), self.max_workers)
        chunks = self._create_chunks(points, signals, chunk_size)

        method_name = "gaussian_kde"
        method_kwargs = {"bandwidth": method.bandwidth, "adaptive": method.adaptive}

        # Process chunks in parallel
        if self.use_processes:
            executor = ProcessPoolExecutor(max_workers=self.max_workers)
        else:
            executor = ThreadPoolExecutor(max_workers=self.max_workers)

        partial_results = []
        with executor as exec:
            futures = []
            for chunk_points, chunk_signals in chunks:
                future = exec.submit(
                    _process_chunk_generic,
                    method_name,
                    method_kwargs,
                    chunk_points,
                    chunk_signals,
                    voxel_grid.bbox_min.copy(),
                    voxel_grid.bbox_max.copy(),
                    voxel_grid.resolution,
                    voxel_grid.aggregation,
                )
                futures.append(future)

            for future in as_completed(futures):
                try:
                    result = future.result()
                    partial_results.append(result)
                except Exception as e:
                    logger.error(f"Error in parallel KDE processing: {e}", exc_info=True)

        return self._merge_voxel_results(partial_results, voxel_grid)

    def _parallel_chunked(
        self,
        method: InterpolationMethod,
        points: np.ndarray,
        signals: Dict[str, np.ndarray],
        voxel_grid: Any,
    ) -> Any:
        """
        Generic chunked parallel processing.

        Strategy: Chunk points, process in parallel, merge results.
        """
        if len(points) < 10000:
            return method.interpolate(points, signals, voxel_grid)

        # Determine method name (fallback to nearest if unknown)
        method_name = "nearest"
        method_kwargs = {}

        chunk_size = self._calculate_chunk_size(len(points), self.max_workers)
        chunks = self._create_chunks(points, signals, chunk_size)

        if self.use_processes:
            executor = ProcessPoolExecutor(max_workers=self.max_workers)
        else:
            executor = ThreadPoolExecutor(max_workers=self.max_workers)

        partial_results = []
        with executor as exec:
            futures = []
            for chunk_points, chunk_signals in chunks:
                future = exec.submit(
                    _process_chunk_generic,
                    method_name,
                    method_kwargs,
                    chunk_points,
                    chunk_signals,
                    voxel_grid.bbox_min.copy(),
                    voxel_grid.bbox_max.copy(),
                    voxel_grid.resolution,
                    voxel_grid.aggregation,
                )
                futures.append(future)

            for future in as_completed(futures):
                try:
                    result = future.result()
                    partial_results.append(result)
                except Exception as e:
                    logger.error(f"Error in parallel chunk processing: {e}", exc_info=True)

        return self._merge_voxel_results(partial_results, voxel_grid)

    def _create_chunks(
        self, points: np.ndarray, signals: Dict[str, np.ndarray], chunk_size: int
    ) -> List[Tuple[np.ndarray, Dict[str, np.ndarray]]]:
        """Create chunks of points and signals for parallel processing."""
        chunks = []
        num_points = len(points)

        for start_idx in range(0, num_points, chunk_size):
            end_idx = min(start_idx + chunk_size, num_points)
            chunk_points = points[start_idx:end_idx]
            chunk_signals = {name: array[start_idx:end_idx] for name, array in signals.items()}
            chunks.append((chunk_points, chunk_signals))

        return chunks

    def _merge_voxel_results(self, partial_results: List[Dict], voxel_grid: Any) -> Any:
        """
        Merge partial voxel results into final voxel grid.

        Args:
            partial_results: List of dictionaries with voxel data from each chunk
            voxel_grid: Target voxel grid

        Returns:
            VoxelGrid with merged results
        """
        # Combine all voxel data
        merged_voxel_data = {}

        for partial_result in partial_results:
            for voxel_key, voxel_data in partial_result.items():
                if voxel_key not in merged_voxel_data:
                    merged_voxel_data[voxel_key] = {
                        "signals": {},
                        "count": 0,
                        "values": {},  # Store values for aggregation
                    }

                merged = merged_voxel_data[voxel_key]

                # Collect signal values for aggregation
                for signal_name, value in voxel_data["signals"].items():
                    if signal_name not in merged["values"]:
                        merged["values"][signal_name] = []
                    merged["values"][signal_name].append(value)

                merged["count"] += voxel_data["count"]

        # Aggregate signals based on voxel_grid.aggregation
        final_voxel_data = {}
        for voxel_key, merged in merged_voxel_data.items():
            aggregated_signals = {}

            for signal_name, values in merged["values"].items():
                values_array = np.array(values)

                if voxel_grid.aggregation == "mean":
                    aggregated_signals[signal_name] = float(np.mean(values_array))
                elif voxel_grid.aggregation == "max":
                    aggregated_signals[signal_name] = float(np.max(values_array))
                elif voxel_grid.aggregation == "min":
                    aggregated_signals[signal_name] = float(np.min(values_array))
                elif voxel_grid.aggregation == "sum":
                    aggregated_signals[signal_name] = float(np.sum(values_array))
                else:
                    aggregated_signals[signal_name] = float(np.mean(values_array))

            final_voxel_data[voxel_key] = {
                "signals": aggregated_signals,
                "count": merged["count"],
            }

        # Build voxel grid
        voxel_grid._build_voxel_grid_batch(final_voxel_data)

        return voxel_grid


# Standalone functions for multiprocessing (must be at module level)
def _process_chunk_nearest(
    chunk_points: np.ndarray,
    chunk_signals: Dict[str, np.ndarray],
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    resolution: float,
    aggregation: str,
) -> Dict:
    """Process a chunk of points using nearest neighbor (for multiprocessing)."""
    from ...voxelization.voxel_grid import VoxelGrid
    from ..methods import NearestNeighborInterpolation

    # Create temporary voxel grid
    temp_grid = VoxelGrid(
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        resolution=resolution,
        aggregation=aggregation,
    )

    # Interpolate
    method = NearestNeighborInterpolation()
    method.interpolate(chunk_points, chunk_signals, temp_grid)

    # Extract voxel data
    result = {}
    for voxel_key, voxel_data in temp_grid.voxels.items():
        result[voxel_key] = {
            "signals": dict(voxel_data.signals),
            "count": voxel_data.count,
        }

    return result


def _process_chunk_generic(
    method_name: str,
    method_kwargs: Dict,
    chunk_points: np.ndarray,
    chunk_signals: Dict[str, np.ndarray],
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    resolution: float,
    aggregation: str,
) -> Dict:
    """Process a chunk using generic method (for multiprocessing)."""
    from ...voxelization.voxel_grid import VoxelGrid

    from ..execution.sequential import INTERPOLATION_METHODS

    method_class = INTERPOLATION_METHODS[method_name]
    method_instance = method_class(**method_kwargs)

    temp_grid = VoxelGrid(
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        resolution=resolution,
        aggregation=aggregation,
    )

    method_instance.interpolate(chunk_points, chunk_signals, temp_grid)

    result = {}
    for voxel_key, voxel_data in temp_grid.voxels.items():
        result[voxel_key] = {
            "signals": dict(voxel_data.signals),
            "count": voxel_data.count,
        }

    return result
