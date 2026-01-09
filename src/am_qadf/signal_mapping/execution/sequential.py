"""
Sequential Execution Backend

Sequential (non-parallel) execution of interpolation methods.
This is the default execution mode for signal mapping.
"""

import numpy as np
from typing import Dict, List, Optional, Any
import logging

from ..methods.base import InterpolationMethod
from ..methods.nearest_neighbor import NearestNeighborInterpolation
from ..methods.linear import LinearInterpolation
from ..methods.idw import IDWInterpolation
from ..methods.kde import GaussianKDEInterpolation
from ...voxelization.voxel_grid import VoxelGrid

logger = logging.getLogger(__name__)


# Method registry for easy access
INTERPOLATION_METHODS = {
    "nearest": NearestNeighborInterpolation,
    "linear": LinearInterpolation,
    "idw": IDWInterpolation,
    "gaussian_kde": GaussianKDEInterpolation,
}


def interpolate_to_voxels(
    points: np.ndarray,
    signals: Dict[str, np.ndarray],
    voxel_grid: VoxelGrid,
    method: str = "nearest",
    use_vectorized: bool = True,
    use_parallel: bool = False,
    use_spark: bool = False,
    spark_session: Optional[Any] = None,
    max_workers: Optional[int] = None,
    chunk_size: Optional[int] = None,
    **method_kwargs,
) -> VoxelGrid:
    """
    Interpolate point data to voxel grid.

    Args:
        points: Array of points (N, 3) with (x, y, z) coordinates in mm
        signals: Dictionary mapping signal names to arrays (N,) of values
        voxel_grid: Target voxel grid
        method: Interpolation method ('nearest', 'linear', 'idw', 'gaussian_kde')
        use_vectorized: Whether to use vectorized implementation (default: True)
        use_parallel: Whether to use parallel processing (default: False)
        use_spark: Whether to use Spark for distributed processing (default: False)
        spark_session: SparkSession instance (required if use_spark=True)
        max_workers: Maximum number of worker processes/threads (for parallel processing)
        chunk_size: Number of points per chunk (for parallel processing)
        **method_kwargs: Additional arguments for specific interpolation methods:
            - For 'linear': k_neighbors (int), radius (float)
            - For 'idw': power (float), k_neighbors (int), radius (float)
            - For 'gaussian_kde': bandwidth (float), adaptive (bool)

    Returns:
        VoxelGrid with interpolated data
    """
    if points.shape[1] != 3:
        raise ValueError(f"Points must be shape (N, 3), got {points.shape}")

    # Use Spark if requested
    if use_spark:
        try:
            from .spark import interpolate_to_voxels_spark, PYSPARK_AVAILABLE

            if not PYSPARK_AVAILABLE:
                raise ImportError("PySpark not available. Install with: pip install pyspark")

            if spark_session is None:
                from ..utils.spark_utils import create_spark_session

                spark_session = create_spark_session()
                if spark_session is None:
                    raise RuntimeError("Failed to create Spark session")

            # Prepare voxel grid config
            voxel_grid_config = {
                "bbox_min": tuple(voxel_grid.bbox_min),
                "bbox_max": tuple(voxel_grid.bbox_max),
                "resolution": voxel_grid.resolution,
                "aggregation": voxel_grid.aggregation,
            }

            # Interpolate using Spark
            result_grid = interpolate_to_voxels_spark(
                spark_session,
                points,
                signals,
                voxel_grid_config,
                method=method,
                **method_kwargs,
            )

            # Copy results to original voxel grid
            voxel_grid.voxels = result_grid.voxels
            voxel_grid.available_signals = result_grid.available_signals

            return voxel_grid
        except ImportError as e:
            logger.warning(f"Spark interpolation not available: {e}. Falling back to sequential.")
            use_spark = False

    # Use parallel processing if requested
    if use_parallel and use_vectorized:
        try:
            from .parallel import ParallelInterpolationExecutor

            executor = ParallelInterpolationExecutor(max_workers=max_workers, chunk_size=chunk_size)
            return executor.execute_parallel(method, points, signals, voxel_grid, method_kwargs)
        except ImportError as e:
            logger.warning(f"Parallel processing not available: {e}. Falling back to sequential.")
            use_parallel = False

    if use_vectorized:
        # Use new vectorized architecture
        if method not in INTERPOLATION_METHODS:
            raise ValueError(
                f"Unknown interpolation method: {method}. " f"Available methods: {list(INTERPOLATION_METHODS.keys())}"
            )

        method_class = INTERPOLATION_METHODS[method]
        method_instance = method_class(**method_kwargs)
        return method_instance.interpolate(points, signals, voxel_grid)
    else:
        # Fallback to legacy implementation for backward compatibility
        if method == "nearest":
            return _nearest_neighbor_interpolation_legacy(points, signals, voxel_grid)
        else:
            raise NotImplementedError(
                f"Non-vectorized interpolation method '{method}' not implemented. " "Use use_vectorized=True for this method."
            )


def _nearest_neighbor_interpolation_legacy(
    points: np.ndarray, signals: Dict[str, np.ndarray], voxel_grid: VoxelGrid
) -> VoxelGrid:
    """
    Legacy nearest neighbor interpolation (for backward compatibility).

    This is the original sequential implementation, kept for comparison
    and fallback purposes.
    """
    # Add each point to the voxel grid
    for i in range(len(points)):
        x, y, z = points[i]

        # Extract signal values for this point
        point_signals = {}
        for signal_name, signal_array in signals.items():
            if i < len(signal_array):
                point_signals[signal_name] = float(signal_array[i])

        # Add point to voxel grid
        voxel_grid.add_point(x, y, z, point_signals)

    # Finalize voxel grid (aggregate multiple points per voxel)
    voxel_grid.finalize()

    return voxel_grid


def interpolate_hatching_paths(
    paths: List[np.ndarray],
    signals: Dict[str, List[np.ndarray]],
    voxel_grid: VoxelGrid,
    points_per_mm: float = 10.0,
    interpolation_method: str = "nearest",
    use_parallel: bool = False,
    use_spark: bool = False,
    spark_session: Optional[Any] = None,
    max_workers: Optional[int] = None,
    **method_kwargs,
) -> VoxelGrid:
    """
    Interpolate hatching paths (polylines) to voxel grid.

    This function samples points along each path and interpolates them to voxels.
    Optimized with vectorized path sampling.

    Args:
        paths: List of path arrays, each shape (N, 3) with (x, y, z) coordinates
        signals: Dictionary mapping signal names to lists of arrays (one per path)
        voxel_grid: Target voxel grid
        points_per_mm: Sampling density along paths (points per millimeter)
        interpolation_method: Method to use for interpolation ('nearest', 'linear', 'idw', 'gaussian_kde')
        **method_kwargs: Additional arguments for interpolation method

    Returns:
        VoxelGrid with interpolated data
    """
    all_points = []
    all_signals = {name: [] for name in signals.keys()}

    # Sample points along each path (vectorized where possible)
    for path_idx, path in enumerate(paths):
        if len(path) < 2:
            continue

        # Calculate path length (vectorized)
        segments = path[1:] - path[:-1]
        segment_lengths = np.linalg.norm(segments, axis=1)
        path_length = np.sum(segment_lengths)

        if path_length == 0:
            continue

        num_samples = max(2, int(path_length * points_per_mm))

        # Vectorized point sampling along path
        t_values = np.linspace(0, 1, num_samples)
        cumulative_lengths = np.concatenate([[0], np.cumsum(segment_lengths)])
        cumulative_lengths = cumulative_lengths / path_length  # Normalize

        # Find segment for each t value
        segment_indices = np.searchsorted(cumulative_lengths, t_values) - 1
        segment_indices = np.clip(segment_indices, 0, len(segments) - 1)

        # Interpolate within segments
        segment_t = (t_values - cumulative_lengths[segment_indices]) / (
            cumulative_lengths[segment_indices + 1] - cumulative_lengths[segment_indices] + 1e-10
        )
        segment_t = np.clip(segment_t, 0.0, 1.0)

        # Sample points
        sampled_points = path[segment_indices] + segment_t[:, np.newaxis] * segments[segment_indices]
        all_points.append(sampled_points)

        # Interpolate signals
        for signal_name in signals.keys():
            if path_idx < len(signals[signal_name]):
                signal_path = signals[signal_name][path_idx]
                if len(signal_path) == len(path):
                    # Linear interpolation of signals
                    signal_segments = signal_path[1:] - signal_path[:-1]
                    sampled_signals = signal_path[segment_indices] + segment_t * signal_segments[segment_indices]
                    all_signals[signal_name].append(sampled_signals)
                else:
                    # Fallback: repeat last value
                    all_signals[signal_name].append(
                        np.full(
                            num_samples,
                            signal_path[-1] if len(signal_path) > 0 else 0.0,
                        )
                    )
            else:
                all_signals[signal_name].append(np.zeros(num_samples))

    # Convert to numpy arrays
    if len(all_points) > 0:
        points_array = np.vstack(all_points)
        signals_dict = {name: np.concatenate(values) for name, values in all_signals.items()}

        # Interpolate to voxel grid using specified method
        return interpolate_to_voxels(
            points_array,
            signals_dict,
            voxel_grid,
            method=interpolation_method,
            use_vectorized=True,
            use_parallel=use_parallel,
            use_spark=use_spark,
            spark_session=spark_session,
            max_workers=max_workers,
            **method_kwargs,
        )
    else:
        return voxel_grid
