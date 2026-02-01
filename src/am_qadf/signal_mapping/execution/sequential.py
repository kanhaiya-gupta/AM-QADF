"""
Signal Mapping Execution - Thin Wrapper for C++

This module provides a thin Python interface to C++ signal mapping implementations.
All core logic is in C++ (am_qadf_native).
"""

import numpy as np
from typing import Dict, List, Optional, Any
import logging

from ...voxelization.uniform_resolution import VoxelGrid

logger = logging.getLogger(__name__)

# Import all interpolation methods (all are C++ wrappers - required, no fallback)
from ..methods import (
    NearestNeighborInterpolation,
    LinearInterpolation,
    GaussianKDEInterpolation,
    IDWInterpolation,
    RBFInterpolation,
)

# Method registry - maps to C++ implementations via wrappers
# All methods are required (C++ bindings must be available)
INTERPOLATION_METHODS = {
    "nearest": NearestNeighborInterpolation,
    "linear": LinearInterpolation,
    "gaussian_kde": GaussianKDEInterpolation,
    "idw": IDWInterpolation,
    "rbf": RBFInterpolation,
}


def interpolate_to_voxels(
    points: np.ndarray,
    signals: Dict[str, np.ndarray],
    voxel_grid: VoxelGrid,
    method: str = "nearest",
    **method_kwargs,
) -> VoxelGrid:
    """
    Interpolate point data to voxel grid using C++ implementations.

    This is a thin wrapper that calls C++ interpolation methods via Python wrappers.
    All core computation is done in C++.

    Args:
        points: Array of points (N, 3) with (x, y, z) coordinates in mm
        signals: Dictionary mapping signal names to arrays (N,) of values
        voxel_grid: Target voxel grid
        method: Interpolation method ('nearest', 'linear', 'idw', 'rbf', 'gaussian_kde')
        **method_kwargs: Additional arguments for specific interpolation methods

    Returns:
        VoxelGrid with interpolated data
    """
    if points.shape[1] != 3:
        raise ValueError(f"Points must be shape (N, 3), got {points.shape}")

    if method not in INTERPOLATION_METHODS:
        raise ValueError(
            f"Unknown interpolation method: {method}. "
            f"Available methods: {list(INTERPOLATION_METHODS.keys())}"
        )

    # Get interpolation method class and create instance
    method_class = INTERPOLATION_METHODS[method]
    method_instance = method_class(**method_kwargs)
    
    # Call C++ implementation via wrapper
    return method_instance.interpolate(points, signals, voxel_grid)


def interpolate_hatching_paths(
    paths: List[np.ndarray],
    signals: Dict[str, List[np.ndarray]],
    voxel_grid: VoxelGrid,
    points_per_mm: float = 10.0,
    interpolation_method: str = "nearest",
    **method_kwargs,
) -> VoxelGrid:
    """
    Interpolate hatching paths (polylines) to voxel grid.

    Samples points along paths and interpolates them using C++ implementations.

    Args:
        paths: List of path arrays, each shape (N, 3) with (x, y, z) coordinates
        signals: Dictionary mapping signal names to lists of arrays (one per path)
        voxel_grid: Target voxel grid
        points_per_mm: Sampling density along paths (points per millimeter)
        interpolation_method: Method to use for interpolation
        **method_kwargs: Additional arguments for interpolation method

    Returns:
        VoxelGrid with interpolated data
    """
    all_points = []
    all_signals = {name: [] for name in signals.keys()}

    # Sample points along each path
    for path_idx, path in enumerate(paths):
        if len(path) < 2:
            continue

        # Calculate path length
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

    # Convert to numpy arrays and interpolate
    if len(all_points) > 0:
        points_array = np.vstack(all_points)
        signals_dict = {name: np.concatenate(values) for name, values in all_signals.items()}

        # Interpolate to voxel grid using C++ implementation
        return interpolate_to_voxels(
            points_array,
            signals_dict,
            voxel_grid,
            method=interpolation_method,
            **method_kwargs,
        )
    else:
        return voxel_grid
