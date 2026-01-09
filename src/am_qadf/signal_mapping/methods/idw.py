"""
Inverse Distance Weighting (IDW) Interpolation

Vectorized IDW interpolation using power parameter for distance weighting.
Formula: v = sum(s_i / d_i^p) / sum(1 / d_i^p)
"""

import numpy as np
from typing import Dict, Optional

from .base import InterpolationMethod
from ..utils._performance import performance_monitor
from ...voxelization.voxel_grid import VoxelGrid


class IDWInterpolation(InterpolationMethod):
    """
    Vectorized Inverse Distance Weighting (IDW) interpolation.

    Similar to linear interpolation but uses power parameter for distance weighting.
    Formula: v = sum(s_i / d_i^p) / sum(1 / d_i^p)
    """

    def __init__(self, power: float = 2.0, k_neighbors: int = 8, radius: Optional[float] = None):
        """
        Initialize IDW interpolation.

        Args:
            power: Power parameter for distance weighting (typically 2.0)
            k_neighbors: Number of nearest neighbors to use
            radius: Optional maximum search radius
        """
        self.power = power
        self.k_neighbors = k_neighbors
        self.radius = radius

    @performance_monitor
    def interpolate(self, points: np.ndarray, signals: Dict[str, np.ndarray], voxel_grid: VoxelGrid) -> VoxelGrid:
        """
        Vectorized IDW interpolation.

        Formula: v = sum(s_i / d_i^p) / sum(1 / d_i^p)
        """
        if len(points) == 0:
            return voxel_grid

        try:
            from scipy.spatial import cKDTree
        except ImportError:
            raise ImportError("scipy is required for IDW interpolation. " "Install with: pip install scipy")

        # Get unique voxel centers
        voxel_indices = self._world_to_voxel_batch(points, voxel_grid)
        unique_voxels = np.unique(voxel_indices, axis=0)
        voxel_centers = voxel_grid.bbox_min + (unique_voxels + 0.5) * voxel_grid.resolution

        # Build KDTree
        tree = cKDTree(points)

        # Find k nearest neighbors
        if self.radius is not None:
            distances_list, indices_list = tree.query(voxel_centers, k=self.k_neighbors, distance_upper_bound=self.radius)
        else:
            distances_list, indices_list = tree.query(voxel_centers, k=self.k_neighbors)

        # Handle case where k_neighbors > available points
        if len(points) < self.k_neighbors:
            distances_list = distances_list.reshape(-1, len(points))
            indices_list = indices_list.reshape(-1, len(points))

        # Aggregate signals per voxel using IDW
        voxel_data = {}
        for voxel_idx, (voxel_center, nn_indices, nn_distances) in enumerate(zip(voxel_centers, indices_list, distances_list)):
            voxel_key = tuple(unique_voxels[voxel_idx])

            # Filter out invalid neighbors
            valid_mask = np.isfinite(nn_distances) & (nn_distances > 0)
            if not np.any(valid_mask):
                continue

            valid_indices = nn_indices[valid_mask]
            valid_distances = nn_distances[valid_mask]

            # IDW weights: 1 / d^p
            weights = 1.0 / (valid_distances**self.power + 1e-10)
            weights = weights / weights.sum()  # Normalize

            # Aggregate signals
            aggregated_signals = {}
            for signal_name, signal_array in signals.items():
                if len(signal_array) == len(points):
                    nn_signals = signal_array[valid_indices]
                    weighted_mean = np.sum(nn_signals * weights)
                    aggregated_signals[signal_name] = float(weighted_mean)

            voxel_data[voxel_key] = {
                "signals": aggregated_signals,
                "count": len(valid_indices),
            }

        self._build_voxel_grid_batch(voxel_grid, voxel_data)
        return voxel_grid
