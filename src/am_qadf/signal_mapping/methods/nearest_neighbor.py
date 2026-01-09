"""
Nearest Neighbor Interpolation

Vectorized nearest neighbor interpolation method.
Assigns each point to its nearest voxel and aggregates multiple points per voxel.
"""

import numpy as np
from typing import Dict

from .base import InterpolationMethod
from ...voxelization.voxel_grid import VoxelGrid


from ..utils._performance import performance_monitor


class NearestNeighborInterpolation(InterpolationMethod):
    """
    Vectorized nearest neighbor interpolation.

    Assigns each point to its nearest voxel and aggregates multiple points
    per voxel using mean, max, min, or sum aggregation.
    """

    @performance_monitor
    def interpolate(self, points: np.ndarray, signals: Dict[str, np.ndarray], voxel_grid: VoxelGrid) -> VoxelGrid:
        """
        Vectorized nearest neighbor interpolation.

        Algorithm:
        1. Calculate all voxel indices at once (vectorized)
        2. Group points by voxel using NumPy
        3. Aggregate signals per voxel (vectorized)
        4. Build voxel grid structure in batch
        """
        if len(points) == 0:
            return voxel_grid

        # Step 1: Vectorized voxel index calculation
        voxel_indices = self._world_to_voxel_batch(points, voxel_grid)  # (N, 3)

        # Step 2: Group points by voxel using NumPy
        # Convert to structured array for efficient unique operation
        voxel_indices_structured = (
            np.ascontiguousarray(voxel_indices).view(np.dtype([("i", int), ("j", int), ("k", int)])).flatten()
        )

        unique_voxels, inverse_indices = np.unique(voxel_indices_structured, return_inverse=True)

        # Convert back to regular array
        unique_voxels_array = np.array([(v["i"], v["j"], v["k"]) for v in unique_voxels], dtype=int)

        # Step 3: Aggregate signals per voxel (vectorized)
        voxel_data = {}
        for voxel_idx, unique_voxel in enumerate(unique_voxels_array):
            voxel_key = tuple(unique_voxel)
            point_mask = inverse_indices == voxel_idx

            # Aggregate signals for this voxel
            aggregated_signals = {}
            for signal_name, signal_array in signals.items():
                if len(signal_array) == len(points):
                    voxel_signal_values = signal_array[point_mask]
                    if len(voxel_signal_values) > 0:
                        # Use aggregation method from voxel_grid
                        if voxel_grid.aggregation == "mean":
                            aggregated_signals[signal_name] = float(np.mean(voxel_signal_values))
                        elif voxel_grid.aggregation == "max":
                            aggregated_signals[signal_name] = float(np.max(voxel_signal_values))
                        elif voxel_grid.aggregation == "min":
                            aggregated_signals[signal_name] = float(np.min(voxel_signal_values))
                        elif voxel_grid.aggregation == "sum":
                            aggregated_signals[signal_name] = float(np.sum(voxel_signal_values))
                        else:
                            aggregated_signals[signal_name] = float(np.mean(voxel_signal_values))

            voxel_data[voxel_key] = {
                "signals": aggregated_signals,
                "count": int(np.sum(point_mask)),
            }

        # Step 4: Build voxel grid structure (batch)
        self._build_voxel_grid_batch(voxel_grid, voxel_data)

        return voxel_grid
