"""
Base Interpolation Method

Abstract base class for all interpolation methods.
Provides common functionality and interface for vectorized interpolation.
"""

import numpy as np
from typing import Dict, Tuple, Any
from abc import ABC, abstractmethod

# Import VoxelGrid
from ...voxelization.voxel_grid import VoxelGrid


class InterpolationMethod(ABC):
    """
    Base class for all interpolation methods.

    Provides common functionality and interface for vectorized interpolation.
    All interpolation methods should inherit from this class.
    """

    def _world_to_voxel_batch(self, points: np.ndarray, voxel_grid: VoxelGrid) -> np.ndarray:
        """
        Vectorized voxel index calculation - shared by all methods.

        Args:
            points: Array of points (N, 3)
            voxel_grid: Target voxel grid

        Returns:
            Array of voxel indices (N, 3)
        """
        return voxel_grid._world_to_voxel_batch(points)

    def _get_voxel_centers(self, points: np.ndarray, voxel_grid: VoxelGrid) -> np.ndarray:
        """
        Get voxel centers for all voxels that contain points.

        Args:
            points: Array of points (N, 3)
            voxel_grid: Target voxel grid

        Returns:
            Array of voxel center coordinates (M, 3) where M is number of unique voxels
        """
        # Get unique voxel indices
        voxel_indices = self._world_to_voxel_batch(points, voxel_grid)
        unique_voxels = np.unique(voxel_indices, axis=0)

        # Convert to world coordinates (voxel centers)
        voxel_centers = voxel_grid.bbox_min + (unique_voxels + 0.5) * voxel_grid.resolution
        return voxel_centers

    def _build_voxel_grid_batch(
        self,
        voxel_grid: VoxelGrid,
        voxel_data: Dict[Tuple[int, int, int], Dict[str, Any]],
    ):
        """
        Build voxel grid structure from pre-aggregated data.

        Args:
            voxel_grid: Target voxel grid
            voxel_data: Dictionary mapping voxel keys to aggregated data
        """
        voxel_grid._build_voxel_grid_batch(voxel_data)

    @abstractmethod
    def interpolate(self, points: np.ndarray, signals: Dict[str, np.ndarray], voxel_grid: VoxelGrid) -> VoxelGrid:
        """
        Interpolate points to voxel grid using this method.

        Args:
            points: Array of points (N, 3) with (x, y, z) coordinates
            signals: Dictionary mapping signal names to arrays (N,) of values
            voxel_grid: Target voxel grid

        Returns:
            VoxelGrid with interpolated data
        """
        pass
