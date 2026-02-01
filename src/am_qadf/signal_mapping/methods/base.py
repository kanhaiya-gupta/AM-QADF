"""
Base Interpolation Method

Abstract base class for all interpolation methods.
Provides common functionality and interface for vectorized interpolation.
"""

import numpy as np
from typing import Dict, Tuple, Any
from abc import ABC, abstractmethod

# Import VoxelGrid
from ...voxelization.uniform_resolution import VoxelGrid


class InterpolationMethod(ABC):
    """
    Base class for all interpolation methods.

    Provides common functionality and interface for vectorized interpolation.
    All interpolation methods should inherit from this class.
    
    All coordinate transformations and grid operations are handled by C++.
    This is a pure abstract interface - no Python-side calculations.
    """

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
