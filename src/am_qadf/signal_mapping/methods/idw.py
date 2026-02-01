"""
Inverse Distance Weighting (IDW) Interpolation - C++ Wrapper

Thin Python wrapper for C++ IDW interpolation implementation.
All core computation is done in C++.
"""

import numpy as np
from typing import Dict

try:
    from am_qadf_native.signal_mapping import IDWMapper
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
    IDWMapper = None

from .base import InterpolationMethod
from ...voxelization.uniform_resolution import VoxelGrid
from ..utils._conversion import (
    voxelgrid_to_floatgrid,
    floatgrid_to_voxelgrid,
    points_to_cpp_points,
)


class IDWInterpolation(InterpolationMethod):
    """
    IDW interpolation - C++ wrapper.
    
    This is a thin wrapper around the C++ IDWMapper implementation.
    All core computation is done in C++.
    """

    def __init__(self, power: float = 2.0, k_neighbors: int = 10):
        """
        Initialize IDW interpolation.

        Args:
            power: IDW power parameter (default: 2.0)
            k_neighbors: Number of nearest neighbors to use (default: 10)
        """
        if not CPP_AVAILABLE:
            raise ImportError(
                "C++ bindings not available. "
                "Please build am_qadf_native with pybind11 bindings."
            )
        self._mapper = IDWMapper(power, k_neighbors)
        self.power = power
        self.k_neighbors = k_neighbors

    def interpolate(self, points: np.ndarray, signals: Dict[str, np.ndarray], voxel_grid: VoxelGrid) -> VoxelGrid:
        """
        Interpolate points to voxel grid using IDW method.

        Args:
            points: Array of points (N, 3) with (x, y, z) coordinates
            signals: Dictionary mapping signal names to arrays (N,) of values
            voxel_grid: Target voxel grid

        Returns:
            VoxelGrid with interpolated data
        """
        if len(points) == 0:
            return voxel_grid

        if len(signals) == 0:
            return voxel_grid

        # Convert points to C++ format
        # Note: Points are in world coordinates, grid is at origin
        # Adjust points to be relative to grid origin (subtract bbox_min)
        points_cpp = points_to_cpp_points(points, bbox_min=voxel_grid.bbox_min)

        # Process each signal separately
        for signal_name, signal_values in signals.items():
            if len(signal_values) != len(points):
                continue

            # Convert signal values to C++ format
            values_cpp = signal_values.astype(np.float32).tolist()

            # Convert VoxelGrid to FloatGrid for this signal
            openvdb_grid = voxelgrid_to_floatgrid(voxel_grid, signal_name, default=0.0)

            # Call C++ mapper
            self._mapper.map(openvdb_grid, points_cpp, values_cpp)

            # Convert FloatGrid back to VoxelGrid
            floatgrid_to_voxelgrid(openvdb_grid, voxel_grid, signal_name)

        return voxel_grid
