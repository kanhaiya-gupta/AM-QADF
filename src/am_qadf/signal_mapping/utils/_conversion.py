"""
Conversion utilities for signal mapping wrappers.

Converts between Python VoxelGrid and C++ FloatGrid.
"""

import numpy as np
from typing import Dict

try:
    from am_qadf_native import numpy_to_openvdb, openvdb_to_numpy
    from am_qadf_native.signal_mapping import Point, numpy_to_points
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
    numpy_to_points = None

from ...voxelization.uniform_resolution import VoxelGrid
from ...core.entities import VoxelData


def voxelgrid_to_floatgrid(voxel_grid: VoxelGrid, signal_name: str, default: float = 0.0):
    """
    Convert Python VoxelGrid to C++ FloatGrid for a specific signal.
    
    Args:
        voxel_grid: Python VoxelGrid instance
        signal_name: Name of signal to extract
        default: Default value for empty voxels
        
    Returns:
        C++ FloatGridPtr (as Python object from bindings)
    """
    if not CPP_AVAILABLE:
        raise ImportError("C++ bindings not available")
    
    # Get signal as numpy array
    signal_array = voxel_grid.get_signal_array(signal_name, default=default)
    
    # Convert to OpenVDB FloatGrid
    # Note: numpy_to_openvdb creates grid at origin (0,0,0) with indices (0,0,0) to (dims-1)
    # The C++ mapper expects points in world coordinates and uses grid->transform() to convert
    # We need to set the grid transform to translate by bbox_min so that:
    # - Grid index (0,0,0) corresponds to world coordinate bbox_min
    # - Grid index (i,j,k) corresponds to world coordinate bbox_min + (i,j,k) * resolution
    
    openvdb_grid = numpy_to_openvdb(signal_array, voxel_grid.resolution)
    
    # TODO: Set grid transform translation to account for bbox_min
    # This requires extending numpy_to_openvdb or accessing transform from Python
    # For now, we'll adjust points to be relative to grid origin before calling mapper
    # OR extend the C++ converter to accept bbox_min parameter
    
    return openvdb_grid


def floatgrid_to_voxelgrid(openvdb_grid, voxel_grid: VoxelGrid, signal_name: str):
    """
    Copy C++ FloatGrid directly into the voxel grid (C++ only, no Python/numpy round-trip).

    Args:
        openvdb_grid: C++ FloatGridPtr (from bindings, e.g. after IDW/KDE mapper)
        voxel_grid: Python VoxelGrid to update
        signal_name: Name of signal to update
    """
    if not CPP_AVAILABLE:
        raise ImportError("C++ bindings not available")

    # Direct C++: copy OpenVDB FloatGrid into UniformVoxelGrid (no numpy conversion)
    cpp_grid = voxel_grid._get_or_create_grid(signal_name)
    cpp_grid.copy_from_grid(openvdb_grid)
    voxel_grid.available_signals.add(signal_name)


def points_to_cpp_points(points: np.ndarray, bbox_min):
    """
    Convert numpy points array to C++ Point vector (C++ only, no Python loops).
    
    Args:
        points: Array of shape (N, 3) with (x, y, z) coordinates in world space
        bbox_min: Required bbox_min offset to adjust points relative to grid origin
        
    Returns:
        std::vector<Point> (from C++ bindings, converted automatically by pybind11)
        
    Raises:
        ValueError: If bbox_min is None or not provided
    """
    if not CPP_AVAILABLE or numpy_to_points is None:
        raise ImportError("C++ bindings not available")
    
    if bbox_min is None:
        raise ValueError("bbox_min is required for coordinate transformation")
    
    # Ensure contiguous float32 array (C++ expects float*)
    points_array = np.ascontiguousarray(points, dtype=np.float32)
    bbox_min_arr = np.ascontiguousarray(bbox_min, dtype=np.float32)
    
    # Call C++ converter (all processing in C++, no Python loops)
    return numpy_to_points(points_array, bbox_min_arr)
