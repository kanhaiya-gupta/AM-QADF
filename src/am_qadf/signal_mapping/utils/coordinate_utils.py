"""
Coordinate Transformation Utilities

Helper functions for coordinate system transformations in signal mapping.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any


def transform_coordinates(points: np.ndarray, from_system: Dict[str, Any], to_system: Dict[str, Any]) -> np.ndarray:
    """
    Transform points from one coordinate system to another.

    Args:
        points: Array of points (N, 3)
        from_system: Source coordinate system dictionary
        to_system: Target coordinate system dictionary

    Returns:
        Transformed points array (N, 3)
    """
    # This is a placeholder - actual implementation would use CoordinateSystemTransformer
    # For now, return points as-is (identity transformation)
    return points.copy()


def align_to_voxel_grid(
    points: np.ndarray,
    voxel_grid_origin: Tuple[float, float, float],
    voxel_resolution: float,
) -> np.ndarray:
    """
    Align points to voxel grid coordinate system.

    Args:
        points: Array of points (N, 3)
        voxel_grid_origin: Origin of voxel grid (x, y, z)
        voxel_resolution: Voxel resolution in mm

    Returns:
        Aligned points array (N, 3)
    """
    # Translate to grid origin
    aligned = points - np.array(voxel_grid_origin)

    # Optionally scale to voxel units
    # aligned = aligned / voxel_resolution

    return aligned


def get_voxel_centers(
    voxel_indices: np.ndarray,
    voxel_grid_origin: Tuple[float, float, float],
    voxel_resolution: float,
) -> np.ndarray:
    """
    Convert voxel indices to world coordinates (voxel centers).

    Args:
        voxel_indices: Array of voxel indices (N, 3)
        voxel_grid_origin: Origin of voxel grid (x, y, z)
        voxel_resolution: Voxel resolution in mm

    Returns:
        Array of voxel center coordinates (N, 3)
    """
    origin = np.array(voxel_grid_origin)
    centers = origin + (voxel_indices + 0.5) * voxel_resolution
    return centers
