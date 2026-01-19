"""
AM-QADF Voxelization Module

Voxel domain processing including:
- Core voxel grid structure
- Coordinate system transformations
- Adaptive resolution grids
- Multi-resolution grid hierarchies
"""

from .voxel_grid import VoxelGrid
from .coordinate_systems import CoordinateSystem, CoordinateSystemType
from .adaptive_resolution import (
    AdaptiveResolutionGrid,
    SpatialResolutionMap,
    TemporalResolutionMap,
)
from .multi_resolution import MultiResolutionGrid, ResolutionLevel
from .transformer import CoordinateSystemTransformer

# STL-based voxelization (requires PyVista)
from .stl_voxelizer import (
    create_voxel_grid_from_stl,
    create_voxel_grid_from_stl_with_signals,
    get_stl_bounding_box,
)
STL_VOXELIZATION_AVAILABLE = True

__all__ = [
    # Core voxel grid
    "VoxelGrid",
    # Coordinate systems
    "CoordinateSystem",
    "CoordinateSystemType",
    "CoordinateSystemTransformer",
    # Adaptive resolution
    "AdaptiveResolutionGrid",
    "SpatialResolutionMap",
    "TemporalResolutionMap",
    # Multi-resolution
    "MultiResolutionGrid",
    "ResolutionLevel",
    # STL-based voxelization
    "create_voxel_grid_from_stl",
    "create_voxel_grid_from_stl_with_signals",
    "get_stl_bounding_box",
]
