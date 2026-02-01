"""
AM-QADF Voxelization Module

Voxel domain processing including:
- Core voxel grid structure
- Adaptive resolution grids (C++ wrapper)
- Multi-resolution grid hierarchies (C++ wrapper)

Note: Coordinate system transformations are now in the coordinate_systems module.
"""

from .uniform_resolution import VoxelGrid
# Coordinate systems moved to dedicated module
from ..coordinate_systems import CoordinateSystem, CoordinateSystemType, CoordinateSystemTransformer

# C++ wrappers (may be None if C++ bindings not available)
try:
    from .adaptive_resolution import (
        AdaptiveResolutionGrid,
        SpatialResolutionMap,
        TemporalResolutionMap,
    )
except (ImportError, NotImplementedError):
    AdaptiveResolutionGrid = None
    SpatialResolutionMap = None
    TemporalResolutionMap = None

try:
    from .multi_resolution import MultiResolutionGrid, ResolutionLevel
except (ImportError, NotImplementedError):
    MultiResolutionGrid = None
    ResolutionLevel = None

# Unified geometry voxelization (STL + Hatching) - PRIMARY INTERFACE
try:
    from .geometry_voxelizer import (
        create_voxel_grid_from_stl_and_hatching,
        export_to_paraview,
        get_stl_bounding_box,  # Utility function
    )
    UNIFIED_VOXELIZATION_AVAILABLE = True
except (ImportError, NotImplementedError):
    UNIFIED_VOXELIZATION_AVAILABLE = False
    create_voxel_grid_from_stl_and_hatching = None
    export_to_paraview = None
    get_stl_bounding_box = None

__all__ = [
    # Core voxel grid
    "VoxelGrid",
    # Coordinate systems
    "CoordinateSystem",
    "CoordinateSystemType",
    "CoordinateSystemTransformer",
    # Unified geometry voxelization (STL + Hatching) - PRIMARY INTERFACE
    "create_voxel_grid_from_stl_and_hatching",
    "export_to_paraview",
    "get_stl_bounding_box",  # Utility function
]

# Add C++ wrappers if available
if AdaptiveResolutionGrid is not None:
    __all__.extend([
        "AdaptiveResolutionGrid",
        "SpatialResolutionMap",
        "TemporalResolutionMap",
    ])

if MultiResolutionGrid is not None:
    __all__.extend([
        "MultiResolutionGrid",
        "ResolutionLevel",
    ])
