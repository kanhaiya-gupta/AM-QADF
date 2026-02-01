"""
Signal Mapping Utilities

Utility functions for signal mapping operations.
"""

# Conversion utilities (for C++ wrappers)
try:
    from ._conversion import (
        voxelgrid_to_floatgrid,
        floatgrid_to_voxelgrid,
        points_to_cpp_points,
    )
    CONVERSION_AVAILABLE = True
except ImportError:
    CONVERSION_AVAILABLE = False

from ._performance import performance_monitor

__all__ = [
    # Conversion utilities
    "voxelgrid_to_floatgrid",
    "floatgrid_to_voxelgrid",
    "points_to_cpp_points",
    # Performance utilities
    "performance_monitor",
]
