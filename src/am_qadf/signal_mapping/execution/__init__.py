"""
Signal Mapping Execution - Thin Wrapper for C++

This module provides a thin Python interface to C++ signal mapping implementations.
All core computation is done in C++ (am_qadf_native).
"""

from .sequential import (
    interpolate_to_voxels,
    interpolate_hatching_paths,
    INTERPOLATION_METHODS,
)

__all__ = [
    "interpolate_to_voxels",
    "interpolate_hatching_paths",
    "INTERPOLATION_METHODS",
]
