"""
Coordinate Systems Module

General coordinate system transformation utilities used across the entire codebase.
Not specific to voxelization - used by query, fusion, signal mapping, etc.
"""

from .transformer import CoordinateSystemTransformer
from .coordinate_systems import (
    CoordinateSystemType,
    CoordinateSystem,
    CoordinateSystemRegistry,
)

__all__ = [
    "CoordinateSystemTransformer",
    "CoordinateSystemType",
    "CoordinateSystem",
    "CoordinateSystemRegistry",
]
