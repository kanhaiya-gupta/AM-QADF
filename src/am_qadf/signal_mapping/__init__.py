"""
AM-QADF Signal Mapping Module

Core signal mapping algorithms for converting point-based data to voxel domain.
Supports multiple interpolation methods and execution backends.
"""

from .methods import (
    InterpolationMethod,
    NearestNeighborInterpolation,
    LinearInterpolation,
    IDWInterpolation,
    GaussianKDEInterpolation,
    RBFInterpolation,
)

from .execution import (
    interpolate_to_voxels,
    interpolate_hatching_paths,
    INTERPOLATION_METHODS,
)

__all__ = [
    # Methods
    "InterpolationMethod",
    "NearestNeighborInterpolation",
    "LinearInterpolation",
    "IDWInterpolation",
    "GaussianKDEInterpolation",
    "RBFInterpolation",
    # Execution
    "interpolate_to_voxels",
    "interpolate_hatching_paths",
    "INTERPOLATION_METHODS",
]
