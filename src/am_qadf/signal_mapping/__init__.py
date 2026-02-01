"""
AM-QADF Signal Mapping Module

Core signal mapping algorithms for converting point-based data to voxel domain.
All interpolation methods are now thin wrappers around C++ implementations.
"""

from .methods import InterpolationMethod

# Import C++ wrappers (may be None if C++ bindings not available)
try:
    from .methods import NearestNeighborInterpolation
except (ImportError, NotImplementedError):
    NearestNeighborInterpolation = None

try:
    from .methods import LinearInterpolation
except (ImportError, NotImplementedError):
    LinearInterpolation = None

try:
    from .methods import IDWInterpolation
except (ImportError, NotImplementedError):
    IDWInterpolation = None

try:
    from .methods import GaussianKDEInterpolation
except (ImportError, NotImplementedError):
    GaussianKDEInterpolation = None

try:
    from .methods import RBFInterpolation
except (ImportError, NotImplementedError):
    RBFInterpolation = None

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
