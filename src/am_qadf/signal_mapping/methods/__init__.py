"""
Signal Mapping Methods

Interpolation method implementations for signal mapping.
All methods are now thin wrappers around C++ implementations.
"""

from .base import InterpolationMethod

# C++ wrappers - all methods now call C++ implementations
try:
    from .nearest_neighbor import NearestNeighborInterpolation
except (NotImplementedError, ImportError):
    NearestNeighborInterpolation = None

try:
    from .linear import LinearInterpolation
except (NotImplementedError, ImportError):
    LinearInterpolation = None

try:
    from .kde import GaussianKDEInterpolation
except (NotImplementedError, ImportError):
    GaussianKDEInterpolation = None

try:
    from .idw import IDWInterpolation
except (NotImplementedError, ImportError):
    IDWInterpolation = None

try:
    from .rbf import RBFInterpolation
except (NotImplementedError, ImportError):
    RBFInterpolation = None

__all__ = [
    "InterpolationMethod",
    "NearestNeighborInterpolation",
    "LinearInterpolation",
    "GaussianKDEInterpolation",
    "IDWInterpolation",
    "RBFInterpolation",
]
