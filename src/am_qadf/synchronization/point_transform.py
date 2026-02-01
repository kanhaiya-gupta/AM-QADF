"""
Point transform - thin C++ wrapper.

Re-exports C++ PointTransformer from am_qadf_native.
All computation is done in C++ (apply 4×4 transform to points).
"""

try:
    try:
        from am_qadf_native.synchronization import PointTransformer
    except ImportError:
        from am_qadf_native import PointTransformer
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
    PointTransformer = None

__all__ = ["PointTransformer", "CPP_AVAILABLE"]
