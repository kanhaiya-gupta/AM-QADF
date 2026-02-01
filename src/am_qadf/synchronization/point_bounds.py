"""
Point bounds - thin C++ wrapper.

Re-exports C++ UnifiedBoundsComputer and BoundingBox from am_qadf_native.
All computation is done in C++ (union bounding box from point sets).
"""

try:
    try:
        from am_qadf_native.synchronization import (
            UnifiedBoundsComputer,
            BoundingBox,
        )
    except ImportError:
        from am_qadf_native import (
            UnifiedBoundsComputer,
            BoundingBox,
        )
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
    UnifiedBoundsComputer = None
    BoundingBox = None

__all__ = ["UnifiedBoundsComputer", "BoundingBox", "CPP_AVAILABLE"]
