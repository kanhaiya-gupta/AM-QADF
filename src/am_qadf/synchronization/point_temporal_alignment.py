"""
Point temporal alignment - thin C++ wrapper.

Re-exports C++ PointTemporalAlignment and LayerAlignmentResult from am_qadf_native.
All computation is done in C++.
"""

try:
    try:
        from am_qadf_native.synchronization import (
            PointTemporalAlignment,
            LayerAlignmentResult,
        )
    except ImportError:
        from am_qadf_native import (
            PointTemporalAlignment,
            LayerAlignmentResult,
        )
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
    PointTemporalAlignment = None
    LayerAlignmentResult = None

__all__ = ["PointTemporalAlignment", "LayerAlignmentResult", "CPP_AVAILABLE"]
