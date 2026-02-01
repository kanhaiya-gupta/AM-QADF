"""
Point transformation estimate - thin C++ wrapper.

Re-exports C++ TransformationComputer and related types from am_qadf_native.
All computation is done in C++ (bbox-corner fit, Kabsch/Umeyama, RANSAC).
"""

try:
    try:
        from am_qadf_native.synchronization import (
            TransformationComputer,
            RANSACResult,
            TransformationQuality,
            BboxFitCandidate,
            ScaleTranslationRotation,
        )
    except ImportError:
        from am_qadf_native import (
            TransformationComputer,
            RANSACResult,
            TransformationQuality,
            BboxFitCandidate,
            ScaleTranslationRotation,
        )
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
    TransformationComputer = None
    RANSACResult = None
    TransformationQuality = None
    BboxFitCandidate = None
    ScaleTranslationRotation = None

__all__ = [
    "TransformationComputer",
    "RANSACResult",
    "TransformationQuality",
    "BboxFitCandidate",
    "ScaleTranslationRotation",
    "CPP_AVAILABLE",
]
