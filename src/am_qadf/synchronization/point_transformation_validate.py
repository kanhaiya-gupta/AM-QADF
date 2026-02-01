"""
Point transformation validate - thin C++ wrapper.

Re-exports C++ TransformationValidator and related types from am_qadf_native.
All computation is done in C++.
"""

try:
    try:
        from am_qadf_native.synchronization import (
            TransformationValidator,
            ValidationResult,
            BboxCorrespondenceValidation,
        )
    except ImportError:
        from am_qadf_native import (
            TransformationValidator,
            ValidationResult,
            BboxCorrespondenceValidation,
        )
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
    TransformationValidator = None
    ValidationResult = None
    BboxCorrespondenceValidation = None

__all__ = [
    "TransformationValidator",
    "ValidationResult",
    "BboxCorrespondenceValidation",
    "CPP_AVAILABLE",
]
