"""
AM-QADF Correction Module

Geometric distortion correction and calibration.
All core correction operations are now C++ wrappers.
"""

# C++ wrappers (may be None if C++ bindings not available)
try:
    from .geometric_distortion import (
        DistortionModel,
        ScalingModel,
        RotationModel,
        WarpingModel,
        CombinedDistortionModel,
    )
except (ImportError, NotImplementedError):
    DistortionModel = None
    ScalingModel = None
    RotationModel = None
    WarpingModel = None
    CombinedDistortionModel = None

try:
    from .calibration import (
        ReferenceMeasurement,
        CalibrationData,
        CalibrationManager,
    )
except (ImportError, NotImplementedError):
    ReferenceMeasurement = None
    CalibrationData = None
    CalibrationManager = None

try:
    from .validation import (
        AlignmentQuality,
        ValidationMetrics,
        CorrectionValidator,
    )
except (ImportError, NotImplementedError):
    AlignmentQuality = None
    ValidationMetrics = None
    CorrectionValidator = None

__all__ = [
    # Geometric distortion
    "DistortionModel",
    "ScalingModel",
    "RotationModel",
    "WarpingModel",
    "CombinedDistortionModel",
    # Calibration
    "ReferenceMeasurement",
    "CalibrationData",
    "CalibrationManager",
    # Validation
    "AlignmentQuality",
    "ValidationMetrics",
    "CorrectionValidator",
]
