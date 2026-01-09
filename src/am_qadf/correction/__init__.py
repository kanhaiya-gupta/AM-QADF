"""
AM-QADF Correction Module

Geometric distortion correction and calibration.
Handles distortion models, calibration data, and validation of corrections.
"""

from .geometric_distortion import (
    DistortionModel,
    ScalingModel,
    RotationModel,
    WarpingModel,
    CombinedDistortionModel,
)

from .calibration import (
    ReferenceMeasurement,
    CalibrationData,
    CalibrationManager,
)

from .validation import (
    AlignmentQuality,
    ValidationMetrics,
    CorrectionValidator,
)

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
