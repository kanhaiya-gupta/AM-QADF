"""
AM-QADF Synchronization Module

Temporal and spatial alignment of multi-source data.
Thin Python wrappers around C++ (grid_*, point_*).
Alignment results are stored in OpenVDB format with grid metadata.
"""

# Grid path (thin C++ wrappers)
try:
    from .grid_synchronizer import SynchronizationClient, CPP_AVAILABLE
except (ImportError, NotImplementedError):
    SynchronizationClient = None
    CPP_AVAILABLE = False

try:
    from .grid_spatial_alignment import GridSpatialAlignment
except (ImportError, NotImplementedError):
    GridSpatialAlignment = None

try:
    from .grid_temporal_alignment import GridTemporalAlignment
except (ImportError, NotImplementedError):
    GridTemporalAlignment = None

# Point path (re-export C++ types)
try:
    from .point_temporal_alignment import PointTemporalAlignment, LayerAlignmentResult
except (ImportError, NotImplementedError):
    PointTemporalAlignment = None
    LayerAlignmentResult = None

try:
    from .point_transformation_estimate import (
        TransformationComputer,
        RANSACResult,
        TransformationQuality,
        BboxFitCandidate,
        ScaleTranslationRotation,
    )
except (ImportError, NotImplementedError):
    TransformationComputer = None
    RANSACResult = None
    TransformationQuality = None
    BboxFitCandidate = None
    ScaleTranslationRotation = None

try:
    from .point_transformation_validate import (
        TransformationValidator,
        ValidationResult,
        BboxCorrespondenceValidation,
    )
except (ImportError, NotImplementedError):
    TransformationValidator = None
    ValidationResult = None
    BboxCorrespondenceValidation = None

try:
    from .point_transform import PointTransformer
except (ImportError, NotImplementedError):
    PointTransformer = None

try:
    from .point_bounds import UnifiedBoundsComputer, BoundingBox
except (ImportError, NotImplementedError):
    UnifiedBoundsComputer = None
    BoundingBox = None

__all__ = [
    "SynchronizationClient",
    "CPP_AVAILABLE",
    "GridSpatialAlignment",
    "GridTemporalAlignment",
    "PointTemporalAlignment",
    "LayerAlignmentResult",
    "TransformationComputer",
    "RANSACResult",
    "TransformationQuality",
    "BboxFitCandidate",
    "ScaleTranslationRotation",
    "TransformationValidator",
    "ValidationResult",
    "BboxCorrespondenceValidation",
    "PointTransformer",
    "UnifiedBoundsComputer",
    "BoundingBox",
]

