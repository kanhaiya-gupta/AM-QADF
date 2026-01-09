"""
AM-QADF Anomaly Detection Utils Module

Utilities and helpers for anomaly detection.
"""

from .preprocessing import (
    PreprocessingConfig,
    DataPreprocessor,
    extract_features_from_fused_data,
)

from .voxel_detection import (
    VoxelAnomalyResult,
    VoxelAnomalyDetector,
)

# Synthetic anomalies will be imported when available
try:
    from .synthetic_anomalies import (
        SyntheticAnomalyGenerator,
        AnomalyInjectionConfig,
        AnomalyInjectionType,
    )

    SYNTHETIC_AVAILABLE = True
except ImportError:
    SYNTHETIC_AVAILABLE = False
    SyntheticAnomalyGenerator = None
    AnomalyInjectionConfig = None
    AnomalyInjectionType = None

__all__ = [
    # Preprocessing
    "PreprocessingConfig",
    "DataPreprocessor",
    "extract_features_from_fused_data",
    # Voxel detection
    "VoxelAnomalyResult",
    "VoxelAnomalyDetector",
]

# Add synthetic if available
if SYNTHETIC_AVAILABLE:
    __all__.extend(
        [
            "SyntheticAnomalyGenerator",
            "AnomalyInjectionConfig",
            "AnomalyInjectionType",
        ]
    )
