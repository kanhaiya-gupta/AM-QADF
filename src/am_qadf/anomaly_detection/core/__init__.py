"""
AM-QADF Anomaly Detection Core Module

Core abstractions and base classes for anomaly detection.
"""

from .base_detector import (
    BaseAnomalyDetector,
    AnomalyDetectionResult,
    AnomalyDetectionConfig,
)

from .types import (
    AnomalyType,
)

__all__ = [
    # Base classes
    "BaseAnomalyDetector",
    "AnomalyDetectionResult",
    "AnomalyDetectionConfig",
    # Types
    "AnomalyType",
]
