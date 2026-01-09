"""
Anomaly Detection Types

Core type definitions for anomaly detection module.
"""

from enum import Enum


class AnomalyType(Enum):
    """Types of anomalies that can be detected."""

    POINT = "point"  # Single data point anomaly
    CONTEXTUAL = "contextual"  # Normal in one context, anomalous in another
    COLLECTIVE = "collective"  # Collection of points is anomalous
    SPATIAL = "spatial"  # Anomaly in 3D spatial distribution
    TEMPORAL = "temporal"  # Anomaly in time-series pattern
