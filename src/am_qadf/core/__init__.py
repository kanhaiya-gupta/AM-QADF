"""
AM-QADF Core Module

Core domain entities, value objects, and exceptions for the framework.
"""

from .entities import VoxelData
from .value_objects import VoxelCoordinates, QualityMetric
from .exceptions import (
    AMQADFError,
    VoxelGridError,
    SignalMappingError,
    InterpolationError,
    FusionError,
    QueryError,
    StorageError,
    ValidationError,
    ConfigurationError,
    CoordinateSystemError,
    QualityAssessmentError,
)

__all__ = [
    # Entities
    "VoxelData",
    # Value Objects
    "VoxelCoordinates",
    "QualityMetric",
    # Exceptions
    "AMQADFError",
    "VoxelGridError",
    "SignalMappingError",
    "InterpolationError",
    "FusionError",
    "QueryError",
    "StorageError",
    "ValidationError",
    "ConfigurationError",
    "CoordinateSystemError",
    "QualityAssessmentError",
]
