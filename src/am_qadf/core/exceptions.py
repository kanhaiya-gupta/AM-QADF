"""
Framework-Specific Exceptions

Custom exception classes for the AM-QADF framework.
These provide more specific error information than generic Python exceptions.
"""


class AMQADFError(Exception):
    """Base exception for all AM-QADF framework errors."""

    pass


class VoxelGridError(AMQADFError):
    """Exception raised for voxel grid related errors."""

    pass


class SignalMappingError(AMQADFError):
    """Exception raised for signal mapping related errors."""

    pass


class InterpolationError(SignalMappingError):
    """Exception raised for interpolation related errors."""

    pass


class FusionError(AMQADFError):
    """Exception raised for data fusion related errors."""

    pass


class QueryError(AMQADFError):
    """Exception raised for query related errors."""

    pass


class StorageError(AMQADFError):
    """Exception raised for storage related errors."""

    pass


class ValidationError(AMQADFError):
    """Exception raised for validation errors."""

    pass


class ConfigurationError(AMQADFError):
    """Exception raised for configuration errors."""

    pass


class CoordinateSystemError(AMQADFError):
    """Exception raised for coordinate system transformation errors."""

    pass


class QualityAssessmentError(AMQADFError):
    """Exception raised for quality assessment related errors."""

    pass
