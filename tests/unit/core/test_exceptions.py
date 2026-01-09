"""
Unit tests for core exceptions.

Tests for exception hierarchy and behavior.
"""

import pytest
from am_qadf.core.exceptions import (
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


class TestExceptionHierarchy:
    """Test suite for exception hierarchy."""

    @pytest.mark.unit
    def test_amqadf_error_base(self):
        """Test AMQADFError is a base exception."""
        error = AMQADFError("Test error")

        assert isinstance(error, Exception)
        assert str(error) == "Test error"

    @pytest.mark.unit
    def test_voxel_grid_error_inheritance(self):
        """Test VoxelGridError inherits from AMQADFError."""
        error = VoxelGridError("Voxel grid error")

        assert isinstance(error, AMQADFError)
        assert isinstance(error, Exception)
        assert str(error) == "Voxel grid error"

    @pytest.mark.unit
    def test_signal_mapping_error_inheritance(self):
        """Test SignalMappingError inherits from AMQADFError."""
        error = SignalMappingError("Signal mapping error")

        assert isinstance(error, AMQADFError)
        assert isinstance(error, Exception)
        assert str(error) == "Signal mapping error"

    @pytest.mark.unit
    def test_interpolation_error_inheritance(self):
        """Test InterpolationError inherits from SignalMappingError."""
        error = InterpolationError("Interpolation error")

        assert isinstance(error, SignalMappingError)
        assert isinstance(error, AMQADFError)
        assert isinstance(error, Exception)
        assert str(error) == "Interpolation error"

    @pytest.mark.unit
    def test_fusion_error_inheritance(self):
        """Test FusionError inherits from AMQADFError."""
        error = FusionError("Fusion error")

        assert isinstance(error, AMQADFError)
        assert isinstance(error, Exception)
        assert str(error) == "Fusion error"

    @pytest.mark.unit
    def test_query_error_inheritance(self):
        """Test QueryError inherits from AMQADFError."""
        error = QueryError("Query error")

        assert isinstance(error, AMQADFError)
        assert isinstance(error, Exception)
        assert str(error) == "Query error"

    @pytest.mark.unit
    def test_storage_error_inheritance(self):
        """Test StorageError inherits from AMQADFError."""
        error = StorageError("Storage error")

        assert isinstance(error, AMQADFError)
        assert isinstance(error, Exception)
        assert str(error) == "Storage error"

    @pytest.mark.unit
    def test_validation_error_inheritance(self):
        """Test ValidationError inherits from AMQADFError."""
        error = ValidationError("Validation error")

        assert isinstance(error, AMQADFError)
        assert isinstance(error, Exception)
        assert str(error) == "Validation error"

    @pytest.mark.unit
    def test_configuration_error_inheritance(self):
        """Test ConfigurationError inherits from AMQADFError."""
        error = ConfigurationError("Configuration error")

        assert isinstance(error, AMQADFError)
        assert isinstance(error, Exception)
        assert str(error) == "Configuration error"

    @pytest.mark.unit
    def test_coordinate_system_error_inheritance(self):
        """Test CoordinateSystemError inherits from AMQADFError."""
        error = CoordinateSystemError("Coordinate system error")

        assert isinstance(error, AMQADFError)
        assert isinstance(error, Exception)
        assert str(error) == "Coordinate system error"

    @pytest.mark.unit
    def test_quality_assessment_error_inheritance(self):
        """Test QualityAssessmentError inherits from AMQADFError."""
        error = QualityAssessmentError("Quality assessment error")

        assert isinstance(error, AMQADFError)
        assert isinstance(error, Exception)
        assert str(error) == "Quality assessment error"

    @pytest.mark.unit
    def test_exception_catching_base(self):
        """Test that catching AMQADFError catches all derived exceptions."""
        errors = [
            VoxelGridError("test"),
            SignalMappingError("test"),
            InterpolationError("test"),
            FusionError("test"),
            QueryError("test"),
            StorageError("test"),
            ValidationError("test"),
            ConfigurationError("test"),
            CoordinateSystemError("test"),
            QualityAssessmentError("test"),
        ]

        for error in errors:
            with pytest.raises(AMQADFError):
                raise error

    @pytest.mark.unit
    def test_exception_catching_specific(self):
        """Test that catching specific exception only catches that type."""
        # InterpolationError should be caught by SignalMappingError
        with pytest.raises(SignalMappingError):
            raise InterpolationError("test")

        # But not by FusionError
        with pytest.raises(InterpolationError):
            try:
                raise InterpolationError("test")
            except FusionError:
                pytest.fail("Should not catch InterpolationError")

    @pytest.mark.unit
    def test_exception_with_cause(self):
        """Test exceptions can have a cause."""
        cause = ValueError("Original error")
        error = VoxelGridError("Voxel grid error")
        error.__cause__ = cause

        assert error.__cause__ == cause

    @pytest.mark.unit
    def test_exception_message_formatting(self):
        """Test exception message formatting."""
        error_msg = "Error with value: {value}"
        error = AMQADFError(error_msg.format(value=42))

        assert "42" in str(error)
