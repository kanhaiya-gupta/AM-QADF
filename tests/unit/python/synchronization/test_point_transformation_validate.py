"""
Unit tests for point_transformation_validate (thin C++ wrapper).

Tests re-export of TransformationValidator and related types. Requires am_qadf_native.
"""

import pytest

pytest.importorskip("am_qadf_native", reason="point_transformation_validate requires am_qadf_native")

from am_qadf.synchronization.point_transformation_validate import (
    TransformationValidator,
    ValidationResult,
    BboxCorrespondenceValidation,
    CPP_AVAILABLE,
)


class TestPointTransformationValidate:
    """Test suite for point_transformation_validate module."""

    @pytest.mark.unit
    def test_import(self):
        """TransformationValidator and related types are importable."""
        assert TransformationValidator is not None
        assert CPP_AVAILABLE is True

    @pytest.mark.unit
    def test_creation(self):
        """TransformationValidator() creates instance when C++ available."""
        validator = TransformationValidator()
        assert validator is not None
