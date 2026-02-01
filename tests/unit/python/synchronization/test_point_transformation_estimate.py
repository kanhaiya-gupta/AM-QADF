"""
Unit tests for point_transformation_estimate (thin C++ wrapper).

Tests re-export of TransformationComputer and related types. Requires am_qadf_native.
"""

import pytest

pytest.importorskip("am_qadf_native", reason="point_transformation_estimate requires am_qadf_native")

from am_qadf.synchronization.point_transformation_estimate import (
    TransformationComputer,
    RANSACResult,
    TransformationQuality,
    BboxFitCandidate,
    ScaleTranslationRotation,
    CPP_AVAILABLE,
)


class TestPointTransformationEstimate:
    """Test suite for point_transformation_estimate module."""

    @pytest.mark.unit
    def test_import(self):
        """TransformationComputer and related types are importable."""
        assert TransformationComputer is not None
        assert CPP_AVAILABLE is True

    @pytest.mark.unit
    def test_creation(self):
        """TransformationComputer() creates instance when C++ available."""
        computer = TransformationComputer()
        assert computer is not None
