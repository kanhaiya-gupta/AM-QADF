"""
Unit tests for point_bounds (thin C++ wrapper).

Tests re-export of UnifiedBoundsComputer and BoundingBox. Requires am_qadf_native.
"""

import pytest

pytest.importorskip("am_qadf_native", reason="point_bounds requires am_qadf_native")

from am_qadf.synchronization.point_bounds import (
    UnifiedBoundsComputer,
    BoundingBox,
    CPP_AVAILABLE,
)


class TestPointBounds:
    """Test suite for point_bounds module."""

    @pytest.mark.unit
    def test_import(self):
        """UnifiedBoundsComputer and BoundingBox are importable."""
        assert UnifiedBoundsComputer is not None
        assert BoundingBox is not None
        assert CPP_AVAILABLE is True

    @pytest.mark.unit
    def test_bounding_box_creation(self):
        """BoundingBox can be created with min/max."""
        bbox = BoundingBox(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        assert bbox is not None
        assert bbox.min_x == 0.0
        assert bbox.max_x == 1.0

    @pytest.mark.unit
    def test_unified_bounds_computer_creation(self):
        """UnifiedBoundsComputer() creates instance when C++ available."""
        computer = UnifiedBoundsComputer()
        assert computer is not None
