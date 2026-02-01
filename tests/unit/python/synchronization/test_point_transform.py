"""
Unit tests for point_transform (thin C++ wrapper).

Tests re-export of PointTransformer. Requires am_qadf_native.
"""

import pytest

pytest.importorskip("am_qadf_native", reason="point_transform requires am_qadf_native")

from am_qadf.synchronization.point_transform import PointTransformer, CPP_AVAILABLE


class TestPointTransform:
    """Test suite for point_transform module."""

    @pytest.mark.unit
    def test_import(self):
        """PointTransformer is importable."""
        assert PointTransformer is not None
        assert CPP_AVAILABLE is True

    @pytest.mark.unit
    def test_creation(self):
        """PointTransformer() creates instance when C++ available."""
        transformer = PointTransformer()
        assert transformer is not None
