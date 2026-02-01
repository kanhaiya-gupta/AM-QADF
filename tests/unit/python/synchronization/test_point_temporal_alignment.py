"""
Unit tests for point_temporal_alignment (thin C++ wrapper).

Tests re-export of PointTemporalAlignment, LayerAlignmentResult. Requires am_qadf_native.
"""

import pytest

pytest.importorskip("am_qadf_native", reason="point_temporal_alignment requires am_qadf_native")

from am_qadf.synchronization.point_temporal_alignment import (
    PointTemporalAlignment,
    LayerAlignmentResult,
    CPP_AVAILABLE,
)


class TestPointTemporalAlignment:
    """Test suite for point_temporal_alignment module."""

    @pytest.mark.unit
    def test_import(self):
        """PointTemporalAlignment and LayerAlignmentResult are importable."""
        assert PointTemporalAlignment is not None
        assert LayerAlignmentResult is not None
        assert CPP_AVAILABLE is True

    @pytest.mark.unit
    def test_creation(self):
        """PointTemporalAlignment() creates instance when C++ available."""
        if PointTemporalAlignment is None:
            pytest.skip("PointTemporalAlignment not available (e.g. EIGEN_AVAILABLE)")
        aligner = PointTemporalAlignment()
        assert aligner is not None
