"""
Unit tests for grid_spatial_alignment (thin C++ wrapper).

Tests GridSpatialAlignment.align. Requires am_qadf_native and VoxelGrid.
"""

import pytest
import numpy as np

pytest.importorskip("am_qadf_native", reason="GridSpatialAlignment requires am_qadf_native")

from am_qadf.synchronization.grid_spatial_alignment import GridSpatialAlignment, CPP_AVAILABLE
from am_qadf.voxelization.uniform_resolution import VoxelGrid


@pytest.fixture
def aligner():
    """GridSpatialAlignment instance."""
    return GridSpatialAlignment()


@pytest.fixture
def source_and_target():
    """Source and target VoxelGrids with signal 'sig'."""
    bbox_min = (0.0, 0.0, 0.0)
    bbox_max = (2.0, 2.0, 2.0)
    res = 1.0
    source = VoxelGrid(bbox_min=bbox_min, bbox_max=bbox_max, resolution=res)
    target = VoxelGrid(bbox_min=bbox_min, bbox_max=bbox_max, resolution=res)
    arr = np.ones((2, 2, 2), dtype=np.float32) * 10.0
    source._get_or_create_grid("sig").populate_from_array(arr)
    source.available_signals.add("sig")
    target._get_or_create_grid("sig").populate_from_array(arr * 0.5)
    target.available_signals.add("sig")
    return source, target


class TestGridSpatialAlignment:
    """Test suite for GridSpatialAlignment."""

    @pytest.mark.unit
    def test_creation(self):
        """GridSpatialAlignment() creates instance when C++ available."""
        a = GridSpatialAlignment()
        assert a is not None
        assert a._aligner is not None

    @pytest.mark.unit
    def test_align_returns_voxel_grid(self, aligner, source_and_target):
        """align returns a single VoxelGrid."""
        source, target = source_and_target
        result = aligner.align(source, target, signal_name="sig", method="trilinear")
        assert isinstance(result, VoxelGrid)
        assert "sig" in result.available_signals

    @pytest.mark.unit
    def test_align_with_method(self, aligner, source_and_target):
        """align accepts method 'nearest' and 'trilinear'."""
        source, target = source_and_target
        r1 = aligner.align(source, target, signal_name="sig", method="nearest")
        r2 = aligner.align(source, target, signal_name="sig", method="trilinear")
        assert isinstance(r1, VoxelGrid)
        assert isinstance(r2, VoxelGrid)

    @pytest.mark.unit
    def test_align_missing_signal_raises(self, aligner):
        """align raises when target does not have signal."""
        bbox_min = (0.0, 0.0, 0.0)
        bbox_max = (2.0, 2.0, 2.0)
        source = VoxelGrid(bbox_min=bbox_min, bbox_max=bbox_max, resolution=1.0)
        source._get_or_create_grid("sig").populate_from_array(np.ones((2, 2, 2), dtype=np.float32))
        source.available_signals.add("sig")
        target = VoxelGrid(bbox_min=bbox_min, bbox_max=bbox_max, resolution=1.0)
        with pytest.raises(ValueError, match="Target grid does not have signal"):
            aligner.align(source, target, signal_name="sig")
