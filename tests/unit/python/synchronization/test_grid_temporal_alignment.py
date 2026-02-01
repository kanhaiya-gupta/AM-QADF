"""
Unit tests for grid_temporal_alignment (thin C++ wrapper).

Tests GridTemporalAlignment.synchronize_temporal. Requires am_qadf_native and VoxelGrid.
"""

import pytest
import numpy as np

pytest.importorskip("am_qadf_native", reason="GridTemporalAlignment requires am_qadf_native")

from am_qadf.synchronization.grid_temporal_alignment import GridTemporalAlignment, CPP_AVAILABLE
from am_qadf.voxelization.uniform_resolution import VoxelGrid


@pytest.fixture
def aligner():
    """GridTemporalAlignment instance."""
    return GridTemporalAlignment()


@pytest.fixture
def two_grids_same_signal():
    """Two VoxelGrids with same bbox/resolution and signal 'sig'."""
    bbox_min = (0.0, 0.0, 0.0)
    bbox_max = (2.0, 2.0, 2.0)
    res = 1.0
    g1 = VoxelGrid(bbox_min=bbox_min, bbox_max=bbox_max, resolution=res)
    g2 = VoxelGrid(bbox_min=bbox_min, bbox_max=bbox_max, resolution=res)
    arr = np.ones((2, 2, 2), dtype=np.float32) * 5.0
    g1._get_or_create_grid("sig").populate_from_array(arr)
    g1.available_signals.add("sig")
    g2._get_or_create_grid("sig").populate_from_array(arr * 2)
    g2.available_signals.add("sig")
    return g1, g2


class TestGridTemporalAlignment:
    """Test suite for GridTemporalAlignment."""

    @pytest.mark.unit
    def test_creation(self):
        """GridTemporalAlignment() creates instance when C++ available."""
        a = GridTemporalAlignment()
        assert a is not None
        assert a._synchronizer is not None

    @pytest.mark.unit
    def test_synchronize_temporal_returns_list(self, aligner, two_grids_same_signal):
        """synchronize_temporal returns list of VoxelGrids."""
        g1, g2 = two_grids_same_signal
        result = aligner.synchronize_temporal(
            [g1, g2], signal_name="sig", temporal_window=0.2, layer_tolerance=1
        )
        assert isinstance(result, list)
        assert len(result) == 2
        for vg in result:
            assert isinstance(vg, VoxelGrid)
            assert "sig" in vg.available_signals

    @pytest.mark.unit
    def test_synchronize_temporal_single_grid(self, aligner, two_grids_same_signal):
        """synchronize_temporal with one grid returns one grid."""
        g1, _ = two_grids_same_signal
        result = aligner.synchronize_temporal([g1], signal_name="sig")
        assert len(result) == 1
        assert isinstance(result[0], VoxelGrid)

    @pytest.mark.unit
    def test_synchronize_temporal_missing_signal_raises(self, aligner):
        """synchronize_temporal raises when grid missing signal."""
        bbox_min = (0.0, 0.0, 0.0)
        bbox_max = (2.0, 2.0, 2.0)
        g = VoxelGrid(bbox_min=bbox_min, bbox_max=bbox_max, resolution=1.0)
        with pytest.raises(ValueError, match="do not have the chosen signal"):
            aligner.synchronize_temporal([g], signal_name="sig")
