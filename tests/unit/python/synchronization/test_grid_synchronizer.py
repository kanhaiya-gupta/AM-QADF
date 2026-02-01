"""
Unit tests for SynchronizationClient (thin C++ wrapper).

SynchronizationClient requires am_qadf_native; tests skip when C++ not built.
"""

import pytest

pytest.importorskip("am_qadf_native", reason="SynchronizationClient requires am_qadf_native")

from am_qadf.synchronization.grid_synchronizer import SynchronizationClient, CPP_AVAILABLE


class TestSynchronizationClient:
    """Test suite for SynchronizationClient."""

    @pytest.mark.unit
    def test_client_creation(self):
        """SynchronizationClient() creates instance when C++ available."""
        client = SynchronizationClient()
        assert client is not None
        assert client._synchronizer is not None

    @pytest.mark.unit
    def test_align_spatial_requires_common_signal(self):
        """align_spatial raises when no common signal between sources and target."""
        pytest.importorskip("am_qadf_native")
        from am_qadf.voxelization.uniform_resolution import VoxelGrid

        client = SynchronizationClient()
        bbox = (0.0, 0.0, 0.0), (2.0, 2.0, 2.0)
        res = 1.0
        g1 = VoxelGrid(bbox_min=bbox[0], bbox_max=bbox[1], resolution=res)
        g2 = VoxelGrid(bbox_min=bbox[0], bbox_max=bbox[1], resolution=res)
        target = VoxelGrid(bbox_min=bbox[0], bbox_max=bbox[1], resolution=res)
        # No signals in any grid -> no common signal
        with pytest.raises(ValueError, match="No common signal|specify signal_name"):
            client.align_spatial([g1, g2], target)

    @pytest.mark.unit
    def test_align_temporal_requires_common_signal(self):
        """align_temporal raises when grids have no common signal."""
        pytest.importorskip("am_qadf_native")
        from am_qadf.voxelization.uniform_resolution import VoxelGrid

        client = SynchronizationClient()
        bbox = (0.0, 0.0, 0.0), (2.0, 2.0, 2.0)
        res = 1.0
        g1 = VoxelGrid(bbox_min=bbox[0], bbox_max=bbox[1], resolution=res)
        g2 = VoxelGrid(bbox_min=bbox[0], bbox_max=bbox[1], resolution=res)
        with pytest.raises(ValueError, match="No common signal|specify signal_name"):
            client.align_temporal([g1, g2], timestamps=[[], []])

    @pytest.mark.unit
    def test_synchronize_requires_common_signal(self):
        """synchronize raises when no common signal across grids and reference."""
        pytest.importorskip("am_qadf_native")
        from am_qadf.voxelization.uniform_resolution import VoxelGrid

        client = SynchronizationClient()
        bbox = (0.0, 0.0, 0.0), (2.0, 2.0, 2.0)
        res = 1.0
        g1 = VoxelGrid(bbox_min=bbox[0], bbox_max=bbox[1], resolution=res)
        ref = VoxelGrid(bbox_min=bbox[0], bbox_max=bbox[1], resolution=res)
        with pytest.raises(ValueError, match="No common signal|specify signal_name"):
            client.synchronize([g1], timestamps=[[]], reference_grid=ref)


class TestSynchronizationClientNoCpp:
    """When C++ is not available, SynchronizationClient() raises."""

    @pytest.mark.unit
    def test_client_raises_without_cpp(self):
        """SynchronizationClient() raises ImportError when C++ not built (mock so test never skips)."""
        import unittest.mock as mock
        with mock.patch("am_qadf.synchronization.grid_synchronizer.CPP_AVAILABLE", False):
            with pytest.raises(ImportError, match="C\\+\\+ bindings not available|am_qadf_native"):
                SynchronizationClient()
