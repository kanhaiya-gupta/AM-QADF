"""
Unit tests for adaptive resolution grids.

Tests for SpatialResolutionMap, TemporalResolutionMap, and AdaptiveResolutionGrid.
"""

import pytest
import numpy as np
from am_qadf.voxelization.adaptive_resolution import (
    SpatialResolutionMap,
    TemporalResolutionMap,
    AdaptiveResolutionGrid,
)


class TestSpatialResolutionMap:
    """Test suite for SpatialResolutionMap."""

    @pytest.mark.unit
    def test_spatial_resolution_map_creation(self):
        """Test creating SpatialResolutionMap."""
        spatial_map = SpatialResolutionMap()

        assert len(spatial_map.regions) == 0
        assert spatial_map.default_resolution == 1.0

    @pytest.mark.unit
    def test_spatial_resolution_map_with_regions(self):
        """Test SpatialResolutionMap with regions."""
        regions = [
            ((0.0, 0.0, 0.0), (10.0, 10.0, 10.0), 0.5),
            ((10.0, 10.0, 10.0), (20.0, 20.0, 20.0), 1.0),
        ]
        spatial_map = SpatialResolutionMap(regions=regions, default_resolution=2.0)

        assert len(spatial_map.regions) == 2
        assert spatial_map.default_resolution == 2.0


class TestTemporalResolutionMap:
    """Test suite for TemporalResolutionMap."""

    @pytest.mark.unit
    def test_temporal_resolution_map_creation(self):
        """Test creating TemporalResolutionMap."""
        temporal_map = TemporalResolutionMap()

        assert len(temporal_map.time_ranges) == 0
        assert len(temporal_map.layer_ranges) == 0
        assert temporal_map.default_resolution == 1.0

    @pytest.mark.unit
    def test_temporal_resolution_map_with_ranges(self):
        """Test TemporalResolutionMap with time and layer ranges."""
        time_ranges = [(0.0, 100.0, 0.5), (100.0, 200.0, 1.0)]
        layer_ranges = [(0, 10, 0.5), (10, 20, 1.0)]
        temporal_map = TemporalResolutionMap(time_ranges=time_ranges, layer_ranges=layer_ranges, default_resolution=2.0)

        assert len(temporal_map.time_ranges) == 2
        assert len(temporal_map.layer_ranges) == 2
        assert temporal_map.default_resolution == 2.0


class TestAdaptiveResolutionGrid:
    """Test suite for AdaptiveResolutionGrid."""

    @pytest.mark.unit
    def test_adaptive_grid_creation(self):
        """Test creating AdaptiveResolutionGrid."""
        grid = AdaptiveResolutionGrid(bbox_min=(0.0, 0.0, 0.0), bbox_max=(10.0, 10.0, 10.0), base_resolution=1.0)

        assert np.array_equal(grid.bbox_min, [0.0, 0.0, 0.0])
        assert np.array_equal(grid.bbox_max, [10.0, 10.0, 10.0])
        assert grid.base_resolution == 1.0
        assert len(grid.available_signals) == 0
        assert not grid._finalized

    @pytest.mark.unit
    def test_adaptive_grid_add_point(self):
        """Test adding point to adaptive grid."""
        grid = AdaptiveResolutionGrid(bbox_min=(0.0, 0.0, 0.0), bbox_max=(10.0, 10.0, 10.0), base_resolution=1.0)

        grid.add_point(5.0, 5.0, 5.0, {"power": 200.0})

        assert "power" in grid.available_signals

    @pytest.mark.unit
    def test_adaptive_grid_add_point_with_temporal(self):
        """Test adding point with temporal information."""
        grid = AdaptiveResolutionGrid(bbox_min=(0.0, 0.0, 0.0), bbox_max=(10.0, 10.0, 10.0), base_resolution=1.0)

        grid.add_point(5.0, 5.0, 5.0, {"power": 200.0}, timestamp=100.0, layer_index=5)

        assert "power" in grid.available_signals

    @pytest.mark.unit
    def test_adaptive_grid_add_point_after_finalize(self):
        """Test that adding point after finalize raises error."""
        grid = AdaptiveResolutionGrid(bbox_min=(0.0, 0.0, 0.0), bbox_max=(10.0, 10.0, 10.0), base_resolution=1.0)

        grid.add_point(5.0, 5.0, 5.0, {"power": 200.0})
        grid.finalize()

        with pytest.raises(ValueError, match="Cannot add points after finalization"):
            grid.add_point(6.0, 6.0, 6.0, {"power": 250.0})

    @pytest.mark.unit
    def test_adaptive_grid_get_resolution_for_point_default(self):
        """Test getting resolution for point with default resolution."""
        grid = AdaptiveResolutionGrid(bbox_min=(0.0, 0.0, 0.0), bbox_max=(10.0, 10.0, 10.0), base_resolution=1.0)

        resolution = grid.get_resolution_for_point(5.0, 5.0, 5.0)

        assert resolution == 1.0

    @pytest.mark.unit
    def test_adaptive_grid_get_resolution_for_point_spatial_region(self):
        """Test getting resolution for point in spatial region."""
        spatial_map = SpatialResolutionMap(regions=[((0.0, 0.0, 0.0), (5.0, 5.0, 5.0), 0.5)], default_resolution=1.0)
        grid = AdaptiveResolutionGrid(
            bbox_min=(0.0, 0.0, 0.0),
            bbox_max=(10.0, 10.0, 10.0),
            base_resolution=1.0,
            spatial_resolution_map=spatial_map,
        )
        # Add a point so C++ grid is created with spatial regions
        grid.add_point(2.0, 2.0, 2.0, {"power": 100.0})

        # Point in region
        resolution = grid.get_resolution_for_point(2.0, 2.0, 2.0)
        assert resolution == 0.5

        # Point outside region
        resolution = grid.get_resolution_for_point(7.0, 7.0, 7.0)
        assert resolution == 1.0

    @pytest.mark.unit
    def test_adaptive_grid_get_resolution_for_point_temporal(self):
        """Test getting resolution for point with temporal information."""
        temporal_map = TemporalResolutionMap(
            time_ranges=[(0.0, 100.0, 0.5)],
            layer_ranges=[(0, 10, 0.3)],
            default_resolution=1.0,
        )
        grid = AdaptiveResolutionGrid(
            bbox_min=(0.0, 0.0, 0.0),
            bbox_max=(10.0, 10.0, 10.0),
            base_resolution=1.0,
            temporal_resolution_map=temporal_map,
        )
        # Add a point so C++ grid is created with temporal ranges
        grid.add_point(5.0, 5.0, 5.0, {"power": 100.0}, timestamp=50.0, layer_index=5)

        # Point with timestamp in range
        resolution = grid.get_resolution_for_point(5.0, 5.0, 5.0, timestamp=50.0)
        assert resolution == 0.5

        # Point with layer in range (use approx for float precision)
        resolution = grid.get_resolution_for_point(5.0, 5.0, 5.0, layer_index=5)
        assert resolution == pytest.approx(0.3)

    @pytest.mark.unit
    def test_adaptive_grid_finalize(self):
        """Test finalizing adaptive grid."""
        grid = AdaptiveResolutionGrid(bbox_min=(0.0, 0.0, 0.0), bbox_max=(10.0, 10.0, 10.0), base_resolution=1.0)

        grid.add_point(5.0, 5.0, 5.0, {"power": 200.0})
        grid.add_point(6.0, 6.0, 6.0, {"power": 250.0})

        grid.finalize()

        assert grid._finalized

    @pytest.mark.unit
    def test_adaptive_grid_get_signal_array(self):
        """Test getting signal array from grid."""
        grid = AdaptiveResolutionGrid(bbox_min=(0.0, 0.0, 0.0), bbox_max=(10.0, 10.0, 10.0), base_resolution=1.0)

        grid.add_point(5.0, 5.0, 5.0, {"power": 200.0})
        grid.finalize()

        signal_array = grid.get_signal_array("power")

        assert isinstance(signal_array, np.ndarray)
        assert signal_array.ndim == 3

    @pytest.mark.unit
    def test_adaptive_grid_get_signal_array_before_finalize(self):
        """Test getting signal array before finalize returns array (C++ wrapper allows it)."""
        grid = AdaptiveResolutionGrid(bbox_min=(0.0, 0.0, 0.0), bbox_max=(10.0, 10.0, 10.0), base_resolution=1.0)

        grid.add_point(5.0, 5.0, 5.0, {"power": 200.0})

        signal_array = grid.get_signal_array("power")
        assert isinstance(signal_array, np.ndarray)
        assert signal_array.ndim == 3

    @pytest.mark.unit
    def test_adaptive_grid_add_spatial_region(self):
        """Test adding spatial region."""
        grid = AdaptiveResolutionGrid(bbox_min=(0.0, 0.0, 0.0), bbox_max=(10.0, 10.0, 10.0), base_resolution=1.0)

        grid.add_spatial_region(bbox_min=(0.0, 0.0, 0.0), bbox_max=(5.0, 5.0, 5.0), resolution=0.5)

        assert len(grid.spatial_map.regions) == 1

    @pytest.mark.unit
    def test_adaptive_grid_add_temporal_range(self):
        """Test adding temporal range."""
        grid = AdaptiveResolutionGrid(bbox_min=(0.0, 0.0, 0.0), bbox_max=(10.0, 10.0, 10.0), base_resolution=1.0)

        grid.add_temporal_range(0.0, 100.0, 0.5)

        assert len(grid.temporal_map.time_ranges) == 1

    @pytest.mark.unit
    def test_adaptive_grid_add_layer_range(self):
        """Test adding layer range."""
        grid = AdaptiveResolutionGrid(bbox_min=(0.0, 0.0, 0.0), bbox_max=(10.0, 10.0, 10.0), base_resolution=1.0)

        grid.add_layer_range(0, 10, 0.5)

        assert len(grid.temporal_map.layer_ranges) == 1

