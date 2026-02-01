"""
Unit tests for multi-resolution grids.

Tests for ResolutionLevel, MultiResolutionGrid, and ResolutionSelector.
"""

import pytest
import numpy as np
from am_qadf.voxelization.multi_resolution import (
    ResolutionLevel,
    MultiResolutionGrid,
    ResolutionSelector,
)


class TestResolutionLevel:
    """Test suite for ResolutionLevel enum."""

    @pytest.mark.unit
    def test_resolution_level_values(self):
        """Test ResolutionLevel enum values."""
        assert ResolutionLevel.COARSE.value == "coarse"
        assert ResolutionLevel.MEDIUM.value == "medium"
        assert ResolutionLevel.FINE.value == "fine"
        assert ResolutionLevel.ULTRA_FINE.value == "ultra_fine"

    @pytest.mark.unit
    def test_resolution_level_enumeration(self):
        """Test that ResolutionLevel can be enumerated."""
        levels = list(ResolutionLevel)
        assert len(levels) == 4
        assert ResolutionLevel.COARSE in levels


class TestMultiResolutionGrid:
    """Test suite for MultiResolutionGrid class."""

    @pytest.mark.unit
    def test_multi_resolution_grid_creation(self):
        """Test creating MultiResolutionGrid."""
        grid = MultiResolutionGrid(
            bbox_min=(0.0, 0.0, 0.0),
            bbox_max=(10.0, 10.0, 10.0),
            base_resolution=1.0,
            num_levels=3,
            level_ratio=2.0,
        )

        assert grid.bbox_min == (0.0, 0.0, 0.0)
        assert grid.bbox_max == (10.0, 10.0, 10.0)
        assert grid.base_resolution == 1.0
        assert grid.num_levels == 3
        assert grid.level_ratio == 2.0
        # grids populated on-demand via get_level()
        assert len(grid.resolutions) == 3

    @pytest.mark.unit
    def test_multi_resolution_grid_resolution_calculation(self):
        """Test resolution calculation for different levels."""
        grid = MultiResolutionGrid(
            bbox_min=(0.0, 0.0, 0.0),
            bbox_max=(10.0, 10.0, 10.0),
            base_resolution=1.0,
            num_levels=3,
            level_ratio=2.0,
        )

        # Level 0 (coarsest): 1.0 * 2^(3-1-0) = 1.0 * 4 = 4.0
        # Level 1 (medium): 1.0 * 2^(3-1-1) = 1.0 * 2 = 2.0
        # Level 2 (finest): 1.0 * 2^(3-1-2) = 1.0 * 1 = 1.0
        assert grid.get_resolution(0) == 4.0
        assert grid.get_resolution(1) == 2.0
        assert grid.get_resolution(2) == 1.0

    @pytest.mark.unit
    def test_multi_resolution_grid_get_level(self):
        """Test getting grid for specific level."""
        grid = MultiResolutionGrid(
            bbox_min=(0.0, 0.0, 0.0),
            bbox_max=(10.0, 10.0, 10.0),
            base_resolution=1.0,
            num_levels=3,
        )

        level_grid = grid.get_level(0)

        assert level_grid is not None
        assert level_grid.resolution == grid.get_resolution(0)

    @pytest.mark.unit
    def test_multi_resolution_grid_get_level_nonexistent(self):
        """Test getting nonexistent level raises ValueError."""
        grid = MultiResolutionGrid(
            bbox_min=(0.0, 0.0, 0.0),
            bbox_max=(10.0, 10.0, 10.0),
            base_resolution=1.0,
            num_levels=3,
        )

        with pytest.raises(ValueError, match="Level 10 not found"):
            grid.get_level(10)

    @pytest.mark.unit
    def test_multi_resolution_grid_add_point_all_levels(self):
        """Test adding point to all levels."""
        grid = MultiResolutionGrid(
            bbox_min=(0.0, 0.0, 0.0),
            bbox_max=(10.0, 10.0, 10.0),
            base_resolution=1.0,
            num_levels=3,
        )

        grid.add_point(5.0, 5.0, 5.0, {"power": 200.0})

        # Should be added to all levels; level grids exist and have resolution
        for level in range(grid.num_levels):
            level_grid = grid.get_level(level)
            assert level_grid is not None
            assert level_grid.resolution == grid.get_resolution(level)

    @pytest.mark.unit
    def test_multi_resolution_grid_add_point_specific_level(self):
        """Test adding point to specific level."""
        grid = MultiResolutionGrid(
            bbox_min=(0.0, 0.0, 0.0),
            bbox_max=(10.0, 10.0, 10.0),
            base_resolution=1.0,
            num_levels=3,
        )

        grid.add_point(5.0, 5.0, 5.0, {"power": 200.0}, level=1)

        # Level grids exist
        assert grid.get_level(0) is not None
        assert grid.get_level(1) is not None
        assert grid.get_level(2) is not None

    @pytest.mark.unit
    def test_multi_resolution_grid_finalize(self):
        """Test finalizing all grids."""
        grid = MultiResolutionGrid(
            bbox_min=(0.0, 0.0, 0.0),
            bbox_max=(10.0, 10.0, 10.0),
            base_resolution=1.0,
            num_levels=3,
        )

        grid.add_point(5.0, 5.0, 5.0, {"power": 200.0})
        grid.add_point(5.1, 5.1, 5.1, {"power": 250.0})
        grid.finalize()

        # All level grids accessible
        for level in range(grid.num_levels):
            level_grid = grid.get_level(level)
            assert level_grid is not None

    @pytest.mark.unit
    def test_multi_resolution_grid_get_signal_array(self):
        """Test getting signal array for specific level."""
        grid = MultiResolutionGrid(
            bbox_min=(0.0, 0.0, 0.0),
            bbox_max=(10.0, 10.0, 10.0),
            base_resolution=1.0,
            num_levels=3,
        )

        grid.add_point(5.0, 5.0, 5.0, {"power": 200.0})
        grid.finalize()

        signal_array = grid.get_signal_array("power", level=0)

        assert isinstance(signal_array, np.ndarray)
        assert signal_array.ndim == 3

    @pytest.mark.unit
    def test_multi_resolution_grid_get_signal_array_invalid_level(self):
        """Test getting signal array for invalid level."""
        grid = MultiResolutionGrid(
            bbox_min=(0.0, 0.0, 0.0),
            bbox_max=(10.0, 10.0, 10.0),
            base_resolution=1.0,
            num_levels=3,
        )

        with pytest.raises(ValueError, match="Level 10 not found"):
            grid.get_signal_array("power", level=10)

    @pytest.mark.unit
    def test_multi_resolution_grid_restrict(self):
        """Test restrict (downsample) from fine to coarse level."""
        grid = MultiResolutionGrid(
            bbox_min=(0.0, 0.0, 0.0),
            bbox_max=(10.0, 10.0, 10.0),
            base_resolution=1.0,
            num_levels=3,
            level_ratio=2.0,
        )
        grid.add_point(5.0, 5.0, 5.0, {"power": 200.0})
        grid.restrict(from_level=2, to_level=0)
        # Invalidate cache for to_level
        assert grid.get_level(0) is not None

    @pytest.mark.unit
    def test_multi_resolution_grid_prolongate(self):
        """Test prolongate (interpolate) from coarse to fine level."""
        grid = MultiResolutionGrid(
            bbox_min=(0.0, 0.0, 0.0),
            bbox_max=(10.0, 10.0, 10.0),
            base_resolution=1.0,
            num_levels=3,
            level_ratio=2.0,
        )
        grid.add_point(5.0, 5.0, 5.0, {"power": 200.0})
        grid.prolongate(from_level=0, to_level=2)
        assert grid.get_level(2) is not None


class TestResolutionSelector:
    """Test suite for ResolutionSelector class."""

    @pytest.mark.unit
    def test_resolution_selector_creation(self):
        """Test creating ResolutionSelector."""
        selector = ResolutionSelector(performance_mode="balanced")

        assert selector.performance_mode == "balanced"

    @pytest.mark.unit
    def test_resolution_selector_select_for_performance_fast(self):
        """Test selecting level for fast performance mode."""
        grid = MultiResolutionGrid(
            bbox_min=(0.0, 0.0, 0.0),
            bbox_max=(10.0, 10.0, 10.0),
            base_resolution=1.0,
            num_levels=3,
        )

        selector = ResolutionSelector(performance_mode="fast")
        level = selector.select_for_performance(grid, num_points=1000)

        assert 0 <= level < grid.num_levels

    @pytest.mark.unit
    def test_resolution_selector_select_for_performance_quality(self):
        """Test selecting level for quality performance mode."""
        grid = MultiResolutionGrid(
            bbox_min=(0.0, 0.0, 0.0),
            bbox_max=(10.0, 10.0, 10.0),
            base_resolution=1.0,
            num_levels=3,
        )

        selector = ResolutionSelector(performance_mode="quality")
        level = selector.select_for_performance(grid, num_points=1000)

        assert 0 <= level < grid.num_levels

    @pytest.mark.unit
    def test_resolution_selector_select_for_performance_with_memory(self):
        """Test selecting level with memory constraint."""
        grid = MultiResolutionGrid(
            bbox_min=(0.0, 0.0, 0.0),
            bbox_max=(10.0, 10.0, 10.0),
            base_resolution=1.0,
            num_levels=3,
        )

        selector = ResolutionSelector(performance_mode="balanced")
        level = selector.select_for_performance(grid, num_points=1000, available_memory=0.001)  # Very small memory (1 MB)

        assert 0 <= level < grid.num_levels

    @pytest.mark.unit
    def test_resolution_selector_select_for_data_density_high(self):
        """Test selecting level for high data density."""
        grid = MultiResolutionGrid(
            bbox_min=(0.0, 0.0, 0.0),
            bbox_max=(10.0, 10.0, 10.0),
            base_resolution=1.0,
            num_levels=3,
        )

        selector = ResolutionSelector()
        level = selector.select_for_data_density(grid, data_density=2.0)  # High density

        # Should prefer finer resolution
        assert level >= grid.num_levels // 2

    @pytest.mark.unit
    def test_resolution_selector_select_for_data_density_low(self):
        """Test selecting level for low data density."""
        grid = MultiResolutionGrid(
            bbox_min=(0.0, 0.0, 0.0),
            bbox_max=(10.0, 10.0, 10.0),
            base_resolution=1.0,
            num_levels=3,
        )

        selector = ResolutionSelector()
        level = selector.select_for_data_density(grid, data_density=0.05)  # Low density

        # Should prefer coarser resolution
        assert level <= grid.num_levels // 2

    @pytest.mark.unit
    def test_resolution_selector_select_for_view(self):
        """Test selecting level based on view parameters."""
        grid = MultiResolutionGrid(
            bbox_min=(0.0, 0.0, 0.0),
            bbox_max=(10.0, 10.0, 10.0),
            base_resolution=1.0,
            num_levels=3,
        )

        selector = ResolutionSelector()

        # Close view
        view_params_close = {"distance": 10.0, "zoom": 1.0, "region_size": 10.0}
        level_close = selector.select_for_view(grid, view_params_close)

        # Far view
        view_params_far = {"distance": 1000.0, "zoom": 1.0, "region_size": 100.0}
        level_far = selector.select_for_view(grid, view_params_far)

        assert 0 <= level_close < grid.num_levels
        assert 0 <= level_far < grid.num_levels
