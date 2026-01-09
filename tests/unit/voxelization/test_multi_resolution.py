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
        assert len(grid.grids) == 3

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
        """Test getting nonexistent level."""
        grid = MultiResolutionGrid(
            bbox_min=(0.0, 0.0, 0.0),
            bbox_max=(10.0, 10.0, 10.0),
            base_resolution=1.0,
            num_levels=3,
        )

        level_grid = grid.get_level(10)

        assert level_grid is None

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

        # Should be added to all levels
        for level in range(grid.num_levels):
            level_grid = grid.get_level(level)
            assert len(level_grid.voxels) > 0

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

        # Should only be in level 1
        assert len(grid.get_level(1).voxels) > 0
        assert len(grid.get_level(0).voxels) == 0
        assert len(grid.get_level(2).voxels) == 0

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

        # All grids should be finalized
        for level in range(grid.num_levels):
            level_grid = grid.get_level(level)
            # After finalize, signals should be aggregated values
            voxel = level_grid.get_voxel(5, 5, 5)
            if voxel:
                assert isinstance(voxel.signals.get("power"), (int, float, np.floating))

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
    def test_multi_resolution_grid_get_statistics(self):
        """Test getting statistics for specific level."""
        grid = MultiResolutionGrid(
            bbox_min=(0.0, 0.0, 0.0),
            bbox_max=(10.0, 10.0, 10.0),
            base_resolution=1.0,
            num_levels=3,
        )

        grid.add_point(5.0, 5.0, 5.0, {"power": 200.0})
        grid.finalize()

        stats = grid.get_statistics(level=0)

        assert "dimensions" in stats
        assert "resolution_mm" in stats
        assert "filled_voxels" in stats

    @pytest.mark.unit
    def test_multi_resolution_grid_select_appropriate_level(self):
        """Test selecting appropriate level based on target resolution."""
        grid = MultiResolutionGrid(
            bbox_min=(0.0, 0.0, 0.0),
            bbox_max=(10.0, 10.0, 10.0),
            base_resolution=1.0,
            num_levels=3,
            level_ratio=2.0,
        )

        # Target resolution 1.0 should select level 2 (finest)
        level = grid.select_appropriate_level(1.0)
        assert level == 2

        # Target resolution 3.0 should select level 1 (closest to 2.0)
        level = grid.select_appropriate_level(3.0)
        assert level == 1

    @pytest.mark.unit
    def test_multi_resolution_grid_select_appropriate_level_prefer_coarse(self):
        """Test selecting level with prefer_coarse option."""
        grid = MultiResolutionGrid(
            bbox_min=(0.0, 0.0, 0.0),
            bbox_max=(10.0, 10.0, 10.0),
            base_resolution=1.0,
            num_levels=3,
            level_ratio=2.0,
        )

        # With prefer_coarse, should select coarser level if acceptable
        level = grid.select_appropriate_level(1.5, prefer_coarse=True)
        # Should prefer level 1 (2.0) over level 2 (1.0) if acceptable
        assert level in [1, 2]

    @pytest.mark.unit
    def test_multi_resolution_grid_get_level_for_view_distance(self):
        """Test selecting level based on view distance."""
        grid = MultiResolutionGrid(
            bbox_min=(0.0, 0.0, 0.0),
            bbox_max=(10.0, 10.0, 10.0),
            base_resolution=1.0,
            num_levels=3,
            level_ratio=2.0,
        )

        # Close view should use finer resolution
        level_close = grid.get_level_for_view_distance(10.0)

        # Far view should use coarser resolution
        level_far = grid.get_level_for_view_distance(1000.0)

        assert level_close >= level_far  # Closer = finer (higher level index)

    @pytest.mark.unit
    def test_multi_resolution_grid_downsample_from_finer(self):
        """Test downsampling from finer to coarser level."""
        grid = MultiResolutionGrid(
            bbox_min=(0.0, 0.0, 0.0),
            bbox_max=(10.0, 10.0, 10.0),
            base_resolution=1.0,
            num_levels=3,
            level_ratio=2.0,
        )

        # Should not raise error (implementation is placeholder)
        grid.downsample_from_finer(source_level=2, target_level=0)

    @pytest.mark.unit
    def test_multi_resolution_grid_downsample_invalid_levels(self):
        """Test downsampling with invalid level order."""
        grid = MultiResolutionGrid(
            bbox_min=(0.0, 0.0, 0.0),
            bbox_max=(10.0, 10.0, 10.0),
            base_resolution=1.0,
            num_levels=3,
        )

        with pytest.raises(ValueError, match="Source level must be finer"):
            grid.downsample_from_finer(source_level=0, target_level=2)


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
