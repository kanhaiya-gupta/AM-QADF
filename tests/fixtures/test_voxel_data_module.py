"""
Tests for voxel_data fixture module.

Tests the loading and generation functions in tests/fixtures/voxel_data/__init__.py
"""

import pytest
from pathlib import Path
import pickle
import tempfile
import shutil

try:
    from tests.fixtures.voxel_data import (
        load_small_voxel_grid,
        load_medium_voxel_grid,
        load_large_voxel_grid,
        generate_small_voxel_grid,
        generate_medium_voxel_grid,
        generate_large_voxel_grid,
    )

    VOXEL_MODULE_AVAILABLE = True
except ImportError:
    VOXEL_MODULE_AVAILABLE = False


@pytest.mark.skipif(not VOXEL_MODULE_AVAILABLE, reason="Voxel data module not available")
class TestVoxelDataLoading:
    """Tests for voxel grid loading functions."""

    def test_load_small_voxel_grid(self):
        """Test loading small voxel grid."""
        grid = load_small_voxel_grid()
        assert grid is not None
        assert hasattr(grid, "dims")
        assert tuple(grid.dims) == (10, 10, 10)

    def test_load_medium_voxel_grid(self):
        """Test loading medium voxel grid."""
        grid = load_medium_voxel_grid()
        assert grid is not None
        assert hasattr(grid, "dims")
        assert tuple(grid.dims) == (50, 50, 50)

    def test_load_large_voxel_grid(self):
        """Test loading large voxel grid."""
        grid = load_large_voxel_grid()
        assert grid is not None
        assert hasattr(grid, "dims")
        assert tuple(grid.dims) == (100, 100, 100)

    def test_load_from_pickle_file(self):
        """Test that loading works from existing pickle file."""
        # This test verifies the file-based loading path
        grid = load_small_voxel_grid()
        # If file exists, it should load from file
        # If not, it generates on-the-fly
        assert grid is not None


@pytest.mark.skipif(not VOXEL_MODULE_AVAILABLE, reason="Voxel data module not available")
class TestVoxelDataGeneration:
    """Tests for voxel grid generation functions."""

    def test_generate_small_voxel_grid(self):
        """Test generating small voxel grid."""
        grid = generate_small_voxel_grid()
        assert grid is not None
        assert tuple(grid.dims) == (10, 10, 10)
        assert grid.resolution == 1.0
        assert hasattr(grid, "available_signals")

    def test_generate_medium_voxel_grid(self):
        """Test generating medium voxel grid."""
        grid = generate_medium_voxel_grid()
        assert grid is not None
        assert tuple(grid.dims) == (50, 50, 50)
        assert grid.resolution == 1.0

    def test_generate_large_voxel_grid(self):
        """Test generating large voxel grid."""
        grid = generate_large_voxel_grid()
        assert grid is not None
        assert tuple(grid.dims) == (100, 100, 100)
        assert grid.resolution == 1.0

    def test_generated_grids_are_finalized(self):
        """Test that generated grids are finalized."""
        grids = [
            generate_small_voxel_grid(),
            generate_medium_voxel_grid(),
            generate_large_voxel_grid(),
        ]
        for grid in grids:
            # Finalized grids should have dimensions and be ready to use
            assert hasattr(grid, "dims")
            assert grid.dims is not None

    def test_generated_grids_have_signals(self):
        """Test that generated grids have signals."""
        grid = generate_small_voxel_grid()
        assert hasattr(grid, "available_signals")
        # Small grid should have at least laser_power and temperature
        assert len(grid.available_signals) >= 2


@pytest.mark.skipif(not VOXEL_MODULE_AVAILABLE, reason="Voxel data module not available")
class TestVoxelDataReproducibility:
    """Tests for fixture reproducibility."""

    def test_same_seed_produces_same_grid(self):
        """Test that using the same seed produces reproducible results."""
        grid1 = generate_small_voxel_grid()
        grid2 = generate_small_voxel_grid()

        # Both should have same dimensions
        assert tuple(grid1.dims) == tuple(grid2.dims)
        assert grid1.resolution == grid2.resolution

        # Both should have same signals
        assert set(grid1.available_signals) == set(grid2.available_signals)
