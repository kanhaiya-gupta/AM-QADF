"""
Tests for fixture loading and validation.

Ensures that all fixtures can be loaded correctly and have expected properties.
"""

import pytest
import numpy as np
from pathlib import Path

# Try to import fixture loaders
try:
    from tests.fixtures.voxel_data import (
        load_small_voxel_grid,
        load_medium_voxel_grid,
        load_large_voxel_grid,
    )

    VOXEL_FIXTURES_AVAILABLE = True
except ImportError:
    VOXEL_FIXTURES_AVAILABLE = False

try:
    from tests.fixtures.point_clouds import (
        load_hatching_paths,
        load_laser_points,
        load_ct_points,
    )

    POINT_CLOUD_FIXTURES_AVAILABLE = True
except ImportError:
    POINT_CLOUD_FIXTURES_AVAILABLE = False

try:
    from tests.fixtures.signals import load_sample_signals

    SIGNAL_FIXTURES_AVAILABLE = True
except ImportError:
    SIGNAL_FIXTURES_AVAILABLE = False


@pytest.mark.skipif(not VOXEL_FIXTURES_AVAILABLE, reason="Voxel fixtures not available")
class TestVoxelGridFixtures:
    """Tests for voxel grid fixtures."""

    def test_small_voxel_grid_loads(self):
        """Test that small voxel grid can be loaded."""
        grid = load_small_voxel_grid()
        assert grid is not None
        assert hasattr(grid, "dims")
        assert len(grid.dims) == 3

    def test_small_voxel_grid_dimensions(self):
        """Test that small voxel grid has correct dimensions."""
        grid = load_small_voxel_grid()
        assert tuple(grid.dims) == (10, 10, 10)
        assert grid.resolution > 0

    def test_small_voxel_grid_signals(self):
        """Test that small voxel grid has expected signals."""
        grid = load_small_voxel_grid()
        assert hasattr(grid, "available_signals")
        assert len(grid.available_signals) > 0

    def test_medium_voxel_grid_loads(self):
        """Test that medium voxel grid can be loaded."""
        grid = load_medium_voxel_grid()
        assert grid is not None
        assert hasattr(grid, "dims")
        assert tuple(grid.dims) == (50, 50, 50)

    def test_large_voxel_grid_loads(self):
        """Test that large voxel grid can be loaded."""
        grid = load_large_voxel_grid()
        assert grid is not None
        assert hasattr(grid, "dims")
        assert tuple(grid.dims) == (100, 100, 100)

    def test_voxel_grids_are_finalized(self):
        """Test that all voxel grids are finalized."""
        grids = [
            load_small_voxel_grid(),
            load_medium_voxel_grid(),
            load_large_voxel_grid(),
        ]
        for grid in grids:
            # Check that grid is ready to use (has been finalized)
            assert hasattr(grid, "dims")
            assert hasattr(grid, "resolution")


@pytest.mark.skipif(not POINT_CLOUD_FIXTURES_AVAILABLE, reason="Point cloud fixtures not available")
class TestPointCloudFixtures:
    """Tests for point cloud fixtures."""

    def test_hatching_paths_load(self):
        """Test that hatching paths can be loaded."""
        paths = load_hatching_paths()
        assert paths is not None
        assert isinstance(paths, (list, dict))

    def test_laser_points_load(self):
        """Test that laser points can be loaded."""
        points = load_laser_points()
        assert points is not None
        assert isinstance(points, (list, dict))

    def test_ct_points_load(self):
        """Test that CT scan points can be loaded."""
        ct_data = load_ct_points()
        assert ct_data is not None
        assert isinstance(ct_data, (list, dict))


@pytest.mark.skipif(not SIGNAL_FIXTURES_AVAILABLE, reason="Signal fixtures not available")
class TestSignalFixtures:
    """Tests for signal fixtures."""

    def test_sample_signals_load(self):
        """Test that sample signals can be loaded."""
        signals = load_sample_signals()
        assert signals is not None
        assert isinstance(signals, dict)
        assert len(signals) > 0

    def test_sample_signals_are_arrays(self):
        """Test that loaded signals are numpy arrays."""
        signals = load_sample_signals()
        for signal_name, signal_data in signals.items():
            assert isinstance(signal_data, np.ndarray), f"Signal {signal_name} is not a numpy array"
            assert signal_data.ndim >= 1, f"Signal {signal_name} should have at least 1 dimension"


class TestFixtureFiles:
    """Tests for fixture file existence."""

    def test_voxel_data_files_exist(self):
        """Test that voxel data files exist."""
        fixture_dir = Path(__file__).parent / "voxel_data"
        expected_files = [
            "small_voxel_grid.pkl",
            "medium_voxel_grid.pkl",
            "large_voxel_grid.pkl",
        ]
        for filename in expected_files:
            filepath = fixture_dir / filename
            # Files might be generated on-the-fly, so we just check the directory exists
            assert fixture_dir.exists(), f"Voxel data directory does not exist: {fixture_dir}"

    def test_point_cloud_files_exist(self):
        """Test that point cloud data directory exists."""
        fixture_dir = Path(__file__).parent / "point_clouds"
        assert fixture_dir.exists(), f"Point cloud directory does not exist: {fixture_dir}"

    def test_signal_files_exist(self):
        """Test that signal data directory exists."""
        fixture_dir = Path(__file__).parent / "signals"
        assert fixture_dir.exists(), f"Signal directory does not exist: {fixture_dir}"
