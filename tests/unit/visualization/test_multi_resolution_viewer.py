"""
Unit tests for MultiResolutionViewer.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from am_qadf.visualization.multi_resolution_viewer import MultiResolutionViewer

try:
    import pyvista as pv

    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False


class MockVoxelGrid:
    """Mock VoxelGrid for testing."""

    def __init__(self, dims=(10, 10, 10), resolution=1.0):
        self.dims = np.array(dims)
        self.resolution = resolution
        self.bbox_min = np.array([0.0, 0.0, 0.0])
        self.bbox_max = np.array([10.0, 10.0, 10.0])
        self.available_signals = {"power", "velocity"}

    def get_signal_array(self, signal_name, default=0.0):
        """Return a mock signal array."""
        return np.random.rand(*self.dims) * 100.0

    def get_statistics(self):
        """Return mock statistics."""
        return {
            "filled_voxels": int(np.prod(self.dims) * 0.5),
            "total_voxels": int(np.prod(self.dims)),
        }

    def get_bounding_box(self):
        """Return bounding box."""
        return self.bbox_min, self.bbox_max


class MockMultiResolutionGrid:
    """Mock MultiResolutionGrid for testing."""

    def __init__(self, num_levels=3):
        self.num_levels = num_levels
        self.base_resolution = 1.0
        self.bbox_min = np.array([0.0, 0.0, 0.0])
        self.bbox_max = np.array([10.0, 10.0, 10.0])
        self._levels = {}

        # Create mock levels
        for i in range(num_levels):
            resolution = self.base_resolution * (2**i)
            dims = (10 // (2**i), 10 // (2**i), 10 // (2**i))
            self._levels[i] = MockVoxelGrid(dims=dims, resolution=resolution)

    def get_level(self, level):
        """Get grid for a specific level."""
        return self._levels.get(level)

    def get_resolution(self, level):
        """Get resolution for a specific level."""
        return self.base_resolution * (2**level)

    def get_level_for_view_distance(self, distance):
        """Get level based on view distance."""
        # Simple logic: closer = higher level
        if distance < 10:
            return 2
        elif distance < 50:
            return 1
        else:
            return 0


@pytest.mark.skipif(not PYVISTA_AVAILABLE, reason="PyVista not installed")
class TestMultiResolutionViewer:
    """Test cases for MultiResolutionViewer."""

    @pytest.fixture
    def multi_resolution_grid(self):
        """Create a mock multi-resolution grid."""
        return MockMultiResolutionGrid(num_levels=3)

    @pytest.fixture
    def viewer(self, multi_resolution_grid):
        """Create a MultiResolutionViewer instance."""
        return MultiResolutionViewer(multi_resolution_grid=multi_resolution_grid, performance_mode="balanced")

    @pytest.mark.unit
    def test_initialization(self, multi_resolution_grid):
        """Test viewer initialization."""
        viewer = MultiResolutionViewer(multi_resolution_grid=multi_resolution_grid, performance_mode="balanced")
        assert viewer.multi_resolution_grid == multi_resolution_grid
        assert viewer._current_level == 0
        assert viewer.resolution_selector is not None

    @pytest.mark.unit
    def test_initialization_without_pyvista(self):
        """Test that initialization fails without PyVista."""
        with patch("am_qadf.visualization.multi_resolution_viewer.PYVISTA_AVAILABLE", False):
            with pytest.raises(ImportError, match="PyVista is required"):
                MultiResolutionViewer()

    @pytest.mark.unit
    def test_initialization_performance_modes(self, multi_resolution_grid):
        """Test initialization with different performance modes."""
        for mode in ["fast", "balanced", "quality"]:
            viewer = MultiResolutionViewer(multi_resolution_grid=multi_resolution_grid, performance_mode=mode)
            assert viewer.resolution_selector.performance_mode == mode

    @pytest.mark.unit
    def test_set_view_parameters(self, viewer):
        """Test setting view parameters."""
        viewer.set_view_parameters(distance=50.0, zoom=2.0, region_size=100.0)
        assert viewer._view_parameters["distance"] == 50.0
        assert viewer._view_parameters["zoom"] == 2.0
        assert viewer._view_parameters["region_size"] == 100.0

    @pytest.mark.unit
    def test_set_view_parameters_partial(self, viewer):
        """Test setting partial view parameters."""
        viewer.set_view_parameters(distance=30.0)
        assert viewer._view_parameters["distance"] == 30.0
        assert "zoom" not in viewer._view_parameters

    @pytest.mark.unit
    def test_select_resolution_manual(self, viewer):
        """Test manual resolution selection."""
        level = viewer.select_resolution(method="manual", target_level=2)
        assert level == 2
        assert viewer._current_level == 2

    @pytest.mark.unit
    def test_select_resolution_view(self, viewer):
        """Test view-based resolution selection."""
        level = viewer.select_resolution(method="view", view_distance=5.0)
        assert level in [0, 1, 2]
        assert viewer._current_level == level

    @pytest.mark.unit
    def test_select_resolution_auto(self, viewer):
        """Test automatic resolution selection."""
        viewer.set_view_parameters(distance=20.0)
        level = viewer.select_resolution(method="auto")
        assert level in [0, 1, 2]

    @pytest.mark.unit
    def test_select_resolution_performance(self, viewer):
        """Test performance-based resolution selection."""
        level = viewer.select_resolution(method="performance")
        assert level in [0, 1, 2]

    @pytest.mark.unit
    def test_select_resolution_without_grid(self):
        """Test resolution selection without grid raises error."""
        viewer = MultiResolutionViewer()
        with pytest.raises(ValueError, match="MultiResolutionGrid not set"):
            viewer.select_resolution()

    @pytest.mark.unit
    def test_render_3d_basic(self, viewer):
        """Test basic 3D rendering."""
        plotter = viewer.render_3d(signal_name="power", level=1, auto_select_level=False, auto_show=False)
        assert plotter is not None

    @pytest.mark.unit
    def test_render_3d_auto_select_level(self, viewer):
        """Test 3D rendering with auto level selection."""
        plotter = viewer.render_3d(signal_name="power", auto_select_level=True, auto_show=False)
        assert plotter is not None

    @pytest.mark.unit
    def test_render_3d_with_view_distance(self, viewer):
        """Test 3D rendering with view distance."""
        plotter = viewer.render_3d(
            signal_name="power",
            view_distance=15.0,
            auto_select_level=True,
            auto_show=False,
        )
        assert plotter is not None

    @pytest.mark.unit
    def test_render_3d_adaptive_threshold(self, viewer):
        """Test 3D rendering with adaptive threshold."""
        plotter = viewer.render_3d(
            signal_name="power",
            level=1,
            threshold=None,  # Should use adaptive
            adaptive_threshold=True,
            auto_show=False,
        )
        assert plotter is not None

    @pytest.mark.unit
    def test_render_3d_fixed_threshold(self, viewer):
        """Test 3D rendering with fixed threshold."""
        plotter = viewer.render_3d(
            signal_name="power",
            level=1,
            threshold=0.5,
            adaptive_threshold=False,
            auto_show=False,
        )
        assert plotter is not None

    @pytest.mark.unit
    def test_render_3d_without_grid(self):
        """Test 3D rendering without grid raises error."""
        viewer = MultiResolutionViewer()
        with pytest.raises(ValueError, match="MultiResolutionGrid not set"):
            viewer.render_3d()

    @pytest.mark.unit
    def test_render_3d_invalid_level(self, viewer):
        """Test 3D rendering with invalid level raises error."""
        with pytest.raises(ValueError, match="Level.*not found"):
            viewer.render_3d(level=999, auto_select_level=False)

    @pytest.mark.unit
    def test_render_slice_basic(self, viewer):
        """Test basic slice rendering."""
        plotter = viewer.render_slice(
            signal_name="power",
            axis="z",
            level=1,
            auto_select_level=False,
            auto_show=False,
        )
        assert plotter is not None

    @pytest.mark.unit
    def test_render_slice_auto_select_level(self, viewer):
        """Test slice rendering with auto level selection."""
        plotter = viewer.render_slice(signal_name="power", axis="z", auto_select_level=True, auto_show=False)
        assert plotter is not None

    @pytest.mark.unit
    def test_render_slice_all_axes(self, viewer):
        """Test slice rendering for all axes."""
        for axis in ["x", "y", "z"]:
            plotter = viewer.render_slice(
                signal_name="power",
                axis=axis,
                level=1,
                auto_select_level=False,
                auto_show=False,
            )
            assert plotter is not None

    @pytest.mark.unit
    def test_render_slice_with_position(self, viewer):
        """Test slice rendering with specific position."""
        plotter = viewer.render_slice(
            signal_name="power",
            axis="z",
            position=5.0,
            level=1,
            auto_select_level=False,
            auto_show=False,
        )
        assert plotter is not None

    @pytest.mark.unit
    def test_get_level_info(self, viewer):
        """Test getting level information."""
        info = viewer.get_level_info()
        assert isinstance(info, dict)
        assert len(info) == 3  # 3 levels
        assert 0 in info
        assert 1 in info
        assert 2 in info

    @pytest.mark.unit
    def test_get_level_info_structure(self, viewer):
        """Test structure of level information."""
        info = viewer.get_level_info()
        level_0_info = info[0]
        assert "resolution" in level_0_info
        assert "dimensions" in level_0_info
        assert "num_voxels" in level_0_info
        assert "filled_voxels" in level_0_info
        assert "available_signals" in level_0_info

    @pytest.mark.unit
    def test_get_level_info_without_grid(self):
        """Test getting level info without grid returns empty dict."""
        viewer = MultiResolutionViewer()
        info = viewer.get_level_info()
        assert info == {}

    @pytest.mark.unit
    def test_adaptive_threshold_calculation(self, viewer):
        """Test adaptive threshold calculation based on fill ratio."""
        # Test with sparse data (low fill ratio)
        sparse_grid = MockVoxelGrid(dims=(10, 10, 10))
        sparse_grid.get_statistics = lambda: {"filled_voxels": 10, "total_voxels": 1000}
        viewer.multi_resolution_grid._levels[1] = sparse_grid

        plotter = viewer.render_3d(
            signal_name="power",
            level=1,
            threshold=None,
            adaptive_threshold=True,
            auto_show=False,
        )
        assert plotter is not None

    @pytest.mark.unit
    def test_resolution_selection_default(self, viewer):
        """Test default resolution selection when no view parameters."""
        level = viewer.select_resolution(method="auto")
        # Should default to medium level
        assert level in [0, 1, 2]
