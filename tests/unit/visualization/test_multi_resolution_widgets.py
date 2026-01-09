"""
Unit tests for MultiResolutionWidgets.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from am_qadf.visualization.multi_resolution_widgets import MultiResolutionWidgets

try:
    import ipywidgets as widgets

    WIDGETS_AVAILABLE = True
except ImportError:
    WIDGETS_AVAILABLE = False

try:
    import pyvista as pv

    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False


class MockVoxelGrid:
    """Mock VoxelGrid for testing."""

    def __init__(self):
        self.dims = np.array([10, 10, 10])
        self.resolution = 1.0
        self.bbox_min = np.array([0.0, 0.0, 0.0])
        self.bbox_max = np.array([10.0, 10.0, 10.0])
        self.available_signals = {"power", "velocity"}


class MockMultiResolutionGrid:
    """Mock MultiResolutionGrid for testing."""

    def __init__(self, num_levels=3):
        self.num_levels = num_levels
        self.base_resolution = 1.0
        self.bbox_min = np.array([0.0, 0.0, 0.0])
        self.bbox_max = np.array([10.0, 10.0, 10.0])
        self._levels = {}

        for i in range(num_levels):
            resolution = self.base_resolution * (2**i)
            dims = (10 // (2**i), 10 // (2**i), 10 // (2**i))
            grid = MockVoxelGrid()
            grid.dims = np.array(dims)
            grid.resolution = resolution
            self._levels[i] = grid

    def get_level(self, level):
        return self._levels.get(level)

    def get_resolution(self, level):
        return self.base_resolution * (2**level)


class MockMultiResolutionViewer:
    """Mock MultiResolutionViewer for testing."""

    def __init__(self):
        self.resolution_selector = Mock()
        self.resolution_selector.performance_mode = "balanced"

    def get_level_info(self):
        return {
            0: {
                "resolution": 1.0,
                "dimensions": [10, 10, 10],
                "num_voxels": 1000,
                "filled_voxels": 500,
                "available_signals": ["power", "velocity"],
            },
            1: {
                "resolution": 2.0,
                "dimensions": [5, 5, 5],
                "num_voxels": 125,
                "filled_voxels": 62,
                "available_signals": ["power", "velocity"],
            },
        }

    def render_3d(self, **kwargs):
        return Mock()

    def render_slice(self, **kwargs):
        return Mock()


@pytest.mark.skipif(not WIDGETS_AVAILABLE, reason="ipywidgets not installed")
@pytest.mark.skipif(not PYVISTA_AVAILABLE, reason="PyVista not installed")
class TestMultiResolutionWidgets:
    """Test cases for MultiResolutionWidgets."""

    @pytest.fixture
    def multi_resolution_grid(self):
        """Create a mock multi-resolution grid."""
        return MockMultiResolutionGrid(num_levels=2)

    @pytest.fixture
    def viewer(self):
        """Create a mock viewer."""
        return MockMultiResolutionViewer()

    @pytest.fixture
    def widgets_instance(self, multi_resolution_grid, viewer):
        """Create a MultiResolutionWidgets instance."""
        return MultiResolutionWidgets(multi_resolution_grid=multi_resolution_grid, viewer=viewer)

    @pytest.mark.unit
    def test_initialization_with_multi_resolution_grid(self, multi_resolution_grid):
        """Test initialization with multi-resolution grid."""
        widgets_instance = MultiResolutionWidgets(multi_resolution_grid=multi_resolution_grid)
        assert widgets_instance.multi_resolution_grid == multi_resolution_grid
        assert widgets_instance.grid_type == "multi_resolution"
        assert widgets_instance.viewer is not None

    @pytest.mark.unit
    def test_initialization_without_widgets(self):
        """Test that initialization fails without ipywidgets."""
        with patch("am_qadf.visualization.multi_resolution_widgets.WIDGETS_AVAILABLE", False):
            with pytest.raises(ImportError, match="ipywidgets is required"):
                MultiResolutionWidgets(multi_resolution_grid=MockMultiResolutionGrid())

    @pytest.mark.unit
    def test_initialization_without_pyvista(self):
        """Test that initialization fails without PyVista."""
        with patch("am_qadf.visualization.multi_resolution_widgets.PYVISTA_AVAILABLE", False):
            with pytest.raises(ImportError, match="PyVista is required"):
                MultiResolutionWidgets(multi_resolution_grid=MockMultiResolutionGrid())

    @pytest.mark.unit
    def test_initialization_without_grid(self):
        """Test that initialization fails without grid."""
        with pytest.raises(
            ValueError,
            match="Either multi_resolution_grid or adaptive_grid must be provided",
        ):
            MultiResolutionWidgets()

    @pytest.mark.unit
    def test_create_widgets(self, widgets_instance):
        """Test widget creation."""
        dashboard = widgets_instance.create_widgets()
        assert dashboard is not None
        assert widgets_instance._dashboard is not None
        assert "performance_mode" in widgets_instance._widgets
        assert "level_selector" in widgets_instance._widgets
        assert "signal_selector" in widgets_instance._widgets

    @pytest.mark.unit
    def test_create_widgets_without_grid(self):
        """Test widget creation without grid raises error."""
        widgets_instance = MultiResolutionWidgets(multi_resolution_grid=MockMultiResolutionGrid())
        widgets_instance.multi_resolution_grid = None
        with pytest.raises(ValueError, match="MultiResolutionGrid not set"):
            widgets_instance.create_widgets()

    @pytest.mark.unit
    def test_widget_structure(self, widgets_instance):
        """Test that all expected widgets are created."""
        widgets_instance.create_widgets()
        expected_widgets = [
            "performance_mode",
            "level_selector",
            "auto_level",
            "signal_selector",
            "colormap_selector",
            "adaptive_threshold",
            "threshold_slider",
            "view_type",
            "slice_position",
            "update_button",
        ]
        for widget_name in expected_widgets:
            assert widget_name in widgets_instance._widgets

    @pytest.mark.unit
    def test_on_performance_change(self, widgets_instance):
        """Test performance mode change handler."""
        widgets_instance.create_widgets()
        change = {"new": "fast"}
        widgets_instance._on_performance_change(change)
        assert widgets_instance.viewer.resolution_selector.performance_mode == "fast"

    @pytest.mark.unit
    def test_on_auto_level_change(self, widgets_instance):
        """Test auto-level toggle change handler."""
        widgets_instance.create_widgets()
        change = {"new": True}
        widgets_instance._on_auto_level_change(change)
        assert widgets_instance._widgets["level_selector"].disabled == True

    @pytest.mark.unit
    def test_on_adaptive_threshold_change(self, widgets_instance):
        """Test adaptive threshold toggle change handler."""
        widgets_instance.create_widgets()
        change = {"new": False}
        widgets_instance._on_adaptive_threshold_change(change)
        assert widgets_instance._widgets["threshold_slider"].disabled == False

    @pytest.mark.unit
    def test_on_view_type_change(self, widgets_instance):
        """Test view type change handler."""
        widgets_instance.create_widgets()
        change = {"new": "slice_z"}
        widgets_instance._on_view_type_change(change)
        assert widgets_instance._widgets["slice_position"].disabled == False

    @pytest.mark.unit
    def test_on_view_type_change_3d(self, widgets_instance):
        """Test view type change to 3D disables slice position."""
        widgets_instance.create_widgets()
        change = {"new": "3d"}
        widgets_instance._on_view_type_change(change)
        assert widgets_instance._widgets["slice_position"].disabled == True

    @pytest.mark.unit
    def test_display_dashboard(self, widgets_instance):
        """Test dashboard display."""
        with patch("am_qadf.visualization.multi_resolution_widgets.display") as mock_display:
            widgets_instance.display_dashboard()
            mock_display.assert_called_once()

    @pytest.mark.unit
    def test_update_visualization_multi_resolution(self, widgets_instance):
        """Test visualization update for multi-resolution grid."""
        widgets_instance.create_widgets()
        widgets_instance._widgets["level_selector"].value = 0
        widgets_instance._widgets["auto_level"].value = False
        widgets_instance._widgets["signal_selector"].value = "power"
        widgets_instance._widgets["view_type"].value = "3d"

        with patch.object(widgets_instance.viewer, "render_3d") as mock_render:
            mock_render.return_value = Mock()
            mock_render.return_value.show = Mock()
            widgets_instance._update_multi_resolution_visualization()
            mock_render.assert_called_once()

    @pytest.mark.unit
    def test_update_visualization_slice(self, widgets_instance):
        """Test visualization update for slice view."""
        widgets_instance.create_widgets()
        widgets_instance._widgets["view_type"].value = "slice_z"
        widgets_instance._widgets["slice_position"].value = 5.0

        with patch.object(widgets_instance.viewer, "render_slice") as mock_render:
            mock_render.return_value = Mock()
            mock_render.return_value.show = Mock()
            widgets_instance._update_multi_resolution_visualization()
            mock_render.assert_called_once()

    @pytest.mark.unit
    def test_widget_debouncing(self, widgets_instance):
        """Test widget change debouncing."""
        widgets_instance.create_widgets()
        widgets_instance._last_update_time = 0.0
        widgets_instance._debounce_delay = 0.3

        change = {"new": "test"}
        # First call should update
        with patch.object(widgets_instance, "_update_visualization") as mock_update:
            widgets_instance._on_widget_change(change)
            # Should be called (enough time passed)
            # Note: actual timing test would require more complex mocking

    @pytest.mark.unit
    def test_error_handling_in_update(self, widgets_instance):
        """Test error handling in visualization update."""
        widgets_instance.create_widgets()
        widgets_instance.viewer.render_3d = Mock(side_effect=Exception("Test error"))

        with patch("am_qadf.visualization.multi_resolution_widgets.clear_output"):
            widgets_instance._update_visualization()
            # Should not raise, should handle error gracefully
