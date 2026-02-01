"""
Unit tests for VoxelVisualizationWidgets (notebook_widgets).
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from am_qadf.visualization.notebook_widgets import VoxelVisualizationWidgets

try:
    import ipywidgets as widgets

    WIDGETS_AVAILABLE = True
except ImportError:
    WIDGETS_AVAILABLE = False


class MockVoxelGrid:
    """Mock VoxelGrid for testing."""

    def __init__(self):
        self.dims = np.array([10, 10, 10])
        self.resolution = 1.0
        self.bbox_min = np.array([0.0, 0.0, 0.0])
        self.bbox_max = np.array([10.0, 10.0, 10.0])
        self.available_signals = {"power", "velocity", "energy"}

    def get_bounding_box(self):
        """Return bounding box."""
        return self.bbox_min, self.bbox_max


class MockQueryClient:
    """Mock QueryClient for testing."""

    def get_layer_count(self):
        """Return mock layer count."""
        return 100

    def list_components(self):
        """Return mock component list."""
        return ["component_1", "component_2"]


@pytest.mark.skipif(not WIDGETS_AVAILABLE, reason="ipywidgets not installed")
class TestVoxelVisualizationWidgets:
    """Test cases for VoxelVisualizationWidgets (ParaView-only dashboard)."""

    @pytest.fixture
    def voxel_grid(self):
        """Create a mock voxel grid."""
        return MockVoxelGrid()

    @pytest.fixture
    def query_client(self):
        """Create a mock query client."""
        return MockQueryClient()

    @pytest.fixture
    def widgets_instance(self, voxel_grid):
        """Create a VoxelVisualizationWidgets instance."""
        return VoxelVisualizationWidgets(voxel_grid=voxel_grid)

    @pytest.mark.unit
    def test_initialization(self, voxel_grid):
        """Test widget initialization."""
        widgets_instance = VoxelVisualizationWidgets(voxel_grid=voxel_grid)
        assert widgets_instance.voxel_grid == voxel_grid
        assert widgets_instance.output_dir is not None
        assert widgets_instance._widgets == {}
        assert widgets_instance._dashboard is None

    @pytest.mark.unit
    def test_initialization_without_widgets(self):
        """Test that initialization fails without ipywidgets."""
        with patch("am_qadf.visualization.notebook_widgets.WIDGETS_AVAILABLE", False):
            with pytest.raises(ImportError, match="ipywidgets is required"):
                VoxelVisualizationWidgets()

    @pytest.mark.unit
    def test_initialization_with_query_client(self, query_client):
        """Test initialization with query client."""
        widgets_instance = VoxelVisualizationWidgets(query_client=query_client)
        assert widgets_instance.query_client == query_client

    @pytest.mark.unit
    def test_create_widgets(self, widgets_instance):
        """Test widget creation."""
        dashboard = widgets_instance.create_widgets()
        assert dashboard is not None
        assert widgets_instance._dashboard is not None
        assert "signals" in widgets_instance._widgets
        assert "export_button" in widgets_instance._widgets

    @pytest.mark.unit
    def test_widget_structure(self, widgets_instance):
        """Test that all expected widgets are created."""
        widgets_instance.create_widgets()
        expected_widgets = [
            "signals",
            "filename",
            "output_dir_display",
            "export_button",
            "launch_button",
            "component",
            "layer_start",
            "layer_end",
            "grid_info",
        ]
        for widget_name in expected_widgets:
            assert widget_name in widgets_instance._widgets

    @pytest.mark.unit
    def test_signal_selector_options(self, widgets_instance):
        """Test signal selector has correct options."""
        widgets_instance.create_widgets()
        signal_widget = widgets_instance._widgets["signals"]
        # Should have signals from voxel grid (SelectMultiple has .options)
        assert len(signal_widget.options) > 0

    @pytest.mark.unit
    def test_component_selector_with_query_client(self, query_client):
        """Test component selector with query client."""
        widgets_instance = VoxelVisualizationWidgets(query_client=query_client)
        widgets_instance.create_widgets()
        component_widget = widgets_instance._widgets["component"]
        # Should have components from query client
        assert len(component_widget.options) > 0

    @pytest.mark.unit
    def test_component_selector_without_query_client(self, widgets_instance):
        """Test component selector without query client."""
        widgets_instance.create_widgets()
        component_widget = widgets_instance._widgets["component"]
        # Should have default "All Components"
        assert len(component_widget.options) > 0

    @pytest.mark.unit
    def test_slice_position_widgets(self, widgets_instance):
        """Test layer range widgets (ParaView dashboard uses layer_start/layer_end)."""
        widgets_instance.create_widgets()
        for key in ["layer_start", "layer_end"]:
            w = widgets_instance._widgets[key]
            assert w is not None
            assert hasattr(w, "value")
            assert hasattr(w, "min")
            assert hasattr(w, "max")

    @pytest.mark.unit
    def test_display(self, widgets_instance):
        """Test dashboard display."""
        with patch("am_qadf.visualization.notebook_widgets.display") as mock_display:
            widgets_instance.display()
            # display() is called for the dashboard; create_widgets() may also call it for HTML
            mock_display.assert_any_call(widgets_instance._dashboard)

    @pytest.mark.unit
    def test_display_creates_widgets_if_needed(self, voxel_grid):
        """Test that display creates widgets if not already created."""
        widgets_instance = VoxelVisualizationWidgets(voxel_grid=voxel_grid)
        assert widgets_instance._dashboard is None

        with patch("am_qadf.visualization.notebook_widgets.display"):
            widgets_instance.display()
            assert widgets_instance._dashboard is not None

    @pytest.mark.unit
    def test_layer_range_widgets(self, widgets_instance):
        """Test layer range widgets."""
        widgets_instance.create_widgets()
        layer_start = widgets_instance._widgets["layer_start"]
        layer_end = widgets_instance._widgets["layer_end"]
        # layer_start max is max_layers-1, layer_end max is max_layers
        assert layer_start.max <= layer_end.max

    @pytest.mark.unit
    def test_layer_range_with_query_client(self, query_client):
        """Test layer range with query client."""
        widgets_instance = VoxelVisualizationWidgets(query_client=query_client)
        widgets_instance.create_widgets()
        layer_start = widgets_instance._widgets["layer_start"]
        # Should use layer count from query client
        assert layer_start.max == 99  # 0-indexed, so max is count - 1

    @pytest.mark.unit
    def test_resolution_widget(self, widgets_instance):
        """ParaView-only dashboard has no resolution widget; verify create_widgets runs."""
        widgets_instance.create_widgets()
        # Current implementation has no "resolution" widget; ensure dashboard exists
        assert "resolution" not in widgets_instance._widgets
        assert widgets_instance._dashboard is not None

    @pytest.mark.unit
    def test_colormap_widget(self, widgets_instance):
        """ParaView-only dashboard has no colormap widget; verify create_widgets runs."""
        widgets_instance.create_widgets()
        assert "colormap" not in widgets_instance._widgets
        assert widgets_instance._dashboard is not None

    @pytest.mark.unit
    def test_dashboard_layout(self, widgets_instance):
        """Test dashboard layout structure."""
        dashboard = widgets_instance.create_widgets()
        assert dashboard is not None
        # Dashboard should be a VBox
        assert hasattr(dashboard, "children")
