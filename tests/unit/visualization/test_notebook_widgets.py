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
        self.available_signals = {"power", "velocity", "energy"}

    def get_bounding_box(self):
        """Return bounding box."""
        return self.bbox_min, self.bbox_max


class MockRenderer:
    """Mock VoxelRenderer for testing."""

    def __init__(self):
        pass

    def render_3d(self, **kwargs):
        """Return mock plotter."""
        plotter = Mock()
        plotter.show = Mock()
        return plotter

    def render_slice(self, **kwargs):
        """Return mock plotter."""
        plotter = Mock()
        plotter.show = Mock()
        return plotter


class MockQueryClient:
    """Mock QueryClient for testing."""

    def get_layer_count(self):
        """Return mock layer count."""
        return 100

    def list_components(self):
        """Return mock component list."""
        return ["component_1", "component_2"]


@pytest.mark.skipif(not WIDGETS_AVAILABLE, reason="ipywidgets not installed")
@pytest.mark.skipif(not PYVISTA_AVAILABLE, reason="PyVista not installed")
class TestVoxelVisualizationWidgets:
    """Test cases for VoxelVisualizationWidgets."""

    @pytest.fixture
    def voxel_grid(self):
        """Create a mock voxel grid."""
        return MockVoxelGrid()

    @pytest.fixture
    def renderer(self):
        """Create a mock renderer."""
        return MockRenderer()

    @pytest.fixture
    def query_client(self):
        """Create a mock query client."""
        return MockQueryClient()

    @pytest.fixture
    def widgets_instance(self, voxel_grid, renderer):
        """Create a VoxelVisualizationWidgets instance."""
        return VoxelVisualizationWidgets(voxel_grid=voxel_grid, renderer=renderer)

    @pytest.mark.unit
    def test_initialization(self, voxel_grid, renderer):
        """Test widget initialization."""
        widgets_instance = VoxelVisualizationWidgets(voxel_grid=voxel_grid, renderer=renderer)
        assert widgets_instance.voxel_grid == voxel_grid
        assert widgets_instance.renderer == renderer
        assert widgets_instance.output_3d is not None
        assert widgets_instance.output_slice_x is not None

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
        assert "signal" in widgets_instance._widgets
        assert "colormap" in widgets_instance._widgets

    @pytest.mark.unit
    def test_widget_structure(self, widgets_instance):
        """Test that all expected widgets are created."""
        widgets_instance.create_widgets()
        expected_widgets = [
            "signal",
            "resolution",
            "layer_start",
            "layer_end",
            "component",
            "colormap",
            "slice_x",
            "slice_y",
            "slice_z",
        ]
        for widget_name in expected_widgets:
            assert widget_name in widgets_instance._widgets

    @pytest.mark.unit
    def test_signal_selector_options(self, widgets_instance):
        """Test signal selector has correct options."""
        widgets_instance.create_widgets()
        signal_widget = widgets_instance._widgets["signal"]
        # Should have signals from voxel grid
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
        # Should have default component
        assert len(component_widget.options) > 0

    @pytest.mark.unit
    def test_slice_position_widgets(self, widgets_instance):
        """Test slice position widgets are created correctly."""
        widgets_instance.create_widgets()
        for axis in ["x", "y", "z"]:
            slice_widget = widgets_instance._widgets[f"slice_{axis}"]
            assert slice_widget is not None
            assert hasattr(slice_widget, "value")
            assert hasattr(slice_widget, "min")
            assert hasattr(slice_widget, "max")

    @pytest.mark.unit
    def test_on_widget_change(self, widgets_instance):
        """Test widget change handler."""
        widgets_instance.create_widgets()
        with patch.object(widgets_instance, "update_visualizations") as mock_update:
            change = {"new": "test"}
            widgets_instance._on_widget_change(change)
            mock_update.assert_called_once()

    @pytest.mark.unit
    def test_update_visualizations_3d(self, widgets_instance):
        """Test 3D visualization update."""
        widgets_instance.create_widgets()
        widgets_instance._widgets["signal"].value = "power"
        widgets_instance._widgets["colormap"].value = "plasma"

        with patch("am_qadf.visualization.notebook_widgets.clear_output"):
            with patch.object(widgets_instance.renderer, "render_3d") as mock_render:
                mock_plotter = Mock()
                mock_plotter.show = Mock()
                mock_render.return_value = mock_plotter
                widgets_instance.update_visualizations()
                mock_render.assert_called_once()

    @pytest.mark.unit
    def test_update_visualizations_slices(self, widgets_instance):
        """Test slice visualization updates."""
        widgets_instance.create_widgets()
        widgets_instance._widgets["signal"].value = "power"
        widgets_instance._widgets["colormap"].value = "plasma"

        with patch("am_qadf.visualization.notebook_widgets.clear_output"):
            with patch.object(widgets_instance.renderer, "render_slice") as mock_render:
                mock_plotter = Mock()
                mock_plotter.show = Mock()
                mock_render.return_value = mock_plotter
                widgets_instance.update_visualizations()
                # Should render 3 slices
                assert mock_render.call_count == 3

    @pytest.mark.unit
    def test_update_visualizations_without_renderer(self, voxel_grid):
        """Test update without renderer does nothing."""
        widgets_instance = VoxelVisualizationWidgets(voxel_grid=voxel_grid)
        widgets_instance.create_widgets()
        # Should not raise error, just return
        widgets_instance.update_visualizations()

    @pytest.mark.unit
    def test_update_visualizations_without_grid(self, renderer):
        """Test update without grid does nothing."""
        widgets_instance = VoxelVisualizationWidgets(renderer=renderer)
        widgets_instance.create_widgets()
        # Should not raise error, just return
        widgets_instance.update_visualizations()

    @pytest.mark.unit
    def test_update_visualizations_error_handling(self, widgets_instance):
        """Test error handling in visualization update."""
        widgets_instance.create_widgets()
        widgets_instance.renderer.render_3d = Mock(side_effect=Exception("Test error"))

        with patch("am_qadf.visualization.notebook_widgets.clear_output"):
            with patch("builtins.print") as mock_print:
                widgets_instance.update_visualizations()
                # Should print error message
                assert mock_print.called

    @pytest.mark.unit
    def test_display(self, widgets_instance):
        """Test dashboard display."""
        with patch("am_qadf.visualization.notebook_widgets.display") as mock_display:
            widgets_instance.display()
            mock_display.assert_called_once()

    @pytest.mark.unit
    def test_display_creates_widgets_if_needed(self, voxel_grid, renderer):
        """Test that display creates widgets if not already created."""
        widgets_instance = VoxelVisualizationWidgets(voxel_grid=voxel_grid, renderer=renderer)
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
        assert layer_start.max < layer_end.max  # End should be >= start

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
        """Test resolution widget."""
        widgets_instance.create_widgets()
        resolution_widget = widgets_instance._widgets["resolution"]
        assert resolution_widget.min == 0.1
        assert resolution_widget.max == 5.0
        assert resolution_widget.step == 0.1

    @pytest.mark.unit
    def test_colormap_widget(self, widgets_instance):
        """Test colormap widget."""
        widgets_instance.create_widgets()
        colormap_widget = widgets_instance._widgets["colormap"]
        assert "plasma" in colormap_widget.options
        assert "viridis" in colormap_widget.options

    @pytest.mark.unit
    def test_dashboard_layout(self, widgets_instance):
        """Test dashboard layout structure."""
        dashboard = widgets_instance.create_widgets()
        assert dashboard is not None
        # Dashboard should be a VBox
        assert hasattr(dashboard, "children")
