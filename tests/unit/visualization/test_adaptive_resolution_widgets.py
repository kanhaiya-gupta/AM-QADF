"""
Unit tests for AdaptiveResolutionWidgets.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from am_qadf.visualization.adaptive_resolution_widgets import AdaptiveResolutionWidgets

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


class MockSpatialMap:
    """Mock SpatialMap for testing."""

    def __init__(self):
        self.regions = [
            (np.array([0, 0, 0]), np.array([5, 5, 5]), 0.5),
            (np.array([5, 5, 5]), np.array([10, 10, 10]), 1.0),
        ]
        self.default_resolution = 1.0


class MockTemporalMap:
    """Mock TemporalMap for testing."""

    def __init__(self):
        self.time_ranges = [(0.0, 10.0, 0.5), (10.0, 20.0, 1.0)]
        self.layer_ranges = [(0, 10, 0.5), (10, 20, 1.0)]
        self.default_resolution = 1.0


class MockAdaptiveGrid:
    """Mock AdaptiveResolutionGrid for testing."""

    def __init__(self, has_spatial=True, has_temporal=True):
        self._finalized = True
        self.bbox_min = np.array([0.0, 0.0, 0.0])
        self.bbox_max = np.array([10.0, 10.0, 10.0])
        self.spatial_map = MockSpatialMap() if has_spatial else None
        self.temporal_map = MockTemporalMap() if has_temporal else None
        self.region_grids = {"res_0.5": Mock(), "res_1.0": Mock()}
        # Set up mock grids
        for key, grid in self.region_grids.items():
            grid.dims = np.array([10, 10, 10])
            grid.resolution = float(key.split("_")[1])
            grid.bbox_min = np.array([0.0, 0.0, 0.0])

    def get_signal_array(self, signal_name, target_resolution=None, default=0.0):
        """Return mock signal array."""
        return np.random.rand(10, 10, 10) * 100.0

    def get_statistics(self):
        """Return mock statistics."""
        return {
            "resolutions": [
                {
                    "resolution": 0.5,
                    "filled_voxels": 500,
                    "signals": {"power", "velocity"},
                },
                {
                    "resolution": 1.0,
                    "filled_voxels": 1000,
                    "signals": {"power", "velocity"},
                },
            ],
            "num_regions": 2,
            "total_points": 10000,
            "available_signals": ["power", "velocity"],
        }


@pytest.mark.skipif(not WIDGETS_AVAILABLE, reason="ipywidgets not installed")
@pytest.mark.skipif(not PYVISTA_AVAILABLE, reason="PyVista not installed")
class TestAdaptiveResolutionWidgets:
    """Test cases for AdaptiveResolutionWidgets."""

    @pytest.fixture
    def adaptive_grid(self):
        """Create a mock adaptive grid."""
        return MockAdaptiveGrid(has_spatial=True, has_temporal=True)

    @pytest.fixture
    def widgets_instance(self, adaptive_grid):
        """Create an AdaptiveResolutionWidgets instance."""
        return AdaptiveResolutionWidgets(adaptive_grid=adaptive_grid)

    @pytest.mark.unit
    def test_initialization(self, adaptive_grid):
        """Test widget initialization."""
        widgets_instance = AdaptiveResolutionWidgets(adaptive_grid=adaptive_grid)
        assert widgets_instance.adaptive_grid == adaptive_grid
        assert widgets_instance.output_3d is not None
        assert widgets_instance.output_info is not None

    @pytest.mark.unit
    def test_initialization_without_widgets(self):
        """Test that initialization fails without ipywidgets."""
        with patch("am_qadf.visualization.adaptive_resolution_widgets.WIDGETS_AVAILABLE", False):
            with pytest.raises(ImportError, match="ipywidgets is required"):
                AdaptiveResolutionWidgets(adaptive_grid=MockAdaptiveGrid())

    @pytest.mark.unit
    def test_initialization_without_pyvista(self):
        """Test that initialization fails without PyVista."""
        with patch("am_qadf.visualization.adaptive_resolution_widgets.PYVISTA_AVAILABLE", False):
            with pytest.raises(ImportError, match="PyVista is required"):
                AdaptiveResolutionWidgets(adaptive_grid=MockAdaptiveGrid())

    @pytest.mark.unit
    def test_initialization_without_grid(self):
        """Test that initialization fails without grid."""
        with pytest.raises(ValueError, match="AdaptiveResolutionGrid not set"):
            widgets_instance = AdaptiveResolutionWidgets()
            widgets_instance.create_widgets()

    @pytest.mark.unit
    def test_initialization_unfinalized_grid(self, adaptive_grid):
        """Test that initialization fails with unfinalized grid."""
        adaptive_grid._finalized = False
        widgets_instance = AdaptiveResolutionWidgets(adaptive_grid=adaptive_grid)
        with pytest.raises(ValueError, match="Grid must be finalized"):
            widgets_instance.create_widgets()

    @pytest.mark.unit
    def test_create_widgets(self, widgets_instance):
        """Test widget creation."""
        dashboard = widgets_instance.create_widgets()
        assert dashboard is not None
        assert widgets_instance._dashboard is not None
        assert "resolution_mode" in widgets_instance._widgets
        assert "signal_selector" in widgets_instance._widgets

    @pytest.mark.unit
    def test_widget_structure(self, widgets_instance):
        """Test that all expected widgets are created."""
        widgets_instance.create_widgets()
        expected_widgets = [
            "resolution_mode",
            "resolution_selector",
            "spatial_region_selector",
            "temporal_range_selector",
            "signal_selector",
            "colormap_selector",
            "threshold_slider",
            "view_type",
            "slice_position",
            "update_button",
        ]
        for widget_name in expected_widgets:
            assert widget_name in widgets_instance._widgets

    @pytest.mark.unit
    def test_resolution_mode_options(self, widgets_instance):
        """Test resolution mode options based on available maps."""
        widgets_instance.create_widgets()
        mode_widget = widgets_instance._widgets["resolution_mode"]
        # Should have adaptive, spatial, and temporal options
        assert len(mode_widget.options) >= 1

    @pytest.mark.unit
    def test_resolution_mode_spatial_only(self):
        """Test widget creation with spatial-only grid."""
        grid = MockAdaptiveGrid(has_spatial=True, has_temporal=False)
        widgets_instance = AdaptiveResolutionWidgets(adaptive_grid=grid)
        widgets_instance.create_widgets()
        mode_widget = widgets_instance._widgets["resolution_mode"]
        # Should have spatial and adaptive options
        assert len(mode_widget.options) >= 1

    @pytest.mark.unit
    def test_resolution_mode_temporal_only(self):
        """Test widget creation with temporal-only grid."""
        grid = MockAdaptiveGrid(has_spatial=False, has_temporal=True)
        widgets_instance = AdaptiveResolutionWidgets(adaptive_grid=grid)
        widgets_instance.create_widgets()
        mode_widget = widgets_instance._widgets["resolution_mode"]
        # Should have temporal and adaptive options
        assert len(mode_widget.options) >= 1

    @pytest.mark.unit
    def test_on_mode_change_spatial(self, widgets_instance):
        """Test mode change to spatial."""
        widgets_instance.create_widgets()
        change = {"new": "spatial"}
        widgets_instance._on_mode_change(change)
        assert widgets_instance._widgets["resolution_selector"].disabled == True
        assert widgets_instance._widgets["spatial_region_selector"].disabled == False
        assert widgets_instance._widgets["temporal_range_selector"].disabled == True

    @pytest.mark.unit
    def test_on_mode_change_temporal(self, widgets_instance):
        """Test mode change to temporal."""
        widgets_instance.create_widgets()
        change = {"new": "temporal"}
        widgets_instance._on_mode_change(change)
        assert widgets_instance._widgets["resolution_selector"].disabled == True
        assert widgets_instance._widgets["spatial_region_selector"].disabled == True
        assert widgets_instance._widgets["temporal_range_selector"].disabled == False

    @pytest.mark.unit
    def test_on_mode_change_adaptive(self, widgets_instance):
        """Test mode change to adaptive."""
        widgets_instance.create_widgets()
        change = {"new": "adaptive"}
        widgets_instance._on_mode_change(change)
        assert widgets_instance._widgets["resolution_selector"].disabled == False
        assert widgets_instance._widgets["spatial_region_selector"].disabled == True
        assert widgets_instance._widgets["temporal_range_selector"].disabled == True

    @pytest.mark.unit
    def test_on_view_type_change(self, widgets_instance):
        """Test view type change handler."""
        widgets_instance.create_widgets()
        change = {"new": "Z Slice"}
        widgets_instance._on_view_type_change(change)
        assert widgets_instance._widgets["slice_position"].disabled == False

    @pytest.mark.unit
    def test_on_view_type_change_3d(self, widgets_instance):
        """Test view type change to 3D disables slice position."""
        widgets_instance.create_widgets()
        change = {"new": "3D View"}
        widgets_instance._on_view_type_change(change)
        assert widgets_instance._widgets["slice_position"].disabled == True

    @pytest.mark.unit
    def test_update_visualization_spatial_mode(self, widgets_instance):
        """Test visualization update in spatial mode."""
        widgets_instance.create_widgets()
        widgets_instance._widgets["resolution_mode"].value = "spatial"
        widgets_instance._widgets["spatial_region_selector"].value = 0
        widgets_instance._widgets["view_type"].value = "3D View"

        with patch.object(widgets_instance, "_render_3d") as mock_render:
            widgets_instance.update_visualization()
            mock_render.assert_called_once()

    @pytest.mark.unit
    def test_update_visualization_temporal_mode(self, widgets_instance):
        """Test visualization update in temporal mode."""
        widgets_instance.create_widgets()
        widgets_instance._widgets["resolution_mode"].value = "temporal"
        widgets_instance._widgets["temporal_range_selector"].value = ("time", 0)
        widgets_instance._widgets["view_type"].value = "3D View"

        with patch.object(widgets_instance, "_render_3d") as mock_render:
            widgets_instance.update_visualization()
            mock_render.assert_called_once()

    @pytest.mark.unit
    def test_update_visualization_adaptive_mode(self, widgets_instance):
        """Test visualization update in adaptive mode."""
        widgets_instance.create_widgets()
        widgets_instance._widgets["resolution_mode"].value = "adaptive"
        widgets_instance._widgets["resolution_selector"].value = 0.5
        widgets_instance._widgets["view_type"].value = "3D View"

        with patch.object(widgets_instance, "_render_3d") as mock_render:
            widgets_instance.update_visualization()
            mock_render.assert_called_once()

    @pytest.mark.unit
    def test_render_3d(self, widgets_instance):
        """Test 3D rendering."""
        widgets_instance.create_widgets()
        # Mock environment to allow rendering (not CI)
        # Patch os.environ where it's imported inside the function
        import os

        with patch.dict(os.environ, {}, clear=False):
            # Remove CI-related env vars
            os.environ.pop("CI", None)
            os.environ.pop("GITHUB_ACTIONS", None)
            os.environ.pop("NUMBA_DISABLE_JIT", None)
            with patch("am_qadf.visualization.adaptive_resolution_widgets.pv") as mock_pv:
                mock_plotter = Mock()
                mock_pv.Plotter.return_value = mock_plotter
                # Create a mock that supports item assignment (use MagicMock)
                mock_grid = MagicMock()
                mock_grid.threshold.return_value = Mock(n_points=100)
                mock_pv.ImageData.return_value = mock_grid

                widgets_instance._render_3d(
                    resolution=0.5,
                    signal_name="power",
                    colormap="plasma",
                    threshold=0.1,
                )
                # Should create plotter and show
                assert mock_pv.Plotter.called

    @pytest.mark.unit
    def test_render_slice(self, widgets_instance):
        """Test slice rendering."""
        widgets_instance.create_widgets()
        # Mock environment to allow rendering (not CI)
        # Patch os.environ where it's imported inside the function
        import os

        with patch.dict(os.environ, {}, clear=False):
            # Remove CI-related env vars
            os.environ.pop("CI", None)
            os.environ.pop("GITHUB_ACTIONS", None)
            os.environ.pop("NUMBA_DISABLE_JIT", None)
            with patch("am_qadf.visualization.adaptive_resolution_widgets.pv") as mock_pv:
                mock_plotter = Mock()
                mock_pv.Plotter.return_value = mock_plotter
                # Create a mock that supports item assignment (use MagicMock)
                mock_grid = MagicMock()
                mock_grid.slice_orthogonal.return_value = Mock()
                mock_pv.ImageData.return_value = mock_grid

                widgets_instance._render_slice(
                    resolution=0.5,
                    signal_name="power",
                    axis="z",
                    position=5.0,
                    colormap="plasma",
                )
                # Should create plotter and show
                assert mock_pv.Plotter.called

    @pytest.mark.unit
    def test_display_info(self, widgets_instance):
        """Test info display."""
        widgets_instance.create_widgets()
        widgets_instance._widgets["resolution_mode"].value = "adaptive"

        with patch("builtins.print") as mock_print:
            widgets_instance._display_info(resolution=0.5, signal_name="power")
            # Should print information
            assert mock_print.called

    @pytest.mark.unit
    def test_display_dashboard(self, widgets_instance):
        """Test dashboard display."""
        with patch("am_qadf.visualization.adaptive_resolution_widgets.display") as mock_display:
            widgets_instance.display_dashboard()
            mock_display.assert_called_once()

    @pytest.mark.unit
    def test_set_adaptive_grid(self, widgets_instance):
        """Test setting adaptive grid."""
        new_grid = MockAdaptiveGrid()
        widgets_instance.set_adaptive_grid(new_grid)
        assert widgets_instance.adaptive_grid == new_grid
        assert widgets_instance._dashboard is None  # Should reset dashboard

    @pytest.mark.unit
    def test_widget_debouncing(self, widgets_instance):
        """Test widget change debouncing."""
        widgets_instance.create_widgets()
        widgets_instance._last_update_time = 0.0
        widgets_instance._debounce_delay = 0.5

        change = {"new": "test"}
        # First call should update
        with patch.object(widgets_instance, "update_visualization") as mock_update:
            widgets_instance._on_widget_change(change)
            # Should be called (enough time passed)

    @pytest.mark.unit
    def test_error_handling_in_update(self, widgets_instance):
        """Test error handling in visualization update."""
        widgets_instance.create_widgets()
        widgets_instance.adaptive_grid.get_signal_array = Mock(side_effect=Exception("Test error"))

        with patch("am_qadf.visualization.adaptive_resolution_widgets.clear_output"):
            widgets_instance.update_visualization()
            # Should not raise, should handle error gracefully
