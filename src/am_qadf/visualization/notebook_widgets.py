"""
Notebook Widgets

Interactive Jupyter notebook widgets for voxel visualization.
Provides component selection, signal selection, and real-time visualization updates.
"""

from typing import Optional, List, Dict, Any, Callable
import numpy as np

try:
    import ipywidgets as widgets
    from ipywidgets import HBox, VBox, Output, interactive
    from IPython.display import display, clear_output

    WIDGETS_AVAILABLE = True
except ImportError:
    WIDGETS_AVAILABLE = False
    widgets = None
    HBox = VBox = Output = None
    display = clear_output = None

try:
    import pyvista as pv

    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    pv = None


class VoxelVisualizationWidgets:
    """
    Interactive widgets for voxel domain visualization.

    Provides:
    - Component selection (for multi-component builds)
    - Signal type selection
    - Resolution control
    - Layer range selection
    - Colormap selection
    - Slice position controls
    - 2x2 grid layout (3D + 3 slices)
    """

    def __init__(self, query_client=None, voxel_grid=None, renderer=None):
        """
        Initialize visualization widgets.

        Args:
            query_client: Query client for data access
            voxel_grid: VoxelGrid object
            renderer: VoxelRenderer object
        """
        if not WIDGETS_AVAILABLE:
            raise ImportError("ipywidgets is required. Install with: pip install ipywidgets")

        self.query_client = query_client
        self.voxel_grid = voxel_grid
        self.renderer = renderer

        # Output widgets for each section
        self.output_3d = Output()
        self.output_slice_x = Output()
        self.output_slice_y = Output()
        self.output_slice_z = Output()

        # Widget state
        self._widgets = {}
        self._dashboard = None

    def create_widgets(self) -> VBox:
        """
        Create and return the widget dashboard.

        Returns:
            VBox containing all widgets and outputs
        """
        # Available signals (from query client or voxel grid)
        available_signals = ["power", "velocity", "energy"]
        if self.voxel_grid:
            available_signals = list(self.voxel_grid.available_signals)

        # Signal selector
        self._widgets["signal"] = widgets.Dropdown(
            options=[(s.title(), s) for s in available_signals],
            value=available_signals[0] if available_signals else "power",
            description="Signal:",
            style={"description_width": "initial"},
        )

        # Resolution selector
        self._widgets["resolution"] = widgets.FloatSlider(
            value=1.0,
            min=0.1,
            max=5.0,
            step=0.1,
            description="Resolution (mm):",
            style={"description_width": "initial"},
        )

        # Layer range
        max_layers = 100  # Default, will be updated if query_client available
        if self.query_client and hasattr(self.query_client, "get_layer_count"):
            max_layers = self.query_client.get_layer_count()

        self._widgets["layer_start"] = widgets.IntSlider(
            value=0,
            min=0,
            max=max_layers - 1,
            step=1,
            description="Layer Start:",
            style={"description_width": "initial"},
        )

        self._widgets["layer_end"] = widgets.IntSlider(
            value=min(20, max_layers),
            min=1,
            max=max_layers,
            step=1,
            description="Layer End:",
            style={"description_width": "initial"},
        )

        # Component selector (if multi-component)
        components = ["component_1"]  # Default
        if self.query_client and hasattr(self.query_client, "list_components"):
            try:
                components = self.query_client.list_components()
            except:
                pass

        self._widgets["component"] = widgets.Dropdown(
            options=components,
            value=components[0] if components else None,
            description="Component:",
            style={"description_width": "initial"},
            disabled=len(components) <= 1,
        )

        # Colormap selector
        colormaps = ["plasma", "viridis", "hot", "cool", "inferno", "magma"]
        self._widgets["colormap"] = widgets.Dropdown(
            options=colormaps,
            value="plasma",
            description="Colormap:",
            style={"description_width": "initial"},
        )

        # Slice positions
        if self.voxel_grid:
            bbox_min, bbox_max = self.voxel_grid.get_bounding_box()
            x_range = (bbox_min[0], bbox_max[0])
            y_range = (bbox_min[1], bbox_max[1])
            z_range = (bbox_min[2], bbox_max[2])
        else:
            x_range = y_range = z_range = (-50, 50)

        self._widgets["slice_x"] = widgets.FloatSlider(
            value=(x_range[0] + x_range[1]) / 2.0,
            min=x_range[0],
            max=x_range[1],
            step=0.1,
            description="X Slice:",
            style={"description_width": "initial"},
        )

        self._widgets["slice_y"] = widgets.FloatSlider(
            value=(y_range[0] + y_range[1]) / 2.0,
            min=y_range[0],
            max=y_range[1],
            step=0.1,
            description="Y Slice:",
            style={"description_width": "initial"},
        )

        self._widgets["slice_z"] = widgets.FloatSlider(
            value=(z_range[0] + z_range[1]) / 2.0,
            min=z_range[0],
            max=z_range[1],
            step=0.1,
            description="Z Slice:",
            style={"description_width": "initial"},
        )

        # Connect widgets to update functions
        for widget in self._widgets.values():
            if hasattr(widget, "observe"):
                widget.observe(self._on_widget_change, names="value")

        # Create layout
        controls = VBox(
            [
                widgets.HTML("<h3>🎛️ Visualization Controls</h3>"),
                self._widgets["signal"],
                self._widgets["component"],
                widgets.HBox([self._widgets["layer_start"], self._widgets["layer_end"]]),
                widgets.HBox([self._widgets["resolution"], self._widgets["colormap"]]),
                widgets.HTML("<h4>Slice Positions</h4>"),
                widgets.HBox(
                    [
                        self._widgets["slice_x"],
                        self._widgets["slice_y"],
                        self._widgets["slice_z"],
                    ]
                ),
            ]
        )

        # 2x2 grid layout
        top_row = HBox(
            [
                VBox(
                    [widgets.HTML("<h4>3D View</h4>"), self.output_3d],
                    layout=widgets.Layout(width="48%", border="1px solid #ccc", padding="10px"),
                ),
                VBox(
                    [widgets.HTML("<h4>X Slice</h4>"), self.output_slice_x],
                    layout=widgets.Layout(width="48%", border="1px solid #ccc", padding="10px"),
                ),
            ]
        )

        bottom_row = HBox(
            [
                VBox(
                    [widgets.HTML("<h4>Y Slice</h4>"), self.output_slice_y],
                    layout=widgets.Layout(width="48%", border="1px solid #ccc", padding="10px"),
                ),
                VBox(
                    [widgets.HTML("<h4>Z Slice</h4>"), self.output_slice_z],
                    layout=widgets.Layout(width="48%", border="1px solid #ccc", padding="10px"),
                ),
            ]
        )

        self._dashboard = VBox(
            [
                widgets.HTML("<h1>🎨 Voxel Domain Visualization Dashboard</h1>"),
                controls,
                widgets.HTML("<hr>"),
                top_row,
                bottom_row,
            ]
        )

        return self._dashboard

    def _on_widget_change(self, change):
        """Handle widget value changes."""
        self.update_visualizations()

    def update_visualizations(self):
        """Update all visualizations based on current widget values."""
        if self.renderer is None or self.voxel_grid is None:
            return

        try:
            signal = self._widgets["signal"].value
            colormap = self._widgets["colormap"].value
            import time

            # Render plots sequentially with delays to prevent memory overflow
            # Update 3D view - display only in output widget
            with self.output_3d:
                clear_output(wait=True)
                try:
                    plotter = self.renderer.render_3d(
                        signal_name=signal,
                        colormap=colormap,
                        title=f"3D View - {signal.title()}",
                        auto_show=False,
                    )
                    # Show plotter only in this output widget context
                    plotter.show(jupyter_backend="static")
                    # Force garbage collection and small delay
                    import gc

                    gc.collect()
                    time.sleep(0.2)
                except Exception as e:
                    print(f"Error rendering 3D view: {e}")

            # Update slice views - display only in output widgets (one at a time)
            with self.output_slice_x:
                clear_output(wait=True)
                try:
                    plotter = self.renderer.render_slice(
                        signal_name=signal,
                        axis="x",
                        position=self._widgets["slice_x"].value,
                        colormap=colormap,
                        title=f"X Slice - {signal.title()}",
                        auto_show=False,
                    )
                    plotter.show(jupyter_backend="static")
                    import gc

                    gc.collect()
                    time.sleep(0.2)
                except Exception as e:
                    print(f"Error rendering X slice: {e}")

            with self.output_slice_y:
                clear_output(wait=True)
                try:
                    plotter = self.renderer.render_slice(
                        signal_name=signal,
                        axis="y",
                        position=self._widgets["slice_y"].value,
                        colormap=colormap,
                        title=f"Y Slice - {signal.title()}",
                        auto_show=False,
                    )
                    plotter.show(jupyter_backend="static")
                    import gc

                    gc.collect()
                    time.sleep(0.2)
                except Exception as e:
                    print(f"Error rendering Y slice: {e}")

            with self.output_slice_z:
                clear_output(wait=True)
                try:
                    plotter = self.renderer.render_slice(
                        signal_name=signal,
                        axis="z",
                        position=self._widgets["slice_z"].value,
                        colormap=colormap,
                        title=f"Z Slice - {signal.title()}",
                        auto_show=False,
                    )
                    plotter.show(jupyter_backend="static")
                    import gc

                    gc.collect()
                except Exception as e:
                    print(f"Error rendering Z slice: {e}")
        except Exception as e:
            print(f"Error in update_visualizations: {e}")
            import traceback

            traceback.print_exc()

    def display(self):
        """Display the widget dashboard."""
        try:
            if self._dashboard is None:
                self.create_widgets()
            display(self._dashboard)
            # Don't render immediately - let user interact first
            # This prevents kernel crash from rendering 4 plots at once
            # User can trigger rendering by changing any widget value
            print("💡 Dashboard displayed. Change any control to generate visualizations.")
            print("   (Rendering all 4 plots at once can cause kernel crashes)")
        except Exception as e:
            print(f"Error displaying dashboard: {e}")
            import traceback

            traceback.print_exc()
