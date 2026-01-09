"""
Multi-Resolution Visualization Widgets

Interactive Jupyter notebook widgets for visualizing multi-resolution voxel grids
with level-of-detail (LOD) support.
"""

from typing import Optional, List, Dict, Any
import numpy as np
import time

try:
    import ipywidgets as widgets
    from ipywidgets import HBox, VBox, Output
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


class MultiResolutionWidgets:
    """
    Interactive widgets for multi-resolution grid visualization.

    Supports both:
    - MultiResolutionGrid (LOD levels)
    - AdaptiveResolutionGrid (spatial/temporal/adaptive modes)

    Provides:
    - Resolution level selection (LOD) or Spatial/Temporal/Adaptive mode selection
    - Signal type selection
    - Colormap selection
    - 3D visualization
    - Slice views
    - Performance mode selection
    """

    def __init__(self, multi_resolution_grid=None, adaptive_grid=None, viewer=None):
        """
        Initialize multi-resolution widgets.

        Args:
            multi_resolution_grid: MultiResolutionGrid object (for LOD viewing)
            adaptive_grid: AdaptiveResolutionGrid object (for spatial/temporal viewing)
            viewer: Optional MultiResolutionViewer object (will create if not provided)
        """
        if not WIDGETS_AVAILABLE:
            raise ImportError("ipywidgets is required. Install with: pip install ipywidgets")

        if not PYVISTA_AVAILABLE:
            raise ImportError("PyVista is required. Install with: pip install pyvista")

        self.multi_resolution_grid = multi_resolution_grid
        self.adaptive_grid = adaptive_grid

        # Determine grid type
        if adaptive_grid is not None:
            self.grid_type = "adaptive"
        elif multi_resolution_grid is not None:
            self.grid_type = "multi_resolution"
        else:
            raise ValueError("Either multi_resolution_grid or adaptive_grid must be provided")

        # Import MultiResolutionViewer if not provided (only for multi-resolution grids)
        if viewer is None and multi_resolution_grid is not None:
            from .multi_resolution_viewer import MultiResolutionViewer

            self.viewer = MultiResolutionViewer(multi_resolution_grid=multi_resolution_grid, performance_mode="balanced")
        else:
            self.viewer = viewer

        # Output widgets
        self.output_3d = Output()
        self.output_info = Output()

        # Widget state
        self._widgets = {}
        self._dashboard = None
        self._last_update_time = 0.0
        self._debounce_delay = 0.3  # 0.3 seconds

    def create_widgets(self) -> VBox:
        """
        Create interactive widgets for multi-resolution visualization.

        Returns:
            VBox containing all widgets
        """
        # Handle adaptive resolution grid
        if self.grid_type == "adaptive":
            return self._create_adaptive_widgets()

        # Handle multi-resolution grid (original behavior)
        if self.multi_resolution_grid is None:
            raise ValueError("MultiResolutionGrid not set")

        # Get available information
        level_info = self.viewer.get_level_info()
        num_levels = len(level_info)

        if num_levels == 0:
            raise ValueError("No resolution levels available")

        # Get available signals from first level
        first_level = list(level_info.keys())[0]
        first_info = level_info[first_level]
        available_signals = list(first_info.get("available_signals", []))

        if not available_signals:
            available_signals = ["power", "velocity", "energy"]  # Default fallback

        # Performance mode selector
        self._widgets["performance_mode"] = widgets.Dropdown(
            options=[
                ("Fast (Low Quality)", "fast"),
                ("Balanced", "balanced"),
                ("Quality (High Detail)", "quality"),
            ],
            value="balanced",
            description="Performance Mode:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="300px"),
        )

        # Resolution level selector
        level_options = []
        for level in sorted(level_info.keys()):
            info = level_info[level]
            resolution = info.get("resolution", 0.0)
            fill_ratio = info.get("filled_voxels", 0) / info.get("num_voxels", 1) if info.get("num_voxels", 0) > 0 else 0.0
            label = f"Level {level} ({resolution:.2f} mm, {fill_ratio*100:.1f}% fill)"
            level_options.append((label, level))

        self._widgets["level_selector"] = widgets.Dropdown(
            options=level_options,
            value=sorted(level_info.keys())[0],
            description="Resolution Level:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="300px"),
        )

        # Auto-select level toggle
        self._widgets["auto_level"] = widgets.Checkbox(
            value=False,
            description="Auto-select level (based on view)",
            style={"description_width": "initial"},
        )

        # Signal selector
        self._widgets["signal_selector"] = widgets.Dropdown(
            options=available_signals,
            value=available_signals[0] if available_signals else "power",
            description="Signal Type:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="300px"),
        )

        # Colormap selector
        colormaps = [
            "plasma",
            "viridis",
            "inferno",
            "magma",
            "coolwarm",
            "jet",
            "hot",
            "cool",
        ]
        self._widgets["colormap_selector"] = widgets.Dropdown(
            options=colormaps,
            value="plasma",
            description="Colormap:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="300px"),
        )

        # Adaptive threshold toggle
        self._widgets["adaptive_threshold"] = widgets.Checkbox(
            value=True,
            description="Adaptive threshold (auto-adjust)",
            style={"description_width": "initial"},
        )

        # Threshold slider (only shown when adaptive_threshold is False)
        self._widgets["threshold_slider"] = widgets.FloatSlider(
            value=0.0,
            min=0.0,
            max=1.0,
            step=0.01,
            description="Threshold:",
            style={"description_width": "initial"},
            disabled=True,
        )

        # View type selector
        self._widgets["view_type"] = widgets.Dropdown(
            options=[
                ("3D Volume", "3d"),
                ("Slice X", "slice_x"),
                ("Slice Y", "slice_y"),
                ("Slice Z", "slice_z"),
            ],
            value="3d",
            description="View Type:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="300px"),
        )

        # Slice position slider (only for slice views)
        self._widgets["slice_position"] = widgets.FloatSlider(
            value=0.5,
            min=0.0,
            max=1.0,
            step=0.01,
            description="Slice Position:",
            style={"description_width": "initial"},
            disabled=True,
        )

        # Update button
        self._widgets["update_button"] = widgets.Button(
            description="🔄 Update Visualization",
            button_style="info",
            layout=widgets.Layout(width="200px"),
        )

        # Connect widgets to update functions
        self._widgets["performance_mode"].observe(self._on_performance_change, names="value")
        self._widgets["auto_level"].observe(self._on_auto_level_change, names="value")
        self._widgets["adaptive_threshold"].observe(self._on_adaptive_threshold_change, names="value")
        self._widgets["view_type"].observe(self._on_view_type_change, names="value")
        self._widgets["update_button"].on_click(self._on_update_click)

        # Auto-update on change (with debouncing)
        for key in [
            "level_selector",
            "signal_selector",
            "colormap_selector",
            "threshold_slider",
            "slice_position",
        ]:
            if key in self._widgets:
                self._widgets[key].observe(self._on_widget_change, names="value")

        # Create layout
        controls = VBox(
            [
                widgets.HTML("<h3>🎨 Multi-Resolution Visualization</h3>"),
                self._widgets["performance_mode"],
                HBox([self._widgets["level_selector"], self._widgets["auto_level"]]),
                self._widgets["signal_selector"],
                HBox(
                    [
                        self._widgets["colormap_selector"],
                        self._widgets["adaptive_threshold"],
                    ]
                ),
                self._widgets["threshold_slider"],
                self._widgets["view_type"],
                self._widgets["slice_position"],
                self._widgets["update_button"],
            ]
        )

        dashboard = VBox([HBox([controls, self.output_3d]), self.output_info])

        self._dashboard = dashboard
        return dashboard

    def _create_adaptive_widgets(self) -> VBox:
        """Create widgets for adaptive resolution grid with spatial/temporal/adaptive modes."""
        if self.adaptive_grid is None:
            raise ValueError("AdaptiveResolutionGrid not set")

        if not self.adaptive_grid._finalized:
            raise ValueError("Grid must be finalized before visualization")

        # Get available information
        stats = self.adaptive_grid.get_statistics()
        resolutions = [r["resolution"] for r in stats["resolutions"]]
        available_signals = stats.get("available_signals", [])

        if not resolutions:
            raise ValueError("No resolution regions available")

        # Check what modes are available
        has_spatial = self.adaptive_grid.spatial_map and len(self.adaptive_grid.spatial_map.regions) > 0
        has_temporal = self.adaptive_grid.temporal_map and (
            len(self.adaptive_grid.temporal_map.time_ranges) > 0 or len(self.adaptive_grid.temporal_map.layer_ranges) > 0
        )

        # Resolution mode selector (Spatial, Temporal, Adaptive)
        mode_options = []
        if has_spatial and has_temporal:
            mode_options = [
                ("🔄 Adaptive (Spatial + Temporal)", "adaptive"),
                ("📍 Spatial Only", "spatial"),
                ("⏰ Temporal Only", "temporal"),
            ]
            default_mode = "adaptive"
        elif has_spatial:
            mode_options = [
                ("📍 Spatial Only", "spatial"),
                ("🔄 All Resolutions", "adaptive"),
            ]
            default_mode = "spatial"
        elif has_temporal:
            mode_options = [
                ("⏰ Temporal Only", "temporal"),
                ("🔄 All Resolutions", "adaptive"),
            ]
            default_mode = "temporal"
        else:
            mode_options = [("🔄 All Resolutions", "adaptive")]
            default_mode = "adaptive"

        self._widgets["resolution_mode"] = widgets.Dropdown(
            options=mode_options,
            value=default_mode,
            description="View Mode:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="350px"),
        )

        # Resolution selector (for adaptive mode)
        self._widgets["resolution_selector"] = widgets.Dropdown(
            options=[(f"{r:.3f} mm", r) for r in sorted(resolutions)],
            value=sorted(resolutions)[0],
            description="Resolution:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="300px"),
            disabled=(default_mode != "adaptive"),
        )

        # Spatial region selector (for spatial mode)
        spatial_regions = []
        if self.adaptive_grid.spatial_map and self.adaptive_grid.spatial_map.regions:
            for i, (bbox_min, bbox_max, res) in enumerate(self.adaptive_grid.spatial_map.regions):
                center = (
                    (bbox_min[0] + bbox_max[0]) / 2,
                    (bbox_min[1] + bbox_max[1]) / 2,
                    (bbox_min[2] + bbox_max[2]) / 2,
                )
                spatial_regions.append((f"📍 Region {i+1}: {res:.3f} mm", i))

        self._widgets["spatial_region_selector"] = widgets.Dropdown(
            options=(spatial_regions if spatial_regions else [("No spatial regions", None)]),
            value=spatial_regions[0][1] if spatial_regions else None,
            description="📍 Spatial Region:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="400px"),
            disabled=(default_mode != "spatial"),
        )

        # Temporal range selector (for temporal mode)
        temporal_ranges = []
        if self.adaptive_grid.temporal_map:
            for i, (t_start, t_end, res) in enumerate(self.adaptive_grid.temporal_map.time_ranges):
                duration = t_end - t_start
                temporal_ranges.append(
                    (
                        f"⏰ Time {t_start:.1f}-{t_end:.1f}s ({duration:.1f}s): {res:.3f} mm",
                        ("time", i),
                    )
                )
            for i, (l_start, l_end, res) in enumerate(self.adaptive_grid.temporal_map.layer_ranges):
                num_layers = l_end - l_start + 1
                temporal_ranges.append(
                    (
                        f"⏰ Layers {l_start}-{l_end} ({num_layers} layers): {res:.3f} mm",
                        ("layer", i),
                    )
                )

        self._widgets["temporal_range_selector"] = widgets.Dropdown(
            options=(temporal_ranges if temporal_ranges else [("No temporal ranges", None)]),
            value=temporal_ranges[0][1] if temporal_ranges else None,
            description="⏰ Temporal Range:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="400px"),
            disabled=(default_mode != "temporal"),
        )

        # Signal selector
        self._widgets["signal_selector"] = widgets.Dropdown(
            options=available_signals if available_signals else ["power"],
            value=available_signals[0] if available_signals else "power",
            description="Signal Type:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="300px"),
        )

        # Colormap selector
        colormaps = [
            "plasma",
            "viridis",
            "inferno",
            "magma",
            "coolwarm",
            "jet",
            "hot",
            "cool",
        ]
        self._widgets["colormap_selector"] = widgets.Dropdown(
            options=colormaps,
            value="plasma",
            description="Colormap:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="300px"),
        )

        # Adaptive threshold toggle
        self._widgets["adaptive_threshold"] = widgets.Checkbox(
            value=True,
            description="Adaptive threshold (auto-adjust)",
            style={"description_width": "initial"},
        )

        # Threshold slider
        self._widgets["threshold_slider"] = widgets.FloatSlider(
            value=0.1,
            min=0.0,
            max=1.0,
            step=0.01,
            description="Threshold:",
            style={"description_width": "initial"},
            disabled=True,
        )

        # View type selector
        self._widgets["view_type"] = widgets.Dropdown(
            options=[
                ("3D Volume", "3d"),
                ("Slice X", "slice_x"),
                ("Slice Y", "slice_y"),
                ("Slice Z", "slice_z"),
            ],
            value="3d",
            description="View Type:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="300px"),
        )

        # Slice position slider
        bbox_min = self.adaptive_grid.bbox_min
        bbox_max = self.adaptive_grid.bbox_max
        slice_min = min(bbox_min[0], bbox_min[1], bbox_min[2])
        slice_max = max(bbox_max[0], bbox_max[1], bbox_max[2])
        slice_center = (slice_min + slice_max) / 2.0

        self._widgets["slice_position"] = widgets.FloatSlider(
            value=slice_center,
            min=slice_min,
            max=slice_max,
            step=(slice_max - slice_min) / 100.0,
            description="Slice Position:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="300px"),
            disabled=True,
        )

        # Update button
        self._widgets["update_button"] = widgets.Button(
            description="🔄 Update Visualization",
            button_style="info",
            layout=widgets.Layout(width="200px"),
        )

        # Connect callbacks
        self._widgets["resolution_mode"].observe(self._on_adaptive_mode_change, names="value")
        self._widgets["adaptive_threshold"].observe(self._on_adaptive_threshold_change, names="value")
        self._widgets["view_type"].observe(self._on_view_type_change, names="value")
        self._widgets["update_button"].on_click(self._on_update_click)

        # Auto-update on change
        for key in [
            "resolution_selector",
            "spatial_region_selector",
            "temporal_range_selector",
            "signal_selector",
            "colormap_selector",
            "threshold_slider",
            "slice_position",
        ]:
            if key in self._widgets:
                self._widgets[key].observe(self._on_widget_change, names="value")

        # Create layout
        controls = VBox(
            [
                widgets.HTML("<h3>🎨 Adaptive Resolution Visualization</h3>"),
                widgets.HTML(
                    "<p><b>Three View Modes:</b></p>"
                    "<ul>"
                    "<li>📍 <b>Spatial Only</b>: View different resolutions in different spatial regions</li>"
                    "<li>⏰ <b>Temporal Only</b>: View different resolutions at different time points/layers</li>"
                    "<li>🔄 <b>Adaptive</b>: View combined spatial + temporal adaptive resolution</li>"
                    "</ul>"
                ),
                widgets.HTML("<h4>🎛️ View Mode Selection</h4>"),
                self._widgets["resolution_mode"],
                widgets.HTML("<hr>"),
                widgets.HTML("<h4>📍 Spatial Options (Spatial Only Mode)</h4>"),
                self._widgets["spatial_region_selector"],
                widgets.HTML("<hr>"),
                widgets.HTML("<h4>⏰ Temporal Options (Temporal Only Mode)</h4>"),
                self._widgets["temporal_range_selector"],
                widgets.HTML("<hr>"),
                widgets.HTML("<h4>🔄 Adaptive Options (All Resolutions)</h4>"),
                self._widgets["resolution_selector"],
                widgets.HTML("<hr>"),
                widgets.HTML("<h4>🎨 Visualization Settings</h4>"),
                self._widgets["signal_selector"],
                self._widgets["colormap_selector"],
                HBox(
                    [
                        self._widgets["adaptive_threshold"],
                        self._widgets["threshold_slider"],
                    ]
                ),
                self._widgets["view_type"],
                self._widgets["slice_position"],
                self._widgets["update_button"],
            ]
        )

        dashboard = VBox([HBox([controls, self.output_3d]), self.output_info])

        self._dashboard = dashboard
        return dashboard

    def _on_adaptive_mode_change(self, change):
        """Handle adaptive resolution mode change."""
        mode = change["new"]

        # Enable/disable appropriate selectors
        if mode == "spatial":
            self._widgets["resolution_selector"].disabled = True
            self._widgets["spatial_region_selector"].disabled = False
            self._widgets["temporal_range_selector"].disabled = True
        elif mode == "temporal":
            self._widgets["resolution_selector"].disabled = True
            self._widgets["spatial_region_selector"].disabled = True
            self._widgets["temporal_range_selector"].disabled = False
        else:  # adaptive
            self._widgets["resolution_selector"].disabled = False
            self._widgets["spatial_region_selector"].disabled = True
            self._widgets["temporal_range_selector"].disabled = True

        # Auto-update visualization
        self._on_widget_change(change)

    def display_dashboard(self):
        """Display the interactive dashboard."""
        if self._dashboard is None:
            self.create_widgets()
        display(self._dashboard)
        # Initial render
        self._update_visualization()

    def _on_performance_change(self, change):
        """Handle performance mode change."""
        mode = change["new"]
        # Update viewer performance mode
        if self.viewer:
            self.viewer.resolution_selector.performance_mode = mode
        self._on_widget_change(change)

    def _on_auto_level_change(self, change):
        """Handle auto-level toggle change."""
        auto = change["new"]
        self._widgets["level_selector"].disabled = auto
        self._on_widget_change(change)

    def _on_adaptive_threshold_change(self, change):
        """Handle adaptive threshold toggle change."""
        adaptive = change["new"]
        self._widgets["threshold_slider"].disabled = adaptive
        self._on_widget_change(change)

    def _on_view_type_change(self, change):
        """Handle view type change."""
        view_type = change["new"]
        self._widgets["slice_position"].disabled = view_type == "3d"
        self._on_widget_change(change)

    def _on_update_click(self, button):
        """Handle update button click."""
        self._update_visualization()

    def _on_widget_change(self, change):
        """Handle any widget change (with debouncing)."""
        current_time = time.time()
        if current_time - self._last_update_time < self._debounce_delay:
            return
        self._last_update_time = current_time
        self._update_visualization()

    def _update_visualization(self):
        """Update the visualization based on current widget values."""
        with self.output_3d:
            clear_output(wait=True)
            try:
                if self.grid_type == "adaptive":
                    self._update_adaptive_visualization()
                else:
                    self._update_multi_resolution_visualization()
            except Exception as e:
                print(f"❌ Visualization error: {e}")
                import traceback

                traceback.print_exc()

    def _update_multi_resolution_visualization(self):
        """Update visualization for multi-resolution grid."""
        # Get current values
        level = self._widgets["level_selector"].value
        auto_level = self._widgets["auto_level"].value
        signal_name = self._widgets["signal_selector"].value
        colormap = self._widgets["colormap_selector"].value
        adaptive_threshold = self._widgets["adaptive_threshold"].value
        threshold = self._widgets["threshold_slider"].value if not adaptive_threshold else None
        view_type = self._widgets["view_type"].value
        slice_position = self._widgets["slice_position"].value

        # Render visualization
        if view_type == "3d":
            plotter = self.viewer.render_3d(
                signal_name=signal_name,
                level=None if auto_level else level,
                colormap=colormap,
                auto_select_level=auto_level,
                adaptive_threshold=adaptive_threshold,
                threshold=threshold,
            )
            plotter.show()
        else:
            # Slice view
            slice_axis = view_type.split("_")[1]  # 'x', 'y', or 'z'
            plotter = self.viewer.render_slice(
                signal_name=signal_name,
                axis=slice_axis,
                position=slice_position,
                level=None if auto_level else level,
                colormap=colormap,
                adaptive_threshold=adaptive_threshold,
                threshold=threshold,
            )
            plotter.show()

        # Update info
        with self.output_info:
            clear_output(wait=True)
            level_info = self.viewer.get_level_info()
            level = self._widgets["level_selector"].value
            info = level_info.get(level, {})
            print(f"📊 Level {level} Info:")
            print(f"   Resolution: {info.get('resolution', 0):.3f} mm")
            print(f"   Filled voxels: {info.get('filled_voxels', 0):,}")
            print(f"   Total voxels: {info.get('num_voxels', 0):,}")
            print(f"   Available signals: {info.get('available_signals', [])}")

    def _update_adaptive_visualization(self):
        """Update visualization for adaptive resolution grid."""
        import pyvista as pv

        # Get current values
        mode = self._widgets["resolution_mode"].value
        signal_name = self._widgets["signal_selector"].value
        colormap = self._widgets["colormap_selector"].value
        adaptive_threshold = self._widgets["adaptive_threshold"].value
        threshold = self._widgets["threshold_slider"].value if not adaptive_threshold else None
        view_type = self._widgets["view_type"].value
        slice_position = self._widgets["slice_position"].value

        # Determine resolution based on mode
        if mode == "spatial":
            region_idx = self._widgets["spatial_region_selector"].value
            if region_idx is not None and self.adaptive_grid.spatial_map.regions:
                _, _, resolution = self.adaptive_grid.spatial_map.regions[region_idx]
            else:
                resolution = self.adaptive_grid.spatial_map.default_resolution
        elif mode == "temporal":
            range_info = self._widgets["temporal_range_selector"].value
            if range_info is not None:
                range_type, range_idx = range_info
                if range_type == "time" and self.adaptive_grid.temporal_map.time_ranges:
                    _, _, resolution = self.adaptive_grid.temporal_map.time_ranges[range_idx]
                elif range_type == "layer" and self.adaptive_grid.temporal_map.layer_ranges:
                    _, _, resolution = self.adaptive_grid.temporal_map.layer_ranges[range_idx]
                else:
                    resolution = self.adaptive_grid.temporal_map.default_resolution
            else:
                resolution = self.adaptive_grid.temporal_map.default_resolution
        else:  # adaptive
            resolution = self._widgets["resolution_selector"].value

        # Get signal array
        signal_array = self.adaptive_grid.get_signal_array(signal_name, target_resolution=resolution, default=0.0)

        # Find the grid with this resolution
        target_grid = None
        for key, grid in self.adaptive_grid.region_grids.items():
            grid_resolution = float(key.split("_")[1])
            if abs(grid_resolution - resolution) < 0.001:
                target_grid = grid
                break

        if target_grid is None:
            print(f"⚠️ No grid found for resolution {resolution:.3f} mm")
            return

        # Create PyVista grid
        pv_grid = pv.ImageData()
        pv_grid.dimensions = tuple(target_grid.dims)
        pv_grid.spacing = (target_grid.resolution,) * 3
        pv_grid.origin = tuple(target_grid.bbox_min)
        pv_grid[signal_name] = signal_array.flatten(order="F")

        # Create plotter
        plotter = pv.Plotter(notebook=True)

        # Render based on view type
        if view_type == "3d":
            # Apply threshold
            threshold_val = threshold if threshold is not None else 0.1
            threshold_mesh = pv_grid.threshold(threshold_val)

            if threshold_mesh.n_points == 0:
                plotter.add_text("No data above threshold", font_size=12)
            else:
                plotter.add_mesh(
                    threshold_mesh,
                    scalars=signal_name,
                    cmap=colormap,
                    show_edges=False,
                    show_scalar_bar=True,
                    scalar_bar_args={"title": f"{signal_name.title()} ({resolution:.3f} mm)"},
                )
        else:
            # Slice view
            slice_axis = view_type.split("_")[1]  # 'x', 'y', or 'z'
            if slice_axis == "x":
                slice_mesh = pv_grid.slice_orthogonal(x=slice_position)
            elif slice_axis == "y":
                slice_mesh = pv_grid.slice_orthogonal(y=slice_position)
            else:  # z
                slice_mesh = pv_grid.slice_orthogonal(z=slice_position)

            plotter.add_mesh(
                slice_mesh,
                scalars=signal_name,
                cmap=colormap,
                show_scalar_bar=True,
                scalar_bar_args={"title": signal_name.title()},
            )

        plotter.add_axes()

        # Add title based on mode
        if mode == "spatial":
            title = f"📍 Spatial Resolution: {resolution:.3f} mm"
        elif mode == "temporal":
            title = f"⏰ Temporal Resolution: {resolution:.3f} mm"
        else:
            title = f"🔄 Adaptive Resolution: {resolution:.3f} mm"

        plotter.add_text(title, font_size=12, position="upper_left")

        plotter.show(jupyter_backend="static")

        # Update info
        with self.output_info:
            clear_output(wait=True)
            stats = self.adaptive_grid.get_statistics()
            print(f"📊 Current View:")
            if mode == "spatial":
                print(f"   📍 Mode: SPATIAL ONLY")
            elif mode == "temporal":
                print(f"   ⏰ Mode: TEMPORAL ONLY")
            else:
                print(f"   🔄 Mode: ADAPTIVE (Spatial + Temporal)")
            print(f"   Resolution: {resolution:.3f} mm")
            print(f"   Signal: {signal_name}")

            # Find region info
            for res_info in stats["resolutions"]:
                if abs(res_info["resolution"] - resolution) < 0.001:
                    print(f"   Filled voxels: {res_info['filled_voxels']:,}")
                    print(f"   Available signals: {list(res_info['signals'])}")
                    break
