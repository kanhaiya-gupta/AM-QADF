"""
Adaptive Resolution Visualization Widgets

Interactive Jupyter notebook widgets for visualizing adaptive resolution grids
with spatially and temporally variable resolution.
"""

from typing import Optional, List, Dict, Any
import numpy as np
import time

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


class AdaptiveResolutionWidgets:
    """
    Interactive widgets for adaptive resolution grid visualization.

    Provides:
    - Resolution region selection
    - Signal type selection
    - Colormap selection
    - 3D visualization
    - Slice views
    """

    def __init__(self, adaptive_grid=None):
        """
        Initialize adaptive resolution widgets.

        Args:
            adaptive_grid: AdaptiveResolutionGrid object
        """
        if not WIDGETS_AVAILABLE:
            raise ImportError("ipywidgets is required. Install with: pip install ipywidgets")

        if not PYVISTA_AVAILABLE:
            raise ImportError("PyVista is required. Install with: pip install pyvista")

        self.adaptive_grid = adaptive_grid

        # Output widgets
        self.output_3d = Output()
        self.output_info = Output()

        # Widget state
        self._widgets = {}
        self._dashboard = None

    def create_widgets(self) -> VBox:
        """
        Create interactive widgets for adaptive resolution visualization.

        Returns:
            VBox containing all widgets
        """
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

        # Resolution mode selector (Spatial, Temporal, Adaptive)
        # Check what modes are available
        has_spatial = self.adaptive_grid.spatial_map and len(self.adaptive_grid.spatial_map.regions) > 0
        has_temporal = self.adaptive_grid.temporal_map and (
            len(self.adaptive_grid.temporal_map.time_ranges) > 0 or len(self.adaptive_grid.temporal_map.layer_ranges) > 0
        )

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

        # Create widgets
        self._widgets["resolution_selector"] = widgets.Dropdown(
            options=[(f"{r:.3f} mm", r) for r in sorted(resolutions)],
            value=sorted(resolutions)[0],
            description="Resolution:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="300px"),
        )

        # Spatial region selector (for spatial mode)
        spatial_regions = []
        if self.adaptive_grid.spatial_map and self.adaptive_grid.spatial_map.regions:
            for i, (bbox_min, bbox_max, res) in enumerate(self.adaptive_grid.spatial_map.regions):
                # Show center point and size for better understanding
                center = (
                    (bbox_min[0] + bbox_max[0]) / 2,
                    (bbox_min[1] + bbox_max[1]) / 2,
                    (bbox_min[2] + bbox_max[2]) / 2,
                )
                size = (
                    bbox_max[0] - bbox_min[0],
                    bbox_max[1] - bbox_min[1],
                    bbox_max[2] - bbox_min[2],
                )
                spatial_regions.append(
                    (
                        f"📍 Region {i+1}: {res:.3f} mm (center: [{center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}])",
                        i,
                    )
                )

        self._widgets["spatial_region_selector"] = widgets.Dropdown(
            options=(spatial_regions if spatial_regions else [("No spatial regions", None)]),
            value=spatial_regions[0][1] if spatial_regions else None,
            description="📍 Spatial Region:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="400px"),
            disabled=True,  # Disabled by default (adaptive mode)
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
            disabled=True,  # Disabled by default (adaptive mode)
        )

        self._widgets["signal_selector"] = widgets.Dropdown(
            options=available_signals if available_signals else ["power"],
            value=available_signals[0] if available_signals else "power",
            description="Signal:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="300px"),
        )

        self._widgets["colormap_selector"] = widgets.Dropdown(
            options=["plasma", "viridis", "hot", "cool", "jet", "turbo", "inferno"],
            value="plasma",
            description="Colormap:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="300px"),
        )

        self._widgets["threshold_slider"] = widgets.FloatSlider(
            value=0.1,
            min=0.0,
            max=1.0,
            step=0.01,
            description="Threshold:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="300px"),
        )

        self._widgets["view_type"] = widgets.Dropdown(
            options=["3D View", "Z Slice", "Y Slice", "X Slice"],
            value="3D View",
            description="View Type:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="300px"),
        )

        # Calculate slice position range from grid bounding box
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
            disabled=True,  # Disabled by default, enabled for slice views
        )

        self._widgets["update_button"] = widgets.Button(
            description="Update Visualization",
            button_style="info",
            layout=widgets.Layout(width="200px"),
        )

        # Connect callbacks for auto-update
        self._widgets["resolution_mode"].observe(self._on_mode_change, names="value")
        self._widgets["resolution_selector"].observe(self._on_widget_change, names="value")
        self._widgets["spatial_region_selector"].observe(self._on_widget_change, names="value")
        self._widgets["temporal_range_selector"].observe(self._on_widget_change, names="value")
        self._widgets["signal_selector"].observe(self._on_widget_change, names="value")
        self._widgets["colormap_selector"].observe(self._on_widget_change, names="value")
        self._widgets["threshold_slider"].observe(self._on_widget_change, names="value")
        self._widgets["view_type"].observe(self._on_view_type_change, names="value")
        self._widgets["slice_position"].observe(self._on_widget_change, names="value")
        self._widgets["update_button"].on_click(self._on_update)

        # Debouncing for auto-updates
        self._update_timer = None
        self._last_update_time = 0.0
        self._debounce_delay = 0.5  # 0.5 seconds

        # Create layout with better organization
        controls = VBox(
            [
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
                self._widgets["threshold_slider"],
                self._widgets["view_type"],
                self._widgets["slice_position"],
                self._widgets["update_button"],
            ]
        )

        dashboard = VBox(
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
                HBox([controls, self.output_3d]),
                self.output_info,
            ]
        )

        self._dashboard = dashboard
        return dashboard

    def _on_mode_change(self, change):
        """Handle resolution mode change."""
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

    def _on_view_type_change(self, change):
        """Handle view type change."""
        if change["new"] == "3D View":
            self._widgets["slice_position"].disabled = True
        else:
            self._widgets["slice_position"].disabled = False
        # Auto-update visualization
        self._on_widget_change(change)

    def _on_widget_change(self, change):
        """Handle widget value change with debouncing."""
        current_time = time.time()

        # Debounce: only update if enough time has passed
        if current_time - self._last_update_time > self._debounce_delay:
            self._last_update_time = current_time
            self.update_visualization()

    def _on_update(self, button):
        """Handle update button click."""
        self.update_visualization()

    def update_visualization(self):
        """Update visualization based on current widget values."""
        if self.adaptive_grid is None:
            return

        mode = self._widgets["resolution_mode"].value

        # Determine resolution based on mode
        if mode == "spatial":
            # Use spatial region resolution
            region_idx = self._widgets["spatial_region_selector"].value
            if region_idx is not None and self.adaptive_grid.spatial_map.regions:
                _, _, resolution = self.adaptive_grid.spatial_map.regions[region_idx]
            else:
                resolution = self.adaptive_grid.spatial_map.default_resolution
        elif mode == "temporal":
            # Use temporal range resolution
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
            # Use resolution selector (all regions)
            resolution = self._widgets["resolution_selector"].value

        signal_name = self._widgets["signal_selector"].value
        colormap = self._widgets["colormap_selector"].value
        threshold = self._widgets["threshold_slider"].value
        view_type = self._widgets["view_type"].value
        slice_pos = self._widgets["slice_position"].value

        with self.output_3d:
            clear_output(wait=True)
            try:
                if view_type == "3D View":
                    self._render_3d(resolution, signal_name, colormap, threshold)
                elif view_type == "Z Slice":
                    self._render_slice(resolution, signal_name, "z", slice_pos, colormap)
                elif view_type == "Y Slice":
                    self._render_slice(resolution, signal_name, "y", slice_pos, colormap)
                elif view_type == "X Slice":
                    self._render_slice(resolution, signal_name, "x", slice_pos, colormap)
            except Exception as e:
                print(f"❌ Visualization error: {e}")
                import traceback

                traceback.print_exc()

        # Update info
        with self.output_info:
            clear_output(wait=True)
            self._display_info(resolution, signal_name)

    def _render_3d(self, resolution: float, signal_name: str, colormap: str, threshold: float):
        """Render 3D visualization."""
        # Skip rendering in CI/headless environments to avoid segmentation faults
        import os

        if (
            os.environ.get("CI") == "true"
            or os.environ.get("GITHUB_ACTIONS") == "true"
            or os.environ.get("NUMBA_DISABLE_JIT") == "1"
        ):
            print(f"⚠️ Skipping 3D rendering in CI environment (resolution: {resolution:.3f} mm, signal: {signal_name})")
            return

        try:
            # Get signal array at target resolution
            signal_array = self.adaptive_grid.get_signal_array(signal_name, target_resolution=resolution, default=0.0)

            # Find the grid with this resolution
            stats = self.adaptive_grid.get_statistics()
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

            # Apply threshold
            threshold_mesh = pv_grid.threshold(threshold)

            if threshold_mesh.n_points == 0:
                plotter.add_text("No data above threshold", font_size=12)
            else:
                plotter.add_mesh(
                    threshold_mesh,
                    scalars=signal_name,
                    cmap=colormap,  # type: ignore[arg-type]
                    show_edges=False,
                    show_scalar_bar=True,
                    scalar_bar_args={"title": f"{signal_name.title()} ({resolution:.3f} mm)"},
                )

            plotter.add_axes()  # type: ignore[call-arg]

            # Add title based on mode
            mode = self._widgets["resolution_mode"].value
            if mode == "spatial":
                title = f"📍 Spatial Resolution: {resolution:.3f} mm"
            elif mode == "temporal":
                title = f"⏰ Temporal Resolution: {resolution:.3f} mm"
            else:
                title = f"🔄 Adaptive Resolution: {resolution:.3f} mm"

            plotter.add_text(title, font_size=12, position="upper_left")

            plotter.show(jupyter_backend="static")

        except Exception as e:
            print(f"❌ Error rendering 3D: {e}")
            import traceback

            traceback.print_exc()

    def _render_slice(
        self,
        resolution: float,
        signal_name: str,
        axis: str,
        position: float,
        colormap: str,
    ):
        """Render slice view."""
        # Skip rendering in CI/headless environments to avoid segmentation faults
        import os

        if (
            os.environ.get("CI") == "true"
            or os.environ.get("GITHUB_ACTIONS") == "true"
            or os.environ.get("NUMBA_DISABLE_JIT") == "1"
        ):
            print(
                f"⚠️ Skipping slice rendering in CI environment (resolution: {resolution:.3f} mm, signal: {signal_name}, axis: {axis})"
            )
            return

        try:
            # Get signal array
            signal_array = self.adaptive_grid.get_signal_array(signal_name, target_resolution=resolution, default=0.0)

            # Find the grid
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

            # Extract slice
            if axis == "x":
                slice_mesh = pv_grid.slice_orthogonal(x=position)
            elif axis == "y":
                slice_mesh = pv_grid.slice_orthogonal(y=position)
            else:  # z
                slice_mesh = pv_grid.slice_orthogonal(z=position)

            plotter.add_mesh(
                slice_mesh,
                scalars=signal_name,
                cmap=colormap,  # type: ignore[arg-type]
                show_scalar_bar=True,
                scalar_bar_args={"title": signal_name.title()},
            )

            plotter.add_axes()  # type: ignore[call-arg]
            plotter.add_text(
                f"{axis.upper()} Slice at {position:.1f} mm (Res: {resolution:.3f} mm)",
                font_size=12,
            )

            plotter.show(jupyter_backend="static")

        except Exception as e:
            print(f"❌ Error rendering slice: {e}")
            import traceback

            traceback.print_exc()

    def _display_info(self, resolution: float, signal_name: str):
        """Display information about the selected resolution region."""
        stats = self.adaptive_grid.get_statistics()
        mode = self._widgets["resolution_mode"].value

        print(f"📊 Current View:")
        if mode == "spatial":
            print(f"   📍 Mode: SPATIAL ONLY (viewing spatial regions)")
        elif mode == "temporal":
            print(f"   ⏰ Mode: TEMPORAL ONLY (viewing temporal ranges)")
        else:
            print(f"   🔄 Mode: ADAPTIVE (Spatial + Temporal combined)")

        print(f"   Resolution: {resolution:.3f} mm")
        print(f"   Signal: {signal_name}")

        # Find region info
        region_info = None
        for r_info in stats["resolutions"]:
            if abs(r_info["resolution"] - resolution) < 0.001:
                region_info = r_info
                break

        if region_info:
            print(f"\n   Grid Statistics:")
            print(f"      Filled voxels: {region_info['filled_voxels']:,}")
            print(f"      Available signals: {list(region_info['signals'])}")

        # Show mode-specific information
        if mode == "spatial":
            region_idx = self._widgets["spatial_region_selector"].value
            if region_idx is not None and self.adaptive_grid.spatial_map.regions:
                bbox_min, bbox_max, res = self.adaptive_grid.spatial_map.regions[region_idx]
                print(f"\n   📍 Spatial Region {region_idx + 1}:")
                print(
                    f"      Bounding box: [{bbox_min[0]:.2f}, {bbox_min[1]:.2f}, {bbox_min[2]:.2f}] to [{bbox_max[0]:.2f}, {bbox_max[1]:.2f}, {bbox_max[2]:.2f}]"
                )
                print(f"      Resolution: {res:.3f} mm")
            else:
                print(f"\n   Using default spatial resolution: {self.adaptive_grid.spatial_map.default_resolution:.3f} mm")
        elif mode == "temporal":
            range_info = self._widgets["temporal_range_selector"].value
            if range_info is not None:
                range_type, range_idx = range_info
                if range_type == "time" and self.adaptive_grid.temporal_map.time_ranges:
                    t_start, t_end, res = self.adaptive_grid.temporal_map.time_ranges[range_idx]
                    print(f"\n   ⏰ Time Range {range_idx + 1}:")
                    print(f"      Time: {t_start:.1f} - {t_end:.1f} seconds")
                    print(f"      Resolution: {res:.3f} mm")
                elif range_type == "layer" and self.adaptive_grid.temporal_map.layer_ranges:
                    l_start, l_end, res = self.adaptive_grid.temporal_map.layer_ranges[range_idx]
                    print(f"\n   ⏰ Layer Range {range_idx + 1}:")
                    print(f"      Layers: {l_start} - {l_end}")
                    print(f"      Resolution: {res:.3f} mm")
            else:
                print(f"\n   Using default temporal resolution: {self.adaptive_grid.temporal_map.default_resolution:.3f} mm")

        print(f"\n   Overall Grid Statistics:")
        print(f"      Total resolution regions: {stats['num_regions']}")
        print(f"      Total points: {stats['total_points']:,}")
        print(f"      All available signals: {stats['available_signals']}")

    def display_dashboard(self):
        """Display the interactive dashboard."""
        if self._dashboard is None:
            self.create_widgets()

        display(self._dashboard)

        # Initial visualization
        self.update_visualization()

    def set_adaptive_grid(self, adaptive_grid):
        """Set the adaptive resolution grid."""
        self.adaptive_grid = adaptive_grid
        self._dashboard = None  # Reset dashboard
