"""
Notebook Widgets - ParaView Integration

Interactive Jupyter notebook widgets for voxel visualization with ParaView.
Provides controls for exporting to ParaView and launching ParaView with .vdb files.

PRIMARY VISUALIZATION METHOD: ParaView with .vdb files.
All visualization is done in ParaView - no Python-side rendering needed.
"""

from typing import Optional, List, Dict, Any, Callable
from pathlib import Path
import logging

try:
    import ipywidgets as widgets
    from ipywidgets import HBox, VBox, Output, interactive
    from IPython.display import display, clear_output, HTML

    WIDGETS_AVAILABLE = True
except ImportError:
    WIDGETS_AVAILABLE = False
    widgets = None
    HBox = VBox = Output = None
    display = clear_output = HTML = None

logger = logging.getLogger(__name__)


class VoxelVisualizationWidgets:
    """
    Interactive widgets for voxel domain visualization with ParaView.

    Provides:
    - Signal type selection
    - Export controls (file name, signals to export)
    - ParaView launch button
    - Export status and file information
    - Component selection (for multi-component builds)
    - Layer range selection (for filtering)
    """

    def __init__(
        self,
        query_client=None,
        voxel_grid=None,
        output_dir: str = "./paraview_exports",
    ):
        """
        Initialize visualization widgets.

        Args:
            query_client: Query client for data access (optional)
            voxel_grid: VoxelGrid object (required for export)
            output_dir: Directory to save exported .vdb files (default: "./paraview_exports")
        """
        if not WIDGETS_AVAILABLE:
            raise ImportError("ipywidgets is required. Install with: pip install ipywidgets")

        self.query_client = query_client
        self.voxel_grid = voxel_grid
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Import ParaView functions
        from .paraview_exporter import export_voxel_grid_to_paraview
        from .paraview_launcher import launch_paraview, create_paraview_button

        self.export_voxel_grid_to_paraview = export_voxel_grid_to_paraview
        self.launch_paraview = launch_paraview
        self.create_paraview_button = create_paraview_button

        # Output widget for status messages
        self.output_status = Output()
        self.output_info = Output()

        # Widget state
        self._widgets = {}
        self._dashboard = None
        self._last_exported_file = None

    def create_widgets(self) -> VBox:
        """
        Create and return the widget dashboard.

        Returns:
            VBox containing all widgets and outputs
        """
        # Available signals (from voxel grid)
        available_signals = []
        if self.voxel_grid:
            available_signals = list(self.voxel_grid.available_signals)
        else:
            available_signals = ["power", "velocity", "energy"]  # Default

        # Signal selector (multi-select for export)
        self._widgets["signals"] = widgets.SelectMultiple(
            options=available_signals,
            value=available_signals if available_signals else [],
            description="Signals to Export:",
            style={"description_width": "initial"},
            layout=widgets.Layout(height="150px", width="auto"),
        )

        # File name input
        self._widgets["filename"] = widgets.Text(
            value="voxel_grid",
            description="File Name:",
            style={"description_width": "initial"},
            placeholder="Enter file name (without .vdb extension)",
        )

        # Output directory display
        self._widgets["output_dir_display"] = widgets.HTML(
            value=f"<b>Output Directory:</b> {self.output_dir.resolve()}"
        )

        # Export button
        self._widgets["export_button"] = widgets.Button(
            description="📦 Export to ParaView",
            button_style="success",
            icon="download",
            tooltip="Export voxel grid to .vdb file",
        )
        self._widgets["export_button"].on_click(self._on_export_clicked)

        # Launch ParaView button (will be created after export)
        self._widgets["launch_button"] = widgets.Button(
            description="🚀 Launch ParaView",
            button_style="info",
            icon="external-link",
            tooltip="Launch ParaView with exported .vdb file",
            disabled=True,  # Disabled until export is done
        )
        self._widgets["launch_button"].on_click(self._on_launch_clicked)

        # Component selector (if multi-component)
        components = ["All Components"]  # Default
        if self.query_client and hasattr(self.query_client, "list_components"):
            try:
                components = ["All Components"] + self.query_client.list_components()
            except:
                pass

        self._widgets["component"] = widgets.Dropdown(
            options=components,
            value=components[0] if components else None,
            description="Component:",
            style={"description_width": "initial"},
            disabled=len(components) <= 1,
        )

        # Layer range (for filtering/info)
        max_layers = 100  # Default
        if self.query_client and hasattr(self.query_client, "get_layer_count"):
            try:
                max_layers = self.query_client.get_layer_count()
            except:
                pass

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

        # Grid info display
        grid_info_html = self._get_grid_info_html()
        self._widgets["grid_info"] = widgets.HTML(value=grid_info_html)

        # Create layout
        controls = VBox(
            [
                widgets.HTML("<h3>🎛️ ParaView Export Controls</h3>"),
                self._widgets["grid_info"],
                widgets.HTML("<hr>"),
                self._widgets["signals"],
                self._widgets["filename"],
                self._widgets["output_dir_display"],
                HBox([self._widgets["export_button"], self._widgets["launch_button"]]),
                widgets.HTML("<hr>"),
                widgets.HTML("<h4>Filter Options (Info Only)</h4>"),
                self._widgets["component"],
                HBox([self._widgets["layer_start"], self._widgets["layer_end"]]),
            ]
        )

        # Status and info section
        status_section = VBox(
            [
                widgets.HTML("<h3>📊 Export Status</h3>"),
                self.output_status,
                widgets.HTML("<h3>ℹ️ Information</h3>"),
                self.output_info,
            ],
            layout=widgets.Layout(width="100%", border="1px solid #ccc", padding="10px"),
        )

        self._dashboard = VBox(
            [
                widgets.HTML("<h1>🎨 ParaView Visualization Dashboard</h1>"),
                widgets.HTML(
                    "<p><b>Primary Visualization Method:</b> ParaView with .vdb files. "
                    "Export your voxel grid and launch ParaView for superior 3D visualization.</p>"
                ),
                HBox(
                    [controls, status_section],
                    layout=widgets.Layout(width="100%"),
                ),
            ]
        )

        # Initialize info display
        with self.output_info:
            clear_output()
            display(
                HTML(
                    """
                    <div style='padding: 10px; background-color: #f0f0f0; border-radius: 5px;'>
                        <h4>📖 How to Use:</h4>
                        <ol>
                            <li>Select signals to export (or leave all selected)</li>
                            <li>Enter a file name (without .vdb extension)</li>
                            <li>Click "Export to ParaView" to create .vdb file</li>
                            <li>Click "Launch ParaView" to open the file in ParaView</li>
                        </ol>
                        <p><b>Note:</b> ParaView provides superior 3D visualization, slice views, 
                        isosurfaces, and more. All visualization is done in ParaView.</p>
                    </div>
                    """
                )
            )

        return self._dashboard

    def _get_grid_info_html(self) -> str:
        """Get HTML string with grid information."""
        if not self.voxel_grid:
            return "<p><i>No voxel grid loaded</i></p>"

        try:
            bbox_min, bbox_max = self.voxel_grid.get_bounding_box()
            resolution = getattr(self.voxel_grid, "resolution", "N/A")
            dims = getattr(self.voxel_grid, "dims", None)
            available_signals = list(self.voxel_grid.available_signals)

            dims_str = f"{dims[0]} × {dims[1]} × {dims[2]}" if dims is not None else "N/A"

            return f"""
            <div style='padding: 10px; background-color: #e8f4f8; border-radius: 5px;'>
                <h4>📐 Grid Information:</h4>
                <ul>
                    <li><b>Resolution:</b> {resolution} mm</li>
                    <li><b>Dimensions:</b> {dims_str} voxels</li>
                    <li><b>Bounding Box:</b> ({bbox_min[0]:.2f}, {bbox_min[1]:.2f}, {bbox_min[2]:.2f}) to 
                        ({bbox_max[0]:.2f}, {bbox_max[1]:.2f}, {bbox_max[2]:.2f}) mm</li>
                    <li><b>Available Signals:</b> {', '.join(available_signals) if available_signals else 'None'}</li>
                </ul>
            </div>
            """
        except Exception as e:
            return f"<p><i>Error getting grid info: {e}</i></p>"

    def _on_export_clicked(self, button):
        """Handle export button click."""
        if not self.voxel_grid:
            with self.output_status:
                clear_output()
                display(HTML("<p style='color: red;'>❌ No voxel grid loaded!</p>"))
            return

        try:
            # Get selected signals
            selected_signals = list(self._widgets["signals"].value)
            if not selected_signals:
                with self.output_status:
                    clear_output()
                    display(HTML("<p style='color: red;'>❌ Please select at least one signal!</p>"))
                return

            # Get file name
            filename = self._widgets["filename"].value.strip()
            if not filename:
                filename = "voxel_grid"

            # Ensure .vdb extension
            if not filename.endswith(".vdb"):
                filename += ".vdb"

            # Create full path
            output_path = self.output_dir / filename

            # Export to ParaView
            with self.output_status:
                clear_output()
                display(HTML(f"<p>📦 Exporting {len(selected_signals)} signal(s) to ParaView...</p>"))

            vdb_path = self.export_voxel_grid_to_paraview(
                self.voxel_grid,
                str(output_path),
                signal_names=selected_signals if selected_signals else None,
            )

            self._last_exported_file = vdb_path

            # Enable launch button
            self._widgets["launch_button"].disabled = False

            # Update status
            with self.output_status:
                clear_output()
                display(
                    HTML(
                        f"""
                        <div style='padding: 10px; background-color: #d4edda; border-radius: 5px;'>
                            <p style='color: green;'>✅ <b>Export Successful!</b></p>
                            <p><b>File:</b> {vdb_path}</p>
                            <p><b>Signals:</b> {', '.join(selected_signals)}</p>
                            <p>Click "Launch ParaView" to open in ParaView.</p>
                        </div>
                        """
                    )
                )

        except Exception as e:
            with self.output_status:
                clear_output()
                display(
                    HTML(
                        f"""
                        <div style='padding: 10px; background-color: #f8d7da; border-radius: 5px;'>
                            <p style='color: red;'>❌ <b>Export Failed!</b></p>
                            <p>Error: {str(e)}</p>
                        </div>
                        """
                    )
                )
            logger.error(f"Export failed: {e}", exc_info=True)

    def _on_launch_clicked(self, button):
        """Handle launch button click."""
        if not self._last_exported_file:
            with self.output_status:
                clear_output()
                display(HTML("<p style='color: red;'>❌ No file exported yet! Please export first.</p>"))
            return

        try:
            with self.output_status:
                clear_output()
                display(HTML("<p>🚀 Launching ParaView...</p>"))

            success = self.launch_paraview(self._last_exported_file)

            if success:
                with self.output_status:
                    clear_output()
                    display(
                        HTML(
                            f"""
                            <div style='padding: 10px; background-color: #d4edda; border-radius: 5px;'>
                                <p style='color: green;'>✅ <b>ParaView Launched!</b></p>
                                <p>ParaView should open with: <b>{self._last_exported_file}</b></p>
                                <p>If ParaView doesn't open, please check if ParaView is installed.</p>
                            </div>
                            """
                        )
                    )
            else:
                with self.output_status:
                    clear_output()
                    display(
                        HTML(
                            f"""
                            <div style='padding: 10px; background-color: #fff3cd; border-radius: 5px;'>
                                <p style='color: orange;'>⚠️ <b>ParaView Launch Failed</b></p>
                                <p>Please check if ParaView is installed and in your PATH.</p>
                                <p>You can manually open: <b>{self._last_exported_file}</b></p>
                            </div>
                            """
                        )
                    )

        except Exception as e:
            with self.output_status:
                clear_output()
                display(
                    HTML(
                        f"""
                        <div style='padding: 10px; background-color: #f8d7da; border-radius: 5px;'>
                            <p style='color: red;'>❌ <b>Launch Failed!</b></p>
                            <p>Error: {str(e)}</p>
                        </div>
                        """
                    )
                )
            logger.error(f"Launch failed: {e}", exc_info=True)

    def set_voxel_grid(self, voxel_grid):
        """Update the voxel grid and refresh widgets."""
        self.voxel_grid = voxel_grid

        # Update available signals
        if voxel_grid:
            available_signals = list(voxel_grid.available_signals)
            if "signals" in self._widgets:
                self._widgets["signals"].options = available_signals
                self._widgets["signals"].value = available_signals

            # Update grid info
            if "grid_info" in self._widgets:
                self._widgets["grid_info"].value = self._get_grid_info_html()

    def display(self):
        """Display the widget dashboard."""
        try:
            if self._dashboard is None:
                self.create_widgets()
            display(self._dashboard)
            print("💡 Dashboard displayed. Use the controls to export and launch ParaView.")
        except Exception as e:
            print(f"Error displaying dashboard: {e}")
            import traceback

            traceback.print_exc()
