"""
PyVista Surface Visualizer

PyVista-based visualization for surface geometry, hatching paths, and sensor data.
No voxelization here—volume voxelization is done in ParaView.

This module provides:
- Surface of 3D geometry (STL mesh, wireframe)
- Hatching vector paths with signals (laser power, scan speed, etc.) via C++ only
- 3D plots of sensor data and associated signals

Hatching: web path uses get_hatching_visualization_data_native() and HTML exporter
create_threejs_viewer_from_hatching_arrays(); no PyVista mesh built from C++ arrays.

Used by web clients (wrappers) and CLI tools.
Returns PyVista Plotter objects for maximum flexibility.
"""

import logging
from typing import Dict, Any, Optional, Tuple, Union

try:
    import pyvista as pv
    import numpy as np
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    pv = None
    np = None

logger = logging.getLogger(__name__)


def get_hatching_visualization_data_native(
    model_id: str,
    layer_start: int,
    layer_end: int,
    scalar_name: str,
    uri: str,
    db_name: str,
) -> Tuple[Any, Any, str, Any, Any, Any, Any]:
    """
    Get ready-to-visualize hatching buffers from C++ (no Python fallback).

    Calls am_qadf_native.visualization.get_hatching_visualization_data.
    C++ computes per-vertex RGB (contour=grey, hatch path=pink, hatch signal=heatmap).

    Returns:
        (positions, scalars, active_scalar_name, segment_types, vertex_colors_rgb,
         scalar_bar_min, scalar_bar_max). Older C++ may return fewer; missing are None.
    """
    try:
        import am_qadf_native
        vis = getattr(am_qadf_native, "visualization", None)
        if vis is None:
            raise ImportError("am_qadf_native.visualization not available")
        get_data = getattr(vis, "get_hatching_visualization_data", None)
        if get_data is None:
            raise ImportError("am_qadf_native.visualization.get_hatching_visualization_data not available")
    except ImportError as e:
        raise ImportError(
            "Hatching visualization requires the C++ extension (am_qadf_native.visualization). "
            "Build and install the native module; there is no Python fallback."
        ) from e

    result = get_data(model_id, layer_start, layer_end, scalar_name, uri, db_name)
    n = len(result)
    positions = result[0]
    scalars = result[1]
    active_scalar_name = result[2]
    segment_types = result[3] if n > 3 else None
    vertex_colors_rgb = result[4] if n > 4 else None
    scalar_bar_min = result[5] if n > 5 else None
    scalar_bar_max = result[6] if n > 6 else None
    return positions, scalars, active_scalar_name, segment_types, vertex_colors_rgb, scalar_bar_min, scalar_bar_max


def get_point_cloud_visualization_data_native(
    model_id: str,
    layer_start: int,
    layer_end: int,
    source: str,
    scalar_name: str,
    uri: str,
    db_name: str,
) -> Tuple[Any, Any, str, Any, Any, Any]:
    """
    Get ready-to-visualize point cloud buffers from C++ (laser_monitoring or ISPM).

    source: "laser_monitoring" or "ispm_thermal" | "ispm_optical" | "ispm_acoustic" | "ispm_strain" | "ispm_plume".
    Returns: (positions, scalars, active_scalar_name, vertex_colors_rgb, scalar_bar_min, scalar_bar_max).
    """
    try:
        import am_qadf_native
        vis = getattr(am_qadf_native, "visualization", None)
        if vis is None:
            raise ImportError("am_qadf_native.visualization not available")
        if source == "laser_monitoring":
            get_data = getattr(vis, "get_laser_monitoring_visualization_data", None)
            if get_data is None:
                raise ImportError("am_qadf_native.visualization.get_laser_monitoring_visualization_data not available")
            result = get_data(model_id, layer_start, layer_end, scalar_name, uri, db_name)
        else:
            get_data = getattr(vis, "get_ispm_visualization_data", None)
            if get_data is None:
                raise ImportError("am_qadf_native.visualization.get_ispm_visualization_data not available")
            result = get_data(model_id, layer_start, layer_end, source, scalar_name, uri, db_name)
    except ImportError as e:
        raise ImportError(
            "Point cloud visualization requires the C++ extension (am_qadf_native.visualization). "
            "Build and install the native module; there is no Python fallback."
        ) from e

    n = len(result)
    positions = result[0]
    scalars = result[1]
    active_scalar_name = result[2]
    vertex_colors_rgb = result[3] if n > 3 else None
    scalar_bar_min = result[4] if n > 4 else None
    scalar_bar_max = result[5] if n > 5 else None
    return positions, scalars, active_scalar_name, vertex_colors_rgb, scalar_bar_min, scalar_bar_max


class PyVistaSurfaceVisualizer:
    """
    PyVista visualization for surface geometry, hatching paths, and sensor data.
    No voxelization; use ParaView for that. Use this class for:
    - 3D geometry surface (STL mesh / wireframe)
    - Hatching vector paths with signals (laser power, scan speed)
    - 3D plots of sensor data and signals

    Returns Plotter objects for export or display.
    """

    def __init__(self, stl_client=None):
        """
        Initialize PyVista surface visualizer.

        Args:
            stl_client: Optional STL client for loading STL files.
                Can be STLModelClient or any object with load_stl_file(model_id).
        """
        if not PYVISTA_AVAILABLE:
            logger.warning("PyVista not available - visualization features will be limited")

        if PYVISTA_AVAILABLE:
            try:
                pv.set_jupyter_backend("static")
            except Exception:
                pass

        self.stl_client = stl_client
        self._current_mesh = None  # set by create_stl_surface_plotter for HTML exporter fallback

    def create_stl_surface_plotter(
        self,
        stl_mesh_path: Optional[str] = None,
        stl_mesh_data: Optional[Any] = None,
        show_wireframe: bool = False,
        wireframe_opacity: float = 0.3,
        color: str = "lightblue",
        opacity: float = 0.9,
    ) -> Any:
        """
        Create PyVista plotter with STL mesh as surface only (no voxelization).

        Use this for data-query STL visualization. Volume voxelization is done in ParaView.

        Args:
            stl_mesh_path: Path to STL file
            stl_mesh_data: PyVista mesh object (alternative to path)
            show_wireframe: Whether to show wireframe overlay
            wireframe_opacity: Opacity of wireframe
            color: Surface color
            opacity: Surface opacity

        Returns:
            PyVista Plotter with surface mesh added.
        """
        if not PYVISTA_AVAILABLE:
            raise ImportError("PyVista is required for visualization")
        try:
            if stl_mesh_data is not None:
                mesh = stl_mesh_data
            elif stl_mesh_path:
                mesh = pv.read(stl_mesh_path)
            else:
                raise ValueError("Either stl_mesh_path or stl_mesh_data must be provided")
            self._current_mesh = mesh  # for HTML exporter fallback (uses points/faces)
            plotter = pv.Plotter(off_screen=True)
            plotter.add_mesh(mesh, color=color, opacity=opacity, show_edges=show_wireframe)
            if show_wireframe:
                plotter.add_mesh(
                    mesh,
                    style="wireframe",
                    color="gray",
                    opacity=wireframe_opacity,
                    line_width=1,
                )
            plotter.add_axes()
            plotter.camera_position = "iso"
            plotter.reset_camera()
            return plotter
        except Exception as e:
            logger.error(f"Failed to create STL surface visualization: {e}", exc_info=True)
            raise

    def export_plotter_to_image(
        self,
        plotter: pv.Plotter,
        width: int = 1200,
        height: int = 800
    ) -> str:
        """
        Export plotter to base64-encoded PNG image.
        
        Args:
            plotter: PyVista Plotter object
            width: Image width in pixels
            height: Image height in pixels
            
        Returns:
            Base64-encoded image data as string
        """
        import base64
        import io
        
        plotter.window_size = (width, height)
        screenshot = plotter.screenshot()
        
        # Convert to base64
        buffer = io.BytesIO()
        try:
            from PIL import Image
            img = Image.fromarray(screenshot)
            img.save(buffer, format='PNG')
            image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        except ImportError:
            # Fallback: use imageio if PIL not available
            import imageio
            buffer = io.BytesIO()
            imageio.imwrite(buffer, screenshot, format='PNG')
            image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        plotter.close()
        return image_data

    def get_current_mesh(self) -> Optional[Any]:
        """
        Get the current mesh if available (from create_stl_surface_plotter).
        Used by HTML exporter for Three.js fallback.

        Returns:
            PyVista PolyData mesh or None
        """
        return self._current_mesh

