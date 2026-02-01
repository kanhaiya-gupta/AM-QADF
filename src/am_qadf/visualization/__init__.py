"""
AM-QADF Visualization Module

- ParaView: voxel/volume data (.vdb), slice views, isosurfaces. Primary for fine-resolution grids.
- PyVista: surface geometry (STL), hatching paths with signals, 3D sensor data. No volume voxels.
"""

from .paraview_exporter import (
    export_voxel_grid_to_paraview,
    export_multiple_grids_to_paraview,
)

from .paraview_launcher import (
    launch_paraview,
    create_paraview_button,
    export_and_launch_paraview,
    find_paraview_executable,
)

from .notebook_widgets import (
    VoxelVisualizationWidgets,
)

from .pyvista_visualizer import (
    PyVistaSurfaceVisualizer,
    get_hatching_visualization_data_native,
    get_point_cloud_visualization_data_native,
)
from .html_exporter import PyVistaHTMLExporter

__all__ = [
    "export_voxel_grid_to_paraview",
    "export_multiple_grids_to_paraview",
    "launch_paraview",
    "create_paraview_button",
    "export_and_launch_paraview",
    "find_paraview_executable",
    "VoxelVisualizationWidgets",
    "PyVistaSurfaceVisualizer",
    "get_hatching_visualization_data_native",
    "get_point_cloud_visualization_data_native",
    "PyVistaHTMLExporter",
]
