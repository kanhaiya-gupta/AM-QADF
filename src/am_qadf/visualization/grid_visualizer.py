"""
Grid Visualizer

Specialized visualizer for voxel grids using PyVista.
Designed to visualize:
1. Empty grid structures (before signal mapping)
2. Grid resolution effects
3. Grid layout and structure
4. Signal-mapped grids (complement to hatching_visualizer)

Works with VoxelGrid, AdaptiveResolutionGrid, and MultiResolutionGrid objects.

PyVista is the primary and only visualization backend for this module.
"""

import logging
from typing import Any, List, Tuple, Optional, Dict, Union
import numpy as np

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    raise ImportError("PyVista is required for grid visualization. Install with: pip install pyvista")

logger = logging.getLogger(__name__)


class GridVisualizer:
    """
    Visualizer for voxel grids using PyVista.
    
    PyVista is the primary and only visualization backend for this module.
    It provides superior 3D visualization with interactive capabilities.
    
    Can visualize:
    - Empty grid structures (resolution visualization)
    - Grid layout and voxel arrangement
    - Signal-mapped grids
    - Resolution effects (uniform, adaptive, multi-resolution)
    """
    
    def __init__(self):
        """Initialize the grid visualizer."""
        if not PYVISTA_AVAILABLE:
            raise ImportError("PyVista is required for grid visualization. Install with: pip install pyvista")
    
    def plot_grid_structure(self,
                           voxel_grid: Any,
                           show_edges: bool = True,
                           edge_color: str = 'black',
                           face_color: str = 'lightblue',
                           opacity: float = 0.7,
                           show_grid_outline: bool = True,
                           notebook: bool = True,
                           auto_show: bool = True) -> Optional[pv.Plotter]:
        """
        Plot empty grid structure using PyVista.
        
        Shows the grid layout, voxel arrangement, and resolution effects.
        Uses PyVista's ImageData with edges to clearly show voxel boundaries.
        Perfect for understanding how resolution affects grid structure.
        
        Args:
            voxel_grid: VoxelGrid object (empty or with data)
            show_edges: Show voxel edges (makes boundaries visible)
            edge_color: Color of voxel edges
            face_color: Color of voxel faces
            opacity: Opacity of voxels (0.0 to 1.0)
            show_grid_outline: Show bounding box outline
            notebook: If True, use notebook backend
            auto_show: If True, automatically show the plotter (default: True)
            
        Returns:
            PyVista Plotter object, or None on error
        """
        try:
            # Create PyVista ImageData grid
            grid = pv.ImageData()
            grid.dimensions = tuple(voxel_grid.dims)
            grid.spacing = (voxel_grid.resolution,) * 3
            grid.origin = tuple(voxel_grid.bbox_min)
            
            # For empty grids, we'll use volume rendering to show the grid structure
            # Create a simple gradient array for visualization
            total_voxels = np.prod(voxel_grid.dims)
            dims = voxel_grid.dims
            
            # Create a gradient based on Z position for better visualization
            data_array = np.zeros(total_voxels)
            for k in range(dims[2]):
                z_norm = k / max(dims[2] - 1, 1)  # Normalize to 0-1
                start_idx = k * dims[0] * dims[1]
                end_idx = (k + 1) * dims[0] * dims[1]
                data_array[start_idx:end_idx] = z_norm * 0.5 + 0.5  # Scale to 0.5-1.0 for better visibility
            
            grid['values'] = data_array
            
            # Create plotter
            plotter = pv.Plotter(notebook=notebook)
            
            # For empty grids, use threshold to show grid structure with colors
            # Apply a low threshold to show all voxels
            threshold_mesh = grid.threshold(0.01)
            
            if threshold_mesh.n_points > 0:
                # Use gradient with colormap
                plotter.add_mesh(
                    threshold_mesh,
                    scalars='values',
                    cmap='viridis',
                    show_edges=show_edges,
                    edge_color=edge_color,
                    opacity=opacity,
                    show_scalar_bar=False
                )
            else:
                # Fallback: use uniform color
                plotter.add_mesh(
                    grid.outline(),
                    color=face_color,
                    line_width=2,
                    opacity=opacity
                )
            
            # Add bounding box outline
            if show_grid_outline:
                outline = grid.outline()
                plotter.add_mesh(outline, color='gray', line_width=2, opacity=0.5)
            
            # Set title
            title = f'Grid Structure - Resolution: {voxel_grid.resolution:.3f} mm'
            title += f' ({voxel_grid.dims[0]}×{voxel_grid.dims[1]}×{voxel_grid.dims[2]} voxels)'
            plotter.add_text(title, font_size=12, position='upper_left')
            
            plotter.add_axes()
            
            if auto_show:
                plotter.show(jupyter_backend='static')
            
            return plotter
            
        except Exception as e:
            logger.error(f"Error creating PyVista visualization: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def plot_grid_slice(self,
                       voxel_grid: Any,
                       axis: str = 'z',
                       position: float = 0.5,
                       signal_name: Optional[str] = None,
                       colormap: str = 'viridis',
                       show_grid_lines: bool = True,
                       notebook: bool = True,
                       auto_show: bool = True) -> Optional[pv.Plotter]:
        """
        Plot a 2D slice through the grid using PyVista.
        
        Args:
            voxel_grid: VoxelGrid object
            axis: Slice axis ('x', 'y', or 'z')
            position: Position along axis (0.0 to 1.0, or absolute mm value if > 1.0)
            signal_name: Optional signal name to color by
            colormap: Colormap for visualization
            show_grid_lines: Show grid lines for empty grids
            notebook: If True, use notebook backend
            auto_show: If True, automatically show the plotter (default: True)
            
        Returns:
            PyVista Plotter object, or None on error
        """
        try:
            # Create PyVista ImageData grid
            grid = pv.ImageData()
            grid.dimensions = tuple(voxel_grid.dims)
            grid.spacing = (voxel_grid.resolution,) * 3
            grid.origin = tuple(voxel_grid.bbox_min)
            
            bbox_min = np.array(voxel_grid.bbox_min)
            bbox_max = np.array(voxel_grid.bbox_max)
            dims = np.array(voxel_grid.dims)
            resolution = voxel_grid.resolution
            total_voxels = np.prod(dims)
            
            # Determine slice position
            if position > 1.0:  # Absolute position in mm
                if axis == 'z':
                    z_abs = position
                    z_norm = (z_abs - bbox_min[2]) / (bbox_max[2] - bbox_min[2]) if bbox_max[2] > bbox_min[2] else 0.5
                elif axis == 'y':
                    y_abs = position
                    z_norm = (y_abs - bbox_min[1]) / (bbox_max[1] - bbox_min[1]) if bbox_max[1] > bbox_min[1] else 0.5
                else:  # x
                    x_abs = position
                    z_norm = (x_abs - bbox_min[0]) / (bbox_max[0] - bbox_min[0]) if bbox_max[0] > bbox_min[0] else 0.5
            else:
                z_norm = position  # Normalized position (0.0 to 1.0)
            
            # Get signal array if available
            slice_data = None
            has_signals = False
            if signal_name and hasattr(voxel_grid, 'get_signal_array'):
                try:
                    signal_array = voxel_grid.get_signal_array(signal_name, default=0.0)
                    grid[signal_name] = signal_array.flatten(order='F')
                    slice_data = signal_array
                    has_signals = True
                except Exception as e:
                    logger.warning(f"Could not get signal array: {e}")
            
            # For empty grids, create a checkerboard pattern to show voxel structure
            if not has_signals:
                # Create checkerboard pattern for entire grid (3D pattern)
                # This creates a pattern that will show clearly when sliced
                checkerboard_data = np.zeros(total_voxels)
                for k in range(dims[2]):
                    for j in range(dims[1]):
                        for i in range(dims[0]):
                            idx = i + j * dims[0] + k * dims[0] * dims[1]
                            # 3D checkerboard: alternate based on all three indices
                            checkerboard_data[idx] = (i + j + k) % 2
                
                grid['checkerboard'] = checkerboard_data
            
            # Create plotter
            plotter = pv.Plotter(notebook=notebook)
            
            # Calculate slice position in world coordinates
            # Ensure we're not slicing at the exact edge (add small offset)
            eps = 1e-6
            if axis == 'z':
                slice_pos = bbox_min[2] + z_norm * (bbox_max[2] - bbox_min[2])
                # Clip to be slightly inside bounds to avoid edge issues
                slice_pos = np.clip(slice_pos, bbox_min[2] + eps, bbox_max[2] - eps)
                slice_mesh = grid.slice(normal='z', origin=(0, 0, slice_pos))
                axis_label = 'Z'
            elif axis == 'y':
                slice_pos = bbox_min[1] + z_norm * (bbox_max[1] - bbox_min[1])
                slice_pos = np.clip(slice_pos, bbox_min[1] + eps, bbox_max[1] - eps)
                slice_mesh = grid.slice(normal='y', origin=(0, slice_pos, 0))
                axis_label = 'Y'
            else:  # x
                slice_pos = bbox_min[0] + z_norm * (bbox_max[0] - bbox_min[0])
                slice_pos = np.clip(slice_pos, bbox_min[0] + eps, bbox_max[0] - eps)
                slice_mesh = grid.slice(normal='x', origin=(slice_pos, 0, 0))
                axis_label = 'X'
            
            # Handle MultiBlock if returned (extract first block)
            if isinstance(slice_mesh, pv.MultiBlock):
                if slice_mesh.n_blocks > 0:
                    slice_mesh = slice_mesh[0]
                else:
                    logger.error(f"Empty MultiBlock slice at {axis_label}={slice_pos:.2f} mm")
                    return None
            
            # Check if slice is empty
            if slice_mesh.n_points == 0:
                logger.error(f"Empty slice mesh at {axis_label}={slice_pos:.2f} mm")
                return None
            
            # Add slice mesh
            if has_signals and signal_name:
                plotter.add_mesh(
                    slice_mesh,
                    scalars=signal_name,
                    cmap=colormap,
                    show_scalar_bar=True,
                    scalar_bar_args={'title': signal_name.replace('_', ' ').title()}
                )
            else:
                # For empty grids, use checkerboard pattern to show voxel boundaries
                plotter.add_mesh(
                    slice_mesh,
                    scalars='checkerboard',
                    cmap='RdYlBu',
                    show_scalar_bar=False,
                    style='surface',  # Surface style for better visibility
                    opacity=1.0
                )
            
            # Set title
            if axis == 'z':
                title = f'XY Slice at {axis_label}={slice_pos:.2f} mm - Resolution: {resolution:.3f} mm'
            elif axis == 'y':
                title = f'XZ Slice at {axis_label}={slice_pos:.2f} mm - Resolution: {resolution:.3f} mm'
            else:
                title = f'YZ Slice at {axis_label}={slice_pos:.2f} mm - Resolution: {resolution:.3f} mm'
            
            plotter.add_text(title, font_size=12, position='upper_left')
            plotter.add_axes()
            
            if auto_show:
                plotter.show(jupyter_backend='static')
            
            return plotter
            
        except Exception as e:
            logger.error(f"Error creating slice visualization: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def plot_resolution_comparison(self,
                                   grids: List[Tuple[str, Any]],
                                   show_edges: bool = True,
                                   edge_color: str = 'black',
                                   opacity: float = 0.7,
                                   colormap: str = 'viridis',
                                   notebook: bool = True,
                                   auto_show: bool = True) -> Optional[pv.Plotter]:
        """
        Compare multiple grids with different resolutions side by side using PyVista.
        
        Args:
            grids: List of (name, VoxelGrid) tuples
            show_edges: Show voxel edges
            edge_color: Color of voxel edges
            opacity: Opacity of voxels
            colormap: Colormap for visualization
            notebook: If True, use notebook backend
            auto_show: If True, automatically show the plotter (default: True)
            
        Returns:
            PyVista Plotter object with subplots, or None on error
        """
        n_grids = len(grids)
        if n_grids == 0:
            raise ValueError("No grids provided for comparison")
        
        try:
            # Create plotter with subplots
            plotter = pv.Plotter(shape=(1, n_grids), notebook=notebook)
            
            for idx, (name, grid) in enumerate(grids):
                plotter.subplot(0, idx)
                
                # Create PyVista ImageData grid
                pv_grid = pv.ImageData()
                pv_grid.dimensions = tuple(grid.dims)
                pv_grid.spacing = (grid.resolution,) * 3
                pv_grid.origin = tuple(grid.bbox_min)
                
                # Create gradient array
                total_voxels = np.prod(grid.dims)
                data_array = np.ones(total_voxels)
                dims = grid.dims
                for k in range(dims[2]):
                    z_norm = k / max(dims[2] - 1, 1)
                    start_idx = k * dims[0] * dims[1]
                    end_idx = (k + 1) * dims[0] * dims[1]
                    data_array[start_idx:end_idx] = z_norm
                
                pv_grid['values'] = data_array
                
                # Add mesh
                plotter.add_mesh(
                    pv_grid,
                    scalars='values',
                    cmap=colormap,
                    show_edges=show_edges,
                    edge_color=edge_color,
                    opacity=opacity,
                    show_scalar_bar=False
                )
                
                # Add outline
                outline = pv_grid.outline()
                plotter.add_mesh(outline, color='gray', line_width=2, opacity=0.5)
                
                # Set title
                title = f'{name}\nRes={grid.resolution:.3f} mm\nDims={grid.dims}'
                plotter.add_text(title, font_size=10, position='upper_left')
                plotter.add_axes()
            
            if auto_show:
                plotter.show(jupyter_backend='static')
            
            return plotter
            
        except Exception as e:
            logger.error(f"Error creating resolution comparison: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
