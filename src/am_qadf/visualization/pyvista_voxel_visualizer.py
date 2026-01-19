"""
PyVista Voxel Visualizer

Core PyVista-based visualization for voxel grids with STL support.
This module provides reusable visualization logic that can be used by:
- Web clients (via wrappers)
- Jupyter notebooks
- CLI tools
- Other modules

Returns PyVista Plotter objects for maximum flexibility.
"""

import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

try:
    import pyvista as pv
    import numpy as np
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    pv = None
    np = None

logger = logging.getLogger(__name__)


class PyVistaVoxelVisualizer:
    """
    Core PyVista visualization for voxel grids with STL support.
    
    This class provides business logic for creating PyVista visualizations.
    It returns Plotter objects, allowing consumers to export as needed.
    """
    
    def __init__(self, stl_client=None):
        """
        Initialize PyVista voxel visualizer.
        
        Args:
            stl_client: Optional STL client for loading STL files.
                       Can be STLModelClient or any object with load_stl_file(model_id) method.
        """
        if not PYVISTA_AVAILABLE:
            logger.warning("PyVista not available - visualization features will be limited")
        
        # Set PyVista to use static backend for headless rendering
        if PYVISTA_AVAILABLE:
            try:
                pv.set_jupyter_backend('static')
            except Exception:
                pass  # Backend setting is optional
        
        self.stl_client = stl_client
        self._current_voxelized_mesh = None
    
    def create_voxel_grid_plotter(
        self,
        grid_data: Dict[str, Any],
        signal_name: Optional[str] = None,
        colormap: str = "plasma",
        threshold: float = 0.0,
        opacity: float = 0.8,
        show_edges: bool = True,
        show_axes: bool = True,
        use_stl_voxelization: bool = True,
        show_grid_structure: bool = True,
        show_geometry: bool = True,
        show_stl_wireframe: bool = False,
    ) -> pv.Plotter:
        """
        Create PyVista plotter for voxel grid visualization.
        
        Args:
            grid_data: Grid data dictionary with metadata and optionally signal arrays
            signal_name: Optional signal name to visualize
            colormap: PyVista colormap name
            threshold: Minimum value to show
            opacity: Opacity (0.0 to 1.0)
            show_edges: Whether to show voxel edges
            show_axes: Whether to show coordinate axes
            use_stl_voxelization: Use STL file for accurate voxelization
            show_grid_structure: Show grid structure visualization
            show_geometry: Show voxelized geometry
            show_stl_wireframe: Show STL wireframe overlay
            
        Returns:
            PyVista Plotter object (caller can export as needed)
        """
        if not PYVISTA_AVAILABLE:
            raise ImportError("PyVista is required for visualization")
        
        try:
            metadata = grid_data.get("metadata", {})
            model_id = grid_data.get("model_id") or metadata.get("model_id")
            bbox_min = np.array(metadata.get("bbox_min", grid_data.get("bbox_min", [0, 0, 0])))
            bbox_max = np.array(metadata.get("bbox_max", grid_data.get("bbox_max", [100, 100, 100])))
            dimensions = metadata.get("dimensions", metadata.get("dims", [10, 10, 10]))
            voxel_size = metadata.get("voxel_size", 
                                      metadata.get("voxelSize",
                                      ((bbox_max - bbox_min) / np.array(dimensions)).min()))
            
            # Try to load and voxelize STL file for more accurate visualization
            voxelized_mesh = None
            if use_stl_voxelization and model_id and self.stl_client:
                try:
                    stl_path = self.stl_client.load_stl_file(model_id)
                    if stl_path and stl_path.exists():
                        logger.info(f"Loading STL file for voxelization: {stl_path}")
                        # Load STL mesh
                        stl_mesh = pv.read(str(stl_path))
                        
                        # Voxelize using PyVista's voxelize method
                        # PyVista supports both uniform spacing (float) and non-uniform spacing (tuple)
                        # spacing can be: float (uniform) or tuple (x, y, z) for anisotropic voxelization
                        if voxel_size > 0:
                            # Check if we have non-uniform spacing info in metadata
                            spacing_x = metadata.get("voxel_size_x", voxel_size)
                            spacing_y = metadata.get("voxel_size_y", voxel_size)
                            spacing_z = metadata.get("voxel_size_z", voxel_size)
                            
                            # Use non-uniform spacing if different values are provided
                            if spacing_x != spacing_y or spacing_y != spacing_z:
                                spacing_tuple = (spacing_x, spacing_y, spacing_z)
                                logger.info(f"Using non-uniform spacing: {spacing_tuple}")
                                voxelized_mesh = stl_mesh.voxelize(spacing=spacing_tuple)
                            else:
                                # Uniform spacing
                                voxelized_mesh = stl_mesh.voxelize(spacing=voxel_size)
                        else:
                            # Use default voxelization
                            voxelized_mesh = stl_mesh.voxelize()
                        
                        logger.info(f"STL voxelized successfully: {voxelized_mesh.n_cells} cells")
                except Exception as e:
                    logger.warning(f"Failed to load/voxelize STL file: {e}. Using grid-based visualization.")
                    voxelized_mesh = None
            
            # Create PyVista ImageData grid (fallback or for signal visualization)
            grid = pv.ImageData()
            grid.dimensions = tuple(dimensions)
            grid.spacing = (voxel_size, voxel_size, voxel_size)
            grid.origin = tuple(bbox_min)
            
            # Always add a default scalar array for grid visualization (even without signals)
            # This allows us to visualize the grid structure
            total_voxels = np.prod(dimensions)
            grid['grid_structure'] = np.ones(total_voxels, dtype=np.float32)
            
            # Add signal data if available
            signal_arrays = grid_data.get("signal_arrays", {})
            if signal_name and signal_name in signal_arrays:
                signal_data = np.array(signal_arrays[signal_name])
                if signal_data.size == total_voxels:
                    # Flatten in Fortran order (PyVista convention)
                    grid[signal_name] = signal_data.flatten(order='F')
                else:
                    logger.warning(f"Signal array size mismatch: expected {total_voxels}, got {signal_data.size}")
                    signal_name = None
            elif signal_arrays:
                # Use first available signal
                signal_name = list(signal_arrays.keys())[0]
                signal_data = np.array(signal_arrays[signal_name])
                if signal_data.size == total_voxels:
                    grid[signal_name] = signal_data.flatten(order='F')
                else:
                    signal_name = None
            
            # Create plotter with light background for better visibility
            plotter = pv.Plotter(off_screen=True)  # Headless rendering
            plotter.background_color = 'white'  # Light background instead of dark
            
            # Store voxelized mesh for potential use later
            self._current_voxelized_mesh = voxelized_mesh
            
            # Determine visualization mode based on user preferences
            # User can toggle between grid structure, geometry, and wireframe
            
            # 1. Show grid structure (with or without signals)
            if show_grid_structure:
                # Check if we have a signal to display
                # If signal_name is provided, use it; otherwise, try to find one
                actual_signal_name = None
                if signal_name and signal_name in grid.array_names:
                    actual_signal_name = signal_name
                elif signal_name is None and grid.array_names:
                    # No signal specified but grid has signals - use first available (except grid_structure)
                    available_signals = [name for name in grid.array_names if name != 'grid_structure']
                    if available_signals:
                        actual_signal_name = available_signals[0]
                        logger.info(f"Using first available signal for grid visualization: {actual_signal_name}")
                
                if actual_signal_name:
                    # Visualize grid structure with signal coloring (BEST for signal mapping)
                    if threshold > 0:
                        mesh = grid.threshold(threshold, scalars=actual_signal_name)
                    else:
                        mesh = grid
                    
                    plotter.add_mesh(
                        mesh,
                        scalars=actual_signal_name,
                        cmap=colormap,
                        opacity=opacity,
                        show_edges=show_edges,
                        show_scalar_bar=True,
                        scalar_bar_args={"title": actual_signal_name.title()},
                    )
                else:
                    # Show grid structure without signals (just the grid)
                    # Use the grid_structure array we already created
                    # For better visualization, we can create a pattern or use distance from center
                    try:
                        # Create a more interesting visualization using distance from center
                        center = (bbox_min + bbox_max) / 2
                        x_coords = np.linspace(bbox_min[0], bbox_max[0], dimensions[0])
                        y_coords = np.linspace(bbox_min[1], bbox_max[1], dimensions[1])
                        z_coords = np.linspace(bbox_min[2], bbox_max[2], dimensions[2])
                        
                        # Create coordinate grids
                        X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
                        
                        # Calculate distance from center for each voxel
                        distances = np.sqrt(
                            (X - center[0])**2 + 
                            (Y - center[1])**2 + 
                            (Z - center[2])**2
                        )
                        
                        # Normalize distances for better visualization
                        max_dist = np.max(distances)
                        if max_dist > 0:
                            normalized_distances = distances / max_dist
                        else:
                            normalized_distances = np.ones_like(distances)
                        
                        # Flatten in Fortran order (PyVista convention)
                        grid['grid_structure'] = normalized_distances.flatten(order='F')
                    except Exception as e:
                        logger.warning(f"Failed to create distance-based grid visualization: {e}")
                        # Fallback to uniform values
                        grid['grid_structure'] = np.ones(total_voxels, dtype=np.float32)
                    
                    plotter.add_mesh(
                        grid,
                        scalars='grid_structure',
                        cmap='viridis',
                        opacity=opacity,
                        show_edges=show_edges,
                        show_scalar_bar=False,  # No need for scalar bar for grid structure
                    )
                    
                    # Add grid outline for better visibility
                    try:
                        outline = grid.outline()
                        plotter.add_mesh(
                            outline,
                            color='gray',
                            style='wireframe',
                            line_width=2,
                            opacity=0.5
                        )
                    except Exception:
                        pass  # Skip outline if it fails
            
            # 2. Show voxelized geometry (if enabled)
            if show_geometry and voxelized_mesh is not None:
                # Add voxelized STL mesh with better colors for visibility
                plotter.add_mesh(
                    voxelized_mesh,
                    color='#4A90E2',  # Bright blue
                    opacity=opacity if not (show_grid_structure and signal_name) else 0.3,  # Lower opacity if grid structure is also shown
                    show_edges=show_edges,
                    edge_color='darkblue' if show_edges else None,
                    line_width=1 if show_edges else 0
                )
            
            # 3. Show STL wireframe overlay (if enabled)
            if show_stl_wireframe and model_id and self.stl_client:
                try:
                    stl_path = self.stl_client.load_stl_file(model_id)
                    if stl_path and stl_path.exists():
                        stl_mesh = pv.read(str(stl_path))
                        plotter.add_mesh(
                            stl_mesh,
                            style='wireframe',
                            color='red',
                            opacity=0.2,
                            line_width=1
                        )
                except Exception:
                    pass  # Skip wireframe if it fails
            
            # 4. Fallback: if nothing is enabled, show at least the grid structure
            if not show_grid_structure and not show_geometry and not show_stl_wireframe:
                # Show basic grid structure
                if threshold > 0:
                    # Create a dummy signal for thresholding
                    grid['dummy'] = np.ones(np.prod(dimensions))
                    mesh = grid.threshold(threshold, scalars='dummy')
                else:
                    mesh = grid
                
                plotter.add_mesh(
                    mesh,
                    color='#4A90E2',  # Bright blue for better visibility
                    opacity=opacity,
                    show_edges=show_edges,
                    edge_color='darkblue' if show_edges else None,
                )
                
                # Add grid outline for reference
                outline = grid.outline()
                plotter.add_mesh(
                    outline,
                    color='gray',
                    style='wireframe',
                    line_width=2,
                    opacity=0.5
                )
            
            # Add axes
            if show_axes:
                plotter.add_axes()
            
            # Set camera to fit grid
            plotter.camera_position = 'iso'
            plotter.reset_camera()
            
            return plotter
                
        except Exception as e:
            logger.error(f"Failed to create PyVista visualization: {e}", exc_info=True)
            raise
    
    def create_voxelized_mesh_plotter(
        self,
        stl_mesh_path: Optional[str] = None,
        stl_mesh_data: Optional[Any] = None,
        spacing: Optional[float] = None,
        dimensions: Optional[Tuple[int, int, int]] = None,
        show_wireframe: bool = True,
        wireframe_opacity: float = 0.3,
    ) -> pv.Plotter:
        """
        Create PyVista plotter from STL voxelization.
        
        This uses PyVista's voxelize method to create a voxelized representation of a mesh.
        
        Args:
            stl_mesh_path: Path to STL file
            stl_mesh_data: PyVista mesh object (alternative to path)
            spacing: Voxel spacing (if None, uses cell_length_percentile)
            dimensions: Grid dimensions (if None, calculated from spacing)
            show_wireframe: Whether to show original mesh wireframe overlay
            wireframe_opacity: Opacity of wireframe overlay
            
        Returns:
            PyVista Plotter object
        """
        if not PYVISTA_AVAILABLE:
            raise ImportError("PyVista is required for visualization")
        
        try:
            # Load mesh
            if stl_mesh_data is not None:
                mesh = stl_mesh_data
            elif stl_mesh_path:
                mesh = pv.read(stl_mesh_path)
            else:
                raise ValueError("Either stl_mesh_path or stl_mesh_data must be provided")
            
            # Voxelize the mesh using PyVista's voxelize method
            if spacing is not None:
                voxelized = mesh.voxelize(spacing=spacing)
            elif dimensions is not None:
                voxelized = mesh.voxelize(dimensions=dimensions)
            else:
                # Use default (10th percentile)
                voxelized = mesh.voxelize()
            
            # Create plotter
            plotter = pv.Plotter(off_screen=True)
            
            # Add voxelized mesh
            plotter.add_mesh(
                voxelized,
                show_edges=True,
                color='lightblue',
                opacity=0.8
            )
            
            # Add original mesh outline (optional)
            if show_wireframe:
                plotter.add_mesh(
                    mesh,
                    style='wireframe',
                    color='red',
                    opacity=wireframe_opacity,
                    line_width=2
                )
            
            plotter.add_axes()
            plotter.camera_position = 'iso'
            plotter.reset_camera()
            
            return plotter
                
        except Exception as e:
            logger.error(f"Failed to create voxelized mesh visualization: {e}", exc_info=True)
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
    
    def get_voxelized_mesh(self) -> Optional[pv.PolyData]:
        """
        Get the current voxelized mesh if available.
        
        Returns:
            PyVista PolyData mesh or None
        """
        return self._current_voxelized_mesh
