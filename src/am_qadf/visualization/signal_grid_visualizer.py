"""
Signal Grid Visualizer

Enhanced PyVista-based visualization for signal-mapped voxel grids.
Works with grids loaded from MongoDB (mapped, corrected, calibrated, fused).

This module provides easy-to-use visualization functions for notebooks.
"""

from typing import Optional, Union, Dict, Any
import numpy as np

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    pv = None


class SignalGridVisualizer:
    """
    Visualizer for signal-mapped voxel grids.
    
    Works with:
    - VoxelGrid objects (from am_qadf.voxelization.voxel_grid)
    - Loaded grids from MongoDB (with _signal_arrays attribute)
    - Demo grids (with _signal_arrays attribute)
    
    Provides methods for:
    - 3D volume rendering with signals
    - 2D slice visualization
    - Isosurface rendering
    - Multi-slice views
    """
    
    def __init__(self, voxel_grid=None):
        """
        Initialize visualizer.
        
        Args:
            voxel_grid: VoxelGrid object or grid with _signal_arrays attribute
        """
        if not PYVISTA_AVAILABLE:
            raise ImportError(
                "PyVista is required for visualization. "
                "Install with: pip install pyvista"
            )
        
        self.voxel_grid = voxel_grid
        self._pyvista_grid_cache = {}
    
    def set_voxel_grid(self, voxel_grid):
        """Set the voxel grid to visualize."""
        self.voxel_grid = voxel_grid
        self._pyvista_grid_cache = {}  # Clear cache
    
    def _get_signal_array(self, signal_name: str) -> Optional[np.ndarray]:
        """
        Get signal array from grid, handling different grid types.
        
        Args:
            signal_name: Name of signal to retrieve
            
        Returns:
            Signal array or None if not found
        """
        if self.voxel_grid is None:
            return None
        
        # Try _signal_arrays first (for loaded grids)
        if hasattr(self.voxel_grid, '_signal_arrays'):
            if signal_name in self.voxel_grid._signal_arrays:
                return self.voxel_grid._signal_arrays[signal_name]
        
        # Try get_signal_array method (for VoxelGrid objects)
        if hasattr(self.voxel_grid, 'get_signal_array'):
            try:
                return self.voxel_grid.get_signal_array(signal_name, default=0.0)
            except (KeyError, AttributeError):
                pass
        
        return None
    
    def _get_grid_metadata(self) -> Dict[str, Any]:
        """
        Extract grid metadata (dims, resolution, bbox) from grid object.
        
        Returns:
            Dictionary with dims, resolution, bbox_min, bbox_max
        """
        if self.voxel_grid is None:
            raise ValueError("No voxel grid set.")
        
        # Get dimensions
        if hasattr(self.voxel_grid, 'dims'):
            dims = self.voxel_grid.dims
        else:
            # Try to infer from signal array
            if hasattr(self.voxel_grid, '_signal_arrays') and self.voxel_grid._signal_arrays:
                first_signal = next(iter(self.voxel_grid._signal_arrays.values()))
                dims = first_signal.shape
            else:
                raise ValueError("Cannot determine grid dimensions")
        
        # Get resolution
        if hasattr(self.voxel_grid, 'resolution'):
            resolution = self.voxel_grid.resolution
        else:
            resolution = 2.0  # Default
        
        # Get bounding box
        if hasattr(self.voxel_grid, 'bbox_min') and hasattr(self.voxel_grid, 'bbox_max'):
            bbox_min = np.array(self.voxel_grid.bbox_min)
            bbox_max = np.array(self.voxel_grid.bbox_max)
        else:
            # Estimate from dims and resolution
            bbox_min = np.array([0, 0, 0])
            bbox_max = np.array(dims) * resolution
        
        return {
            'dims': dims,
            'resolution': resolution,
            'bbox_min': bbox_min,
            'bbox_max': bbox_max
        }
    
    def _create_pyvista_grid(self, signal_name: str) -> Optional[pv.ImageData]:
        """
        Create PyVista ImageData from voxel grid with signal.
        
        Args:
            signal_name: Name of signal to include in grid
            
        Returns:
            PyVista ImageData object or None
        """
        if self.voxel_grid is None:
            return None
        
        # Check cache
        cache_key = signal_name
        if cache_key in self._pyvista_grid_cache:
            return self._pyvista_grid_cache[cache_key]
        
        # Get metadata
        metadata = self._get_grid_metadata()
        
        # Create PyVista grid
        grid = pv.ImageData()
        grid.dimensions = tuple(metadata['dims'])
        grid.spacing = (metadata['resolution'],) * 3
        grid.origin = tuple(metadata['bbox_min'])
        
        # Get signal array
        signal_array = self._get_signal_array(signal_name)
        if signal_array is None:
            # Create zero array
            signal_array = np.zeros(metadata['dims'])
        
        # Ensure array matches grid dimensions
        if signal_array.shape != tuple(metadata['dims']):
            # Try to reshape
            if signal_array.size == np.prod(metadata['dims']):
                signal_array = signal_array.reshape(metadata['dims'])
            else:
                raise ValueError(
                    f"Signal array shape {signal_array.shape} doesn't match "
                    f"grid dimensions {metadata['dims']}"
                )
        
        # Flatten in Fortran order (PyVista convention)
        grid[signal_name] = signal_array.flatten(order='F')
        
        # Cache it
        self._pyvista_grid_cache[cache_key] = grid
        
        return grid
    
    def render_3d(
        self,
        signal_name: str,
        colormap: str = "plasma",
        threshold: Optional[float] = None,
        opacity: float = 1.0,
        show_edges: bool = True,
        show_scalar_bar: bool = True,
        title: Optional[str] = None,
        auto_show: bool = True,
    ) -> pv.Plotter:
        """
        Render 3D voxel visualization with signal.
        
        Args:
            signal_name: Name of signal to visualize
            colormap: PyVista colormap name (default: "plasma")
            threshold: Minimum value to show (None = auto-detect)
            opacity: Opacity of voxels (0.0 to 1.0)
            show_edges: Whether to show grid lines/edges (default: True)
            show_scalar_bar: Whether to show color bar
            title: Optional title for the plot
            auto_show: If True, automatically display (default: True)
            
        Returns:
            PyVista Plotter object
        """
        if self.voxel_grid is None:
            raise ValueError("No voxel grid set. Call set_voxel_grid() first.")
        
        # Get signal array for threshold detection
        signal_array = self._get_signal_array(signal_name)
        if signal_array is None:
            raise ValueError(f"Signal '{signal_name}' not found in grid")
        
        # Auto-detect threshold if not provided
        if threshold is None:
            non_zero = signal_array[signal_array != 0.0]
            if len(non_zero) > 0:
                threshold = float(np.percentile(non_zero, 10))  # Bottom 10%
            else:
                threshold = 0.0
        
        # Create PyVista grid
        grid = self._create_pyvista_grid(signal_name)
        
        # Handle large grids
        if signal_array.size >= 1e6:  # 1 million voxels
            import warnings
            warnings.warn(
                f"Large grid detected ({signal_array.size:,} voxels). "
                "Consider using higher threshold for better performance.",
                UserWarning
            )
            threshold = max(threshold, np.percentile(signal_array, 25))
        
        plotter = pv.Plotter(notebook=True)
        
        # Apply threshold
        threshold_mesh = grid.threshold(threshold, scalars=signal_name)
        
        if threshold_mesh.n_points == 0:
            plotter.add_text(
                f"No data above threshold ({threshold:.3f})",
                font_size=12,
                position='upper_left'
            )
        else:
            plotter.add_mesh(
                threshold_mesh,
                scalars=signal_name,
                cmap=colormap,
                opacity=opacity,
                show_edges=show_edges,
                edge_color='black',
                line_width=1,
                show_scalar_bar=show_scalar_bar,
                scalar_bar_args={"title": signal_name.replace('_', ' ').title()},
            )
        
        plotter.add_axes()
        
        if title:
            plotter.add_text(title, font_size=12, position='upper_right')
        else:
            plotter.add_text(
                f"3D View: {signal_name.replace('_', ' ').title()}",
                font_size=12,
                position='upper_right'
            )
        
        if auto_show:
            plotter.show(jupyter_backend="static")
        
        return plotter
    
    def render_slice(
        self,
        signal_name: str,
        axis: str = "z",
        position: Optional[float] = None,
        colormap: str = "plasma",
        show_edges: bool = True,
        show_scalar_bar: bool = True,
        title: Optional[str] = None,
        auto_show: bool = True,
    ) -> pv.Plotter:
        """
        Render 2D slice through voxel grid.
        
        Args:
            signal_name: Name of signal to visualize
            axis: Slice axis ('x', 'y', or 'z')
            position: Position along axis (None = center)
            colormap: PyVista colormap name
            show_edges: Whether to show grid lines/edges (default: True)
            show_scalar_bar: Whether to show color bar
            title: Optional title for the plot
            auto_show: If True, automatically display (default: True)
            
        Returns:
            PyVista Plotter object
        """
        if self.voxel_grid is None:
            raise ValueError("No voxel grid set. Call set_voxel_grid() first.")
        
        grid = self._create_pyvista_grid(signal_name)
        metadata = self._get_grid_metadata()
        
        # Determine slice position
        if position is None:
            if axis == "x":
                position = (metadata['bbox_min'][0] + metadata['bbox_max'][0]) / 2.0
            elif axis == "y":
                position = (metadata['bbox_min'][1] + metadata['bbox_max'][1]) / 2.0
            else:  # z
                position = (metadata['bbox_min'][2] + metadata['bbox_max'][2]) / 2.0
        
        plotter = pv.Plotter(notebook=True)
        
        # Extract slice
        if axis == "x":
            slice_mesh = grid.slice(normal="x", origin=(position, 0, 0))
            camera_pos = "yz"
        elif axis == "y":
            slice_mesh = grid.slice(normal="y", origin=(0, position, 0))
            camera_pos = "xz"
        else:  # z
            slice_mesh = grid.slice(normal="z", origin=(0, 0, position))
            camera_pos = "xy"
        
        plotter.add_mesh(
            slice_mesh,
            scalars=signal_name,
            cmap=colormap,
            show_edges=show_edges,
            edge_color='black',
            line_width=1,
            show_scalar_bar=show_scalar_bar,
            scalar_bar_args={"title": signal_name.replace('_', ' ').title()},
        )
        
        plotter.camera_position = camera_pos
        plotter.add_axes()
        
        if title:
            plotter.add_text(title, font_size=12)
        else:
            axis_upper = axis.upper()
            plotter.add_text(
                f"{axis_upper} Slice: {signal_name.replace('_', ' ').title()}",
                font_size=12
            )
        
        if auto_show:
            plotter.show(jupyter_backend="static")
        
        return plotter
    
    def render_isosurface(
        self,
        signal_name: str,
        isovalue: Optional[float] = None,
        colormap: str = "plasma",
        opacity: float = 0.7,
        show_scalar_bar: bool = True,
        title: Optional[str] = None,
        auto_show: bool = True,
    ) -> pv.Plotter:
        """
        Render isosurface at a specific value.
        
        Args:
            signal_name: Name of signal to visualize
            isovalue: Isosurface value (None = use mean value)
            colormap: PyVista colormap name
            opacity: Opacity of surface (0.0 to 1.0)
            show_scalar_bar: Whether to show color bar
            title: Optional title for the plot
            auto_show: If True, automatically display (default: True)
            
        Returns:
            PyVista Plotter object
        """
        if self.voxel_grid is None:
            raise ValueError("No voxel grid set. Call set_voxel_grid() first.")
        
        grid = self._create_pyvista_grid(signal_name)
        signal_array = self._get_signal_array(signal_name)
        
        # Determine isovalue
        if isovalue is None:
            non_zero = signal_array[signal_array != 0.0]
            isovalue = float(np.mean(non_zero)) if len(non_zero) > 0 else 0.0
        
        plotter = pv.Plotter(notebook=True)
        
        # Extract isosurface
        isosurface = grid.contour([isovalue], scalars=signal_name)
        
        if isosurface.n_points > 0:
            plotter.add_mesh(
                isosurface,
                scalars=signal_name,
                cmap=colormap,
                opacity=opacity,
                show_edges=True,
                show_scalar_bar=show_scalar_bar,
                scalar_bar_args={"title": signal_name.replace('_', ' ').title()},
            )
        else:
            plotter.add_text(
                f"No isosurface at value {isovalue:.3f}",
                font_size=12
            )
        
        plotter.add_axes()
        
        if title:
            plotter.add_text(title, font_size=12, position='upper_right')
        else:
            plotter.add_text(
                f"Isosurface: {signal_name.replace('_', ' ').title()} = {isovalue:.3f}",
                font_size=12,
                position='upper_right'
            )
        
        if auto_show:
            plotter.show(jupyter_backend="static")
        
        return plotter
    
    def render_ct_density(
        self,
        signal_name: str = "density",
        mode: str = "slices",
        show_defects: bool = False,
        defect_locations: Optional[list] = None,
        threshold: Optional[float] = None,
        opacity: float = 0.8,
        auto_show: bool = True,
    ) -> Optional[pv.Plotter]:
        """
        Render CT/density data with optimized visualization.
        
        Supports both 2D slice views (grayscale, medical CT style) and 3D volume rendering.
        This is optimized for density/CT scan data visualization.
        
        Args:
            signal_name: Name of density signal (default: "density")
            mode: Visualization mode - 'slices' (2D) or '3d' (volume rendering)
            show_defects: Whether to overlay defect locations (2D mode only)
            defect_locations: List of defect coordinates [(x, y, z), ...]
            threshold: Minimum density to show in 3D mode (None = auto-detect)
            opacity: Opacity for 3D volume rendering (0.0 to 1.0)
            auto_show: If True, automatically display (default: True)
            
        Returns:
            PyVista Plotter object (if using PyVista) or None (if using matplotlib)
        """
        if self.voxel_grid is None:
            raise ValueError("No voxel grid set. Call set_voxel_grid() first.")
        
        signal_array = self._get_signal_array(signal_name)
        if signal_array is None:
            raise ValueError(f"Signal '{signal_name}' not found in grid")
        
        metadata = self._get_grid_metadata()
        dims = metadata['dims']
        
        # 3D volume rendering mode
        if mode == '3d':
            try:
                # Use PyVista for 3D volume rendering
                grid = self._create_pyvista_grid(signal_name)
                
                # Auto-detect threshold if not provided
                if threshold is None:
                    non_zero = signal_array[signal_array != 0.0]
                    if len(non_zero) > 0:
                        threshold = float(np.percentile(non_zero, 10))  # Bottom 10%
                    else:
                        threshold = 0.0
                
                plotter = pv.Plotter(notebook=True)
                
                # Apply threshold
                threshold_mesh = grid.threshold(threshold, scalars=signal_name)
                
                if threshold_mesh.n_points == 0:
                    plotter.add_text(
                        f"No data above threshold ({threshold:.3f})",
                        font_size=12,
                        position='upper_left'
                    )
                else:
                    # Use grayscale colormap for CT data
                    plotter.add_mesh(
                        threshold_mesh,
                        scalars=signal_name,
                        cmap='gray',  # Grayscale for CT
                        opacity=opacity,
                        show_edges=True,  # Show grid lines like laser signals
                        edge_color='black',
                        line_width=1,
                        show_scalar_bar=True,
                        scalar_bar_args={"title": "Density (g/cm³)"},
                    )
                
                plotter.add_axes()
                plotter.add_text(
                    f"CT 3D View: {signal_name.replace('_', ' ').title()}",
                    font_size=12,
                    position='upper_right'
                )
                
                if auto_show:
                    plotter.show(jupyter_backend="static")
                
                return plotter
                
            except Exception as e:
                # Fallback to 2D slices if 3D fails
                import warnings
                warnings.warn(f"3D CT visualization failed: {e}. Falling back to 2D slices.")
                mode = 'slices'
        
        # 2D slice mode (default) - Use matplotlib for CT-style visualization
        if mode == 'slices':
            import matplotlib.pyplot as plt
            from IPython.display import display, HTML
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # XY Slice (Z = middle)
        z_slice = dims[2] // 2 if len(dims) > 2 else 0
        if len(signal_array.shape) == 3:
            xy_slice = signal_array[:, :, z_slice]
        else:
            xy_slice = signal_array
        im1 = axes[0, 0].imshow(xy_slice, cmap='gray', origin='lower', aspect='auto')
        axes[0, 0].set_title(f'XY Slice (Z = {z_slice})')
        axes[0, 0].set_xlabel('X (voxels)')
        axes[0, 0].set_ylabel('Y (voxels)')
        plt.colorbar(im1, ax=axes[0, 0], label='Density')
        
        # XZ Slice (Y = middle)
        y_slice = dims[1] // 2 if len(dims) > 1 else 0
        if len(signal_array.shape) == 3:
            xz_slice = signal_array[:, y_slice, :]
        else:
            xz_slice = signal_array
        im2 = axes[0, 1].imshow(xz_slice, cmap='gray', origin='lower', aspect='auto')
        axes[0, 1].set_title(f'XZ Slice (Y = {y_slice})')
        axes[0, 1].set_xlabel('X (voxels)')
        axes[0, 1].set_ylabel('Z (voxels)')
        plt.colorbar(im2, ax=axes[0, 1], label='Density')
        
        # YZ Slice (X = middle)
        x_slice = dims[0] // 2 if len(dims) > 0 else 0
        if len(signal_array.shape) == 3:
            yz_slice = signal_array[x_slice, :, :]
        else:
            yz_slice = signal_array
        im3 = axes[1, 0].imshow(yz_slice, cmap='gray', origin='lower', aspect='auto')
        axes[1, 0].set_title(f'YZ Slice (X = {x_slice})')
        axes[1, 0].set_xlabel('Y (voxels)')
        axes[1, 0].set_ylabel('Z (voxels)')
        plt.colorbar(im3, ax=axes[1, 0], label='Density')
        
        # Defect locations or statistics
        if show_defects and defect_locations and len(defect_locations) > 0:
            try:
                defect_coords_list = []
                for d in defect_locations[:100]:  # Limit to 100 for performance
                    if isinstance(d, dict):
                        x = d.get('x', 0)
                        y = d.get('y', 0)
                        z = d.get('z', 0)
                    elif isinstance(d, (list, tuple, np.ndarray)) and len(d) >= 3:
                        x, y, z = d[0], d[1], d[2]
                    else:
                        continue
                    defect_coords_list.append([x, y, z])
                
                if defect_coords_list:
                    defect_coords = np.array(defect_coords_list)
                    scatter = axes[1, 1].scatter(
                        defect_coords[:, 0],
                        defect_coords[:, 1],
                        c=defect_coords[:, 2],
                        cmap='Reds',
                        s=50,
                        alpha=0.7
                    )
                    axes[1, 1].set_xlabel('X (mm)')
                    axes[1, 1].set_ylabel('Y (mm)')
                    axes[1, 1].set_title(f'Defect Locations ({len(defect_locations)} total)')
                    axes[1, 1].grid(True, alpha=0.3)
                    plt.colorbar(scatter, ax=axes[1, 1], label='Z (mm)')
                else:
                    axes[1, 1].text(0.5, 0.5, 'No valid defect coordinates',
                                   ha='center', va='center', transform=axes[1, 1].transAxes)
                    axes[1, 1].set_title('Defect Locations')
            except Exception as e:
                axes[1, 1].text(0.5, 0.5, f'Error displaying defects: {str(e)}',
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Defect Locations')
        else:
            # Show statistics
            stats_text = f"""
            Density Statistics:
            Min: {np.min(signal_array):.3f}
            Max: {np.max(signal_array):.3f}
            Mean: {np.mean(signal_array):.3f}
            Std: {np.std(signal_array):.3f}
            Grid Size: {dims}
            """
            axes[1, 1].text(0.5, 0.5, stats_text,
                           ha='center', va='center', transform=axes[1, 1].transAxes,
                           fontsize=10, family='monospace')
            axes[1, 1].set_title('Statistics')
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        if auto_show:
            plt.show()
        
        return None  # Matplotlib doesn't return a plotter object
    
    def render_ispm_temperature(
        self,
        signal_name: str = "temperature",
        mode: str = "slices",
        colormap: str = "hot",
        show_statistics: bool = True,
        threshold: Optional[float] = None,
        opacity: float = 0.8,
        auto_show: bool = True,
    ) -> Optional[pv.Plotter]:
        """
        Render ISPM temperature data with optimized visualization.
        
        Supports both 2D slice views (multiple thermal views) and 3D volume rendering.
        Uses 'hot' colormap (standard for temperature) optimized for thermal/ISPM data visualization.
        
        Args:
            signal_name: Name of temperature signal (default: "temperature")
            mode: Visualization mode - 'slices' (2D) or '3d' (volume rendering)
            colormap: Colormap for temperature (default: "hot")
            show_statistics: Whether to show temperature statistics (2D mode only)
            threshold: Minimum temperature to show in 3D mode (None = auto-detect)
            opacity: Opacity for 3D volume rendering (0.0 to 1.0)
            auto_show: If True, automatically display (default: True)
            
        Returns:
            PyVista Plotter object (if using PyVista) or None (if using matplotlib)
        """
        if self.voxel_grid is None:
            raise ValueError("No voxel grid set. Call set_voxel_grid() first.")
        
        signal_array = self._get_signal_array(signal_name)
        if signal_array is None:
            raise ValueError(f"Signal '{signal_name}' not found in grid")
        
        metadata = self._get_grid_metadata()
        dims = metadata['dims']
        
        # 3D volume rendering mode for ISPM
        if mode == '3d':
            try:
                # Use PyVista for 3D volume rendering with hot colormap
                grid = self._create_pyvista_grid(signal_name)
                
                # Auto-detect threshold if not provided
                if threshold is None:
                    non_zero = signal_array[signal_array != 0.0]
                    if len(non_zero) > 0:
                        threshold = float(np.percentile(non_zero, 10))  # Bottom 10%
                    else:
                        threshold = 0.0
                
                plotter = pv.Plotter(notebook=True)
                
                # Apply threshold
                threshold_mesh = grid.threshold(threshold, scalars=signal_name)
                
                if threshold_mesh.n_points == 0:
                    plotter.add_text(
                        f"No data above threshold ({threshold:.1f} K)",
                        font_size=12,
                        position='upper_left'
                    )
                else:
                    # Use hot colormap for temperature data
                    plotter.add_mesh(
                        threshold_mesh,
                        scalars=signal_name,
                        cmap=colormap,  # 'hot' for temperature
                        opacity=opacity,
                        show_edges=True,  # Show grid lines like laser signals
                        edge_color='black',
                        line_width=1,
                        show_scalar_bar=True,
                        scalar_bar_args={"title": "Temperature (K)"},
                    )
                
                plotter.add_axes()
                plotter.add_text(
                    f"ISPM 3D View: {signal_name.replace('_', ' ').title()}",
                    font_size=12,
                    position='upper_right'
                )
                
                if auto_show:
                    plotter.show(jupyter_backend="static")
                
                return plotter
                
            except Exception as e:
                # Fallback to 2D slices if 3D fails
                import warnings
                warnings.warn(f"3D ISPM visualization failed: {e}. Falling back to 2D slices.")
                mode = 'slices'
        
        # 2D slice mode (default) - Use matplotlib for ISPM-style visualization
        if mode == 'slices':
            import matplotlib.pyplot as plt
            from IPython.display import display, HTML
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Top-left: XY Slice (Z = middle) - Temperature distribution
        z_slice = dims[2] // 2 if len(dims) > 2 else 0
        if len(signal_array.shape) == 3:
            xy_slice = signal_array[:, :, z_slice]
        else:
            xy_slice = signal_array
        im1 = axes[0, 0].imshow(xy_slice, cmap=colormap, origin='lower', aspect='auto')
        axes[0, 0].set_title(f'Temperature Distribution - XY Slice (Z = {z_slice})')
        axes[0, 0].set_xlabel('X (voxels)')
        axes[0, 0].set_ylabel('Y (voxels)')
        plt.colorbar(im1, ax=axes[0, 0], label='Temperature (K)')
        
        # Top-right: Temperature histogram
        temp_flat = signal_array.flatten()
        temp_flat = temp_flat[temp_flat > 0]  # Remove zeros
        if len(temp_flat) > 0:
            axes[0, 1].hist(temp_flat, bins=50, color='red', alpha=0.7, edgecolor='black')
            axes[0, 1].set_xlabel('Temperature (K)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Temperature Distribution')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Add statistics text
            mean_temp = np.mean(temp_flat)
            std_temp = np.std(temp_flat)
            max_temp = np.max(temp_flat)
            min_temp = np.min(temp_flat)
            stats_text = f'Mean: {mean_temp:.1f} K\nStd: {std_temp:.1f} K\nMax: {max_temp:.1f} K\nMin: {min_temp:.1f} K'
            axes[0, 1].text(0.98, 0.98, stats_text, transform=axes[0, 1].transAxes,
                           verticalalignment='top', horizontalalignment='right',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                           fontsize=9)
        
        # Bottom-left: XZ Slice (Y = middle)
        y_slice = dims[1] // 2 if len(dims) > 1 else 0
        if len(signal_array.shape) == 3:
            xz_slice = signal_array[:, y_slice, :]
        else:
            xz_slice = signal_array
        im3 = axes[1, 0].imshow(xz_slice, cmap=colormap, origin='lower', aspect='auto')
        axes[1, 0].set_title(f'Temperature - XZ Slice (Y = {y_slice})')
        axes[1, 0].set_xlabel('X (voxels)')
        axes[1, 0].set_ylabel('Z (voxels)')
        plt.colorbar(im3, ax=axes[1, 0], label='Temperature (K)')
        
        # Bottom-right: Temperature profile along Z axis (if 3D)
        if len(signal_array.shape) == 3 and dims[2] > 1:
            # Average temperature per Z layer
            z_profile = np.mean(signal_array, axis=(0, 1))
            z_indices = np.arange(len(z_profile))
            axes[1, 1].plot(z_indices, z_profile, 'r-', linewidth=2, marker='o', markersize=4)
            axes[1, 1].set_xlabel('Z Layer (voxels)')
            axes[1, 1].set_ylabel('Average Temperature (K)')
            axes[1, 1].set_title('Temperature Profile vs Z Layer')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add trend line if significant
            if len(z_profile) > 2:
                z_coords = np.arange(len(z_profile))
                coeffs = np.polyfit(z_coords, z_profile, 1)
                trend_line = np.polyval(coeffs, z_coords)
                axes[1, 1].plot(z_coords, trend_line, 'b--', alpha=0.5, label=f'Trend (slope: {coeffs[0]:.2f} K/layer)')
                axes[1, 1].legend()
        else:
            # Show statistics panel
            if show_statistics and len(temp_flat) > 0:
                stats_text = f"""
                Temperature Statistics:
                Min: {np.min(temp_flat):.1f} K
                Max: {np.max(temp_flat):.1f} K
                Mean: {np.mean(temp_flat):.1f} K
                Std: {np.std(temp_flat):.1f} K
                Median: {np.median(temp_flat):.1f} K
                Grid Size: {dims}
                """
                axes[1, 1].text(0.5, 0.5, stats_text,
                               ha='center', va='center', transform=axes[1, 1].transAxes,
                               fontsize=10, family='monospace',
                               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
                axes[1, 1].set_title('Statistics')
                axes[1, 1].axis('off')
            else:
                axes[1, 1].text(0.5, 0.5, 'No statistics available',
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Statistics')
        
        plt.tight_layout()
        if auto_show:
            plt.show()
        
        return None  # Matplotlib doesn't return a plotter object
    
    def get_available_signals(self) -> list:
        """
        Get list of available signals in the grid.
        
        Returns:
            List of signal names
        """
        if self.voxel_grid is None:
            return []
        
        # Try _signal_arrays
        if hasattr(self.voxel_grid, '_signal_arrays'):
            return list(self.voxel_grid._signal_arrays.keys())
        
        # Try available_signals attribute
        if hasattr(self.voxel_grid, 'available_signals'):
            if isinstance(self.voxel_grid.available_signals, (set, list)):
                return sorted(list(self.voxel_grid.available_signals))
        
        return []


# Convenience function for notebooks
def visualize_signal_grid(
    voxel_grid,
    signal_name: str,
    mode: str = "3d",
    **kwargs
) -> pv.Plotter:
    """
    Convenience function to visualize a signal grid.
    
    Args:
        voxel_grid: VoxelGrid object or grid with _signal_arrays
        signal_name: Name of signal to visualize
        mode: Visualization mode ('3d', 'slice', 'isosurface')
        **kwargs: Additional arguments passed to render method
        
    Returns:
        PyVista Plotter object
        
    Example:
        >>> from am_qadf.visualization import visualize_signal_grid
        >>> plotter = visualize_signal_grid(mapped_grid, 'temperature', mode='3d')
    """
    visualizer = SignalGridVisualizer(voxel_grid)
    
    if mode == "3d":
        return visualizer.render_3d(signal_name, **kwargs)
    elif mode == "slice":
        return visualizer.render_slice(signal_name, **kwargs)
    elif mode == "isosurface":
        return visualizer.render_isosurface(signal_name, **kwargs)
    elif mode == "ct" or mode == "density":
        return visualizer.render_ct_density(signal_name, **kwargs)
    elif mode == "ispm" or mode == "temperature":
        return visualizer.render_ispm_temperature(signal_name, **kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use '3d', 'slice', 'isosurface', 'ct'/'density', or 'ispm'/'temperature'")
