"""
Custom Hatching Visualizer

Inspired by pyslm's visualization functions, but works with:
1. pyslm Layer objects (direct use)
2. MongoDB data (reconstructed from stored point arrays)

Provides the same interface as pyslm.visualise.plot() and plotLayers()
"""

import logging
from typing import Any, List, Tuple, Optional, Dict, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.collections as mc

logger = logging.getLogger(__name__)


class HatchingVisualizer:
    """
    Custom hatching visualizer inspired by pyslm's visualization.
    
    Can visualize:
    - pyslm Layer objects (direct)
    - MongoDB layer documents (reconstructed)
    """
    
    def __init__(self):
        """Initialize the visualizer."""
        pass
    
    def plot_layers(self, 
                   layers: Union[List[Any], List[Dict]],
                   plot_contours: bool = True,
                   plot_hatches: bool = True,
                   plot_points: bool = True,
                   plot_3d: bool = True,
                   plot_colorbar: bool = False,
                   index: Optional[str] = None,
                   handle: Optional[Tuple[plt.Figure, plt.Axes]] = None,
                   colormap: str = 'rainbow',
                   linewidth: float = 0.5) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot multiple layers (inspired by pyslm.visualise.plotLayers).
        
        Args:
            layers: List of pyslm Layer objects or MongoDB layer documents
            plot_contours: Plot contour paths
            plot_hatches: Plot hatch paths
            plot_points: Plot point exposures
            plot_3d: Use 3D projection
            plot_colorbar: Show colorbar (only one colorbar for all layers)
            index: Property to color by ('laser_power', 'scan_speed', 'energy_density', 'length', etc.)
            handle: Existing matplotlib (fig, ax) to reuse
            colormap: Matplotlib colormap name
            linewidth: Line width for hatches
            
        Returns:
            (fig, ax) tuple
        """
        if handle:
            fig, ax = handle[0], handle[1]
        else:
            if plot_3d:
                fig = plt.figure()
                ax = plt.axes(projection='3d')
            else:
                fig, ax = plt.subplots()
        
        # Check if layers are pyslm Layer objects or MongoDB documents
        is_pyslm_layer = False
        if layers and hasattr(layers[0], 'getHatchGeometry'):
            is_pyslm_layer = True
        
        # Collect all color values across all layers to create a unified colormap
        all_color_values = []
        line_collections = []
        
        # First pass: collect all data and create LineCollections
        for layer in layers:
            if is_pyslm_layer:
                # Use pyslm Layer object directly
                z_pos = float(layer.z) / 1000.0  # Convert microns to mm
                result = self._plot_pyslm_layer(
                    layer, z_pos, fig, ax,
                    plot_contours, plot_hatches, plot_points,
                    plot_3d, False, index, colormap, linewidth  # Don't add colorbar per layer
                )
                (fig, ax), lc, color_vals = result
                if lc is not None:
                    line_collections.append(lc)
                    if color_vals is not None:
                        all_color_values.extend(color_vals)
            else:
                # Reconstruct from MongoDB document
                z_pos = layer.get('z_position', 0.0)
                result = self._plot_mongodb_layer(
                    layer, z_pos, fig, ax,
                    plot_contours, plot_hatches, plot_points,
                    plot_3d, False, index, colormap, linewidth  # Don't add colorbar per layer
                )
                (fig, ax), lc, color_vals = result
                if lc is not None:
                    line_collections.append(lc)
                    if color_vals is not None:
                        all_color_values.extend(color_vals)
        
        # Apply unified normalization and add single colorbar if requested
        if plot_colorbar and line_collections and all_color_values:
            # Create unified normalization
            all_color_values = np.array(all_color_values)
            vmin, vmax = all_color_values.min(), all_color_values.max()
            
            # Apply same normalization to all LineCollections
            for lc in line_collections:
                lc.set_clim(vmin, vmax)
            
            # Add single colorbar using the last LineCollection
            label = index.replace('_', ' ').title() if index else 'Hatch Index'
            plt.colorbar(line_collections[-1], ax=ax, label=label)
        
        return fig, ax
    
    def _plot_pyslm_layer(self,
                          layer: Any,
                          z_pos: float,
                          fig: plt.Figure,
                          ax: plt.Axes,
                          plot_contours: bool,
                          plot_hatches: bool,
                          plot_points: bool,
                          plot_3d: bool,
                          plot_colorbar: bool,
                          index: Optional[str],
                          colormap: str,
                          linewidth: float) -> Tuple[Tuple[plt.Figure, plt.Axes], Optional[mc.LineCollection], Optional[List[float]]]:
        """Plot a pyslm Layer object (uses pyslm's native structure)."""
        try:
            import pyslm
            from pyslm import visualise
            
            # Use pyslm's plot function directly
            # Note: pyslm's plot doesn't return LineCollection, so we fall back to manual
            # for better control over colorbar
            result = visualise.plot(
                layer,
                zPos=z_pos,
                plotContours=plot_contours,
                plotHatches=plot_hatches,
                plotPoints=plot_points,
                plot3D=plot_3d,
                plotColorbar=False,  # Don't let pyslm add colorbar, we'll handle it
                index=index or '',
                handle=(fig, ax)
            )
            # pyslm's plot doesn't return LineCollection, so return None
            # This means unified colorbar won't work with pyslm's native plot
            return (result[0], result[1]), None, None
        except Exception as e:
            logger.warning(f"Could not use pyslm plot, falling back to manual: {e}")
            return self._plot_pyslm_layer_manual(
                layer, z_pos, fig, ax, plot_contours, plot_hatches, plot_points,
                plot_3d, plot_colorbar, index, colormap, linewidth
            )
    
    def _plot_pyslm_layer_manual(self,
                                 layer: Any,
                                 z_pos: float,
                                 fig: plt.Figure,
                                 ax: plt.Axes,
                                 plot_contours: bool,
                                 plot_hatches: bool,
                                 plot_points: bool,
                                 plot_3d: bool,
                                 plot_colorbar: bool,
                                 index: Optional[str],
                                 colormap: str,
                                 linewidth: float) -> Tuple[Tuple[plt.Figure, plt.Axes], Optional[mc.LineCollection], Optional[List[float]]]:
        """Manually plot pyslm Layer (inspired by pyslm's plot function)."""
        lc = None
        color_values = None
        
        # Plot hatches
        if plot_hatches:
            hatch_geoms = layer.getHatchGeometry()
            if len(hatch_geoms) > 0:
                hatches = np.vstack([hatch_geom.coords.reshape(-1, 2, 2) for hatch_geom in hatch_geoms])
                lc = mc.LineCollection(hatches, cmap=plt.cm.get_cmap(colormap), linewidths=linewidth)
                
                # Set color array based on index
                if index and hasattr(hatch_geoms[0], index):
                    values = np.vstack([
                        np.tile(getattr(hatch_geom, index), [int(len(hatch_geom.coords)/2), 1])
                        for hatch_geom in hatch_geoms
                    ])
                    color_values = values.ravel().tolist()
                    lc.set_array(values.ravel())
                elif index == 'length':
                    delta = hatches[:, 1, :] - hatches[:, 0, :]
                    dist = np.sqrt(delta[:, 0]**2 + delta[:, 1]**2)
                    color_values = dist.ravel().tolist()
                    lc.set_array(dist.ravel())
                else:
                    color_values = np.arange(len(hatches)).tolist()
                    lc.set_array(np.arange(len(hatches)))
                
                if plot_3d:
                    ax.add_collection3d(lc, zs=z_pos)
                else:
                    ax.add_collection(lc)
                
                # Only add colorbar if explicitly requested (for single layer plots)
                if plot_colorbar:
                    plt.colorbar(lc, ax=ax, label=index or 'Hatch Index')
        
        # Plot contours
        if plot_contours:
            for contour_geom in layer.getContourGeometry():
                if hasattr(contour_geom, 'subType'):
                    if contour_geom.subType == 'inner':
                        line_color = '#f57900'
                        line_width = 1.0
                    elif contour_geom.subType == 'outer':
                        line_color = '#204a87'
                        line_width = 1.4
                    else:
                        line_color = 'k'
                        line_width = 0.7
                else:
                    line_color = 'k'
                    line_width = 0.7
                
                if plot_3d:
                    ax.plot(contour_geom.coords[:, 0], contour_geom.coords[:, 1],
                           zs=z_pos, color=line_color, linewidth=line_width)
                else:
                    ax.plot(contour_geom.coords[:, 0], contour_geom.coords[:, 1],
                           color=line_color, linewidth=line_width)
        
        # Plot points
        if plot_points:
            point_geoms = layer.getPointsGeometry()
            if len(point_geoms) > 0:
                scatter_points = np.vstack([points_geom.coords for points_geom in point_geoms])
                if plot_3d:
                    ax.scatter3D(scatter_points[:, 0], scatter_points[:, 1], z_pos, c='black', s=1.0)
                else:
                    ax.scatter(scatter_points[:, 0], scatter_points[:, 1], c='black', s=1.0)
        
        return (fig, ax), lc, color_values
    
    def _plot_mongodb_layer(self,
                           layer_doc: Dict,
                           z_pos: float,
                           fig: plt.Figure,
                           ax: plt.Axes,
                           plot_contours: bool,
                           plot_hatches: bool,
                           plot_points: bool,
                           plot_3d: bool,
                           plot_colorbar: bool,
                           index: Optional[str],
                           colormap: str,
                           linewidth: float) -> Tuple[Tuple[plt.Figure, plt.Axes], Optional[mc.LineCollection], Optional[List[float]]]:
        """
        Plot a layer from MongoDB document (reconstructed from stored points).
        Inspired by pyslm's plot function but works with MongoDB data structure.
        
        Returns:
            ((fig, ax), LineCollection, color_values) tuple
        """
        lc = None
        color_values = None
        
        # Plot hatches
        if plot_hatches:
            hatches_list = layer_doc.get('hatches', [])
            if len(hatches_list) > 0:
                # Reconstruct hatch segments from MongoDB data
                hatch_segments = []
                color_values = []
                
                for hatch in hatches_list:
                    points = hatch.get('points', [])
                    if len(points) < 2:
                        continue
                    
                    points_array = np.array(points)
                    # Extract only x, y coordinates (2D) for LineCollection
                    # Points may be 2D or 3D, so we take only first 2 columns
                    if points_array.shape[1] >= 2:
                        points_2d = points_array[:, :2]
                    else:
                        points_2d = points_array
                    
                    # Convert to (N, 2, 2) format: each segment is [start, end]
                    for i in range(len(points_2d) - 1):
                        segment = np.array([[points_2d[i], points_2d[i+1]]])
                        hatch_segments.append(segment)
                        
                        # Get color value based on index
                        if index == 'laser_power':
                            color_values.append(hatch.get('laser_power', 200.0))
                        elif index == 'scan_speed':
                            color_values.append(hatch.get('scan_speed', 500.0))
                        elif index == 'energy_density':
                            color_values.append(hatch.get('energy_density', 0.4))
                        elif index == 'length':
                            # Calculate segment length (using 2D coordinates)
                            delta = points_2d[i+1] - points_2d[i]
                            length = np.sqrt(np.sum(delta**2))
                            color_values.append(length)
                        else:
                            color_values.append(len(hatch_segments))
                
                if hatch_segments:
                    hatches = np.vstack(hatch_segments)
                    lc = mc.LineCollection(hatches, cmap=plt.cm.get_cmap(colormap), linewidths=linewidth)
                    lc.set_array(np.array(color_values))
                    
                    if plot_3d:
                        ax.add_collection3d(lc, zs=z_pos)
                    else:
                        ax.add_collection(lc)
                    
                    # Only add colorbar if explicitly requested (for single layer plots)
                    if plot_colorbar:
                        label = index.replace('_', ' ').title() if index else 'Hatch Index'
                        plt.colorbar(lc, ax=ax, label=label)
        
        # Plot contours (if stored in MongoDB)
        if plot_contours:
            contours = layer_doc.get('contours', [])
            for contour in contours:
                points = contour.get('points', [])
                if len(points) > 0:
                    points_array = np.array(points)
                    line_color = contour.get('color', 'k')
                    line_width = contour.get('linewidth', 0.7)
                    
                    if plot_3d:
                        ax.plot(points_array[:, 0], points_array[:, 1],
                               zs=z_pos, color=line_color, linewidth=line_width)
                    else:
                        ax.plot(points_array[:, 0], points_array[:, 1],
                               color=line_color, linewidth=line_width)
        
        return (fig, ax), lc, color_values
    
    def plot_single_layer(self,
                         layer: Union[Any, Dict],
                         z_pos: Optional[float] = None,
                         plot_contours: bool = True,
                         plot_hatches: bool = True,
                         plot_points: bool = True,
                         plot_3d: bool = True,
                         plot_colorbar: bool = False,
                         index: Optional[str] = None,
                         handle: Optional[Tuple[plt.Figure, plt.Axes]] = None,
                         colormap: str = 'rainbow',
                         linewidth: float = 0.5) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot a single layer (inspired by pyslm.visualise.plot).
        
        Args:
            layer: pyslm Layer object or MongoDB layer document
            z_pos: Z position (auto-detected from layer if not provided)
            plot_contours: Plot contour paths
            plot_hatches: Plot hatch paths
            plot_points: Plot point exposures
            plot_3d: Use 3D projection
            plot_colorbar: Show colorbar
            index: Property to color by
            handle: Existing matplotlib (fig, ax) to reuse
            colormap: Matplotlib colormap name
            linewidth: Line width for hatches
            
        Returns:
            (fig, ax) tuple
        """
        if handle:
            fig, ax = handle[0], handle[1]
        else:
            if plot_3d:
                fig = plt.figure()
                ax = plt.axes(projection='3d')
            else:
                fig, ax = plt.subplots()
        
        # Determine if it's a pyslm Layer or MongoDB document
        is_pyslm_layer = hasattr(layer, 'getHatchGeometry')
        
        if z_pos is None:
            if is_pyslm_layer:
                z_pos = float(layer.z) / 1000.0
            else:
                z_pos = layer.get('z_position', 0.0)
        
        if is_pyslm_layer:
            result = self._plot_pyslm_layer(
                layer, z_pos, fig, ax, plot_contours, plot_hatches, plot_points,
                plot_3d, plot_colorbar, index, colormap, linewidth
            )
            return result[0]  # Return (fig, ax) tuple
        else:
            result = self._plot_mongodb_layer(
                layer, z_pos, fig, ax, plot_contours, plot_hatches, plot_points,
                plot_3d, plot_colorbar, index, colormap, linewidth
            )
            return result[0]  # Return (fig, ax) tuple
    
    def plot_grid(self,
                  voxel_grid: Any,
                  signal_name: Optional[str] = None,
                  plot_3d: bool = True,
                  plot_slices: bool = False,
                  slice_z_levels: Optional[List[float]] = None,
                  plot_voxel_centers: bool = True,
                  plot_wireframe: bool = False,
                  colormap: str = 'plasma',
                  plot_colorbar: bool = True,
                  handle: Optional[Tuple[plt.Figure, plt.Axes]] = None,
                  alpha: float = 0.6,
                  point_size: float = 10.0) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot voxel grid for signal mapping visualization.
        
        Args:
            voxel_grid: VoxelGrid object (from am_qadf.voxelization)
            signal_name: Signal name to color by (e.g., 'power', 'temperature')
            plot_3d: Use 3D projection
            plot_slices: Plot 2D slices at specific Z levels
            slice_z_levels: List of Z positions for slices (if None, auto-select)
            plot_voxel_centers: Plot voxel centers as points colored by signal
            plot_wireframe: Plot voxel boundaries as wireframe
            colormap: Matplotlib colormap name
            plot_colorbar: Show colorbar
            handle: Existing matplotlib (fig, ax) to reuse
            alpha: Transparency for voxel visualization
            point_size: Size of voxel center points
            
        Returns:
            (fig, ax) tuple
        """
        if handle:
            fig, ax = handle[0], handle[1]
        else:
            if plot_3d:
                fig = plt.figure()
                ax = plt.axes(projection='3d')
            else:
                fig, ax = plt.subplots()
        
        # Get signal array if signal_name provided
        signal_array = None
        if signal_name and hasattr(voxel_grid, 'get_signal_array'):
            try:
                signal_array = voxel_grid.get_signal_array(signal_name, default=0.0)
            except Exception as e:
                logger.warning(f"Could not get signal array '{signal_name}': {e}")
        
        # Plot voxel centers as points
        if plot_voxel_centers:
            voxel_centers = []
            signal_values = []
            
            # Get all voxel centers
            for voxel_key, voxel_data in voxel_grid.voxels.items():
                i, j, k = voxel_key
                x, y, z = voxel_grid._voxel_to_world(i, j, k)
                voxel_centers.append([x, y, z])
                
                # Get signal value
                if signal_array is not None:
                    # Get value from signal array
                    if i < signal_array.shape[0] and j < signal_array.shape[1] and k < signal_array.shape[2]:
                        signal_values.append(float(signal_array[i, j, k]))
                    else:
                        signal_values.append(0.0)
                elif signal_name and hasattr(voxel_data, 'signals') and signal_name in voxel_data.signals:
                    signal_values.append(float(voxel_data.signals[signal_name]))
                else:
                    signal_values.append(0.0)
            
            if voxel_centers:
                centers_array = np.array(voxel_centers)
                values_array = np.array(signal_values)
                
                if plot_3d:
                    scatter = ax.scatter(centers_array[:, 0], centers_array[:, 1], centers_array[:, 2],
                                        c=values_array, cmap=colormap, s=point_size, alpha=alpha)
                else:
                    scatter = ax.scatter(centers_array[:, 0], centers_array[:, 1],
                                        c=values_array, cmap=colormap, s=point_size, alpha=alpha)
                
                if plot_colorbar and signal_name:
                    label = signal_name.replace('_', ' ').title()
                    plt.colorbar(scatter, ax=ax, label=label)
        
        # Plot 2D slices
        if plot_slices:
            if signal_array is None:
                logger.warning("Cannot plot slices without signal array")
            else:
                if slice_z_levels is None:
                    # Auto-select slices: evenly distributed across Z range
                    z_min = voxel_grid.bbox_min[2]
                    z_max = voxel_grid.bbox_max[2]
                    z_range = z_max - z_min
                    
                    # Show more slices for better visualization (5-10 slices depending on grid size)
                    n_slices = min(max(5, int(voxel_grid.dims[2] / 10)), 10)  # 5-10 slices
                    if z_range > 0:
                        slice_z_levels = np.linspace(z_min, z_max, n_slices).tolist()
                    else:
                        slice_z_levels = [z_min]
                
                for z_level in slice_z_levels:
                    # Find closest Z index
                    k = int((z_level - voxel_grid.bbox_min[2]) / voxel_grid.resolution)
                    k = np.clip(k, 0, voxel_grid.dims[2] - 1)
                    
                    # Extract slice
                    slice_data = signal_array[:, :, k]
                    
                    # Create meshgrid for X, Y coordinates
                    x_coords = np.linspace(voxel_grid.bbox_min[0], voxel_grid.bbox_max[0], voxel_grid.dims[0])
                    y_coords = np.linspace(voxel_grid.bbox_min[1], voxel_grid.bbox_max[1], voxel_grid.dims[1])
                    X, Y = np.meshgrid(x_coords, y_coords)
                    
                    if plot_3d:
                        ax.contourf(X, Y, slice_data.T, levels=20, zdir='z', offset=z_level,
                                   cmap=colormap, alpha=alpha)
                    else:
                        ax.contourf(X, Y, slice_data.T, levels=20, cmap=colormap, alpha=alpha)
        
        # Plot wireframe (voxel boundaries)
        if plot_wireframe:
            # Plot bounding box wireframe
            bbox_min = voxel_grid.bbox_min
            bbox_max = voxel_grid.bbox_max
            
            # Create wireframe cube for bounding box
            if plot_3d:
                # Draw bounding box edges using plot
                # Bottom face
                ax.plot([bbox_min[0], bbox_max[0]], [bbox_min[1], bbox_min[1]], [bbox_min[2], bbox_min[2]], 
                       'gray', linewidth=0.5, alpha=0.3)
                ax.plot([bbox_min[0], bbox_min[0]], [bbox_min[1], bbox_max[1]], [bbox_min[2], bbox_min[2]], 
                       'gray', linewidth=0.5, alpha=0.3)
                ax.plot([bbox_max[0], bbox_max[0]], [bbox_min[1], bbox_max[1]], [bbox_min[2], bbox_min[2]], 
                       'gray', linewidth=0.5, alpha=0.3)
                ax.plot([bbox_min[0], bbox_max[0]], [bbox_max[1], bbox_max[1]], [bbox_min[2], bbox_min[2]], 
                       'gray', linewidth=0.5, alpha=0.3)
                # Top face
                ax.plot([bbox_min[0], bbox_max[0]], [bbox_min[1], bbox_min[1]], [bbox_max[2], bbox_max[2]], 
                       'gray', linewidth=0.5, alpha=0.3)
                ax.plot([bbox_min[0], bbox_min[0]], [bbox_min[1], bbox_max[1]], [bbox_max[2], bbox_max[2]], 
                       'gray', linewidth=0.5, alpha=0.3)
                ax.plot([bbox_max[0], bbox_max[0]], [bbox_min[1], bbox_max[1]], [bbox_max[2], bbox_max[2]], 
                       'gray', linewidth=0.5, alpha=0.3)
                ax.plot([bbox_min[0], bbox_max[0]], [bbox_max[1], bbox_max[1]], [bbox_max[2], bbox_max[2]], 
                       'gray', linewidth=0.5, alpha=0.3)
                # Vertical edges
                ax.plot([bbox_min[0], bbox_min[0]], [bbox_min[1], bbox_min[1]], [bbox_min[2], bbox_max[2]], 
                       'gray', linewidth=0.5, alpha=0.3)
                ax.plot([bbox_max[0], bbox_max[0]], [bbox_min[1], bbox_min[1]], [bbox_min[2], bbox_max[2]], 
                       'gray', linewidth=0.5, alpha=0.3)
                ax.plot([bbox_min[0], bbox_min[0]], [bbox_max[1], bbox_max[1]], [bbox_min[2], bbox_max[2]], 
                       'gray', linewidth=0.5, alpha=0.3)
                ax.plot([bbox_max[0], bbox_max[0]], [bbox_max[1], bbox_max[1]], [bbox_min[2], bbox_max[2]], 
                       'gray', linewidth=0.5, alpha=0.3)
            else:
                # 2D: just plot bounding box rectangle
                from matplotlib.patches import Rectangle
                rect = Rectangle((bbox_min[0], bbox_min[1]),
                               bbox_max[0] - bbox_min[0],
                               bbox_max[1] - bbox_min[1],
                               fill=False, edgecolor='gray', linewidth=0.5, alpha=0.3)
                ax.add_patch(rect)
        
        # Set labels
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        if plot_3d:
            ax.set_zlabel('Z (mm)')
        
        return fig, ax
