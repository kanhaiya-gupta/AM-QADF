"""
Voxel Renderer

PyVista-based rendering for voxel grids.
Supports 3D visualization, slice views, and isosurfaces.
"""

from typing import Optional, Tuple, List, Dict, Any
import numpy as np

try:
    import pyvista as pv

    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    pv = None


class VoxelRenderer:
    """
    Renderer for voxel grids using PyVista.

    Provides methods for:
    - 3D voxel visualization
    - Slice views (X, Y, Z planes)
    - Isosurface generation
    - Multi-signal overlay
    """

    def __init__(self, voxel_grid=None):
        """
        Initialize voxel renderer.

        Args:
            voxel_grid: VoxelGrid object to render
        """
        if not PYVISTA_AVAILABLE:
            raise ImportError("PyVista is required for visualization. Install with: pip install pyvista")

        self.voxel_grid = voxel_grid
        self._pyvista_grid = None

    def set_voxel_grid(self, voxel_grid):
        """Set the voxel grid to render."""
        self.voxel_grid = voxel_grid
        self._pyvista_grid = None  # Reset cached grid

    def _create_pyvista_grid(self) -> Optional[pv.ImageData]:
        """
        Create PyVista ImageData from voxel grid.

        Returns:
            PyVista ImageData object or None if voxel_grid is not set
        """
        if self.voxel_grid is None:
            return None

        if self._pyvista_grid is None:
            # Create PyVista grid
            grid = pv.ImageData()
            grid.dimensions = tuple(self.voxel_grid.dims)
            grid.spacing = (self.voxel_grid.resolution,) * 3
            grid.origin = tuple(self.voxel_grid.bbox_min)
            self._pyvista_grid = grid

        return self._pyvista_grid

    def render_3d(
        self,
        signal_name: str = "power",
        colormap: str = "plasma",
        threshold: float = 0.1,
        opacity: float = 1.0,
        show_scalar_bar: bool = True,
        title: Optional[str] = None,
        auto_show: bool = False,
    ) -> pv.Plotter:
        """
        Render 3D voxel visualization.

        Args:
            signal_name: Name of signal to visualize
            colormap: PyVista colormap name
            threshold: Minimum value to show (voxels below this are hidden)
            opacity: Opacity of voxels (0.0 to 1.0)
            show_scalar_bar: Whether to show color bar
            title: Optional title for the plot
            auto_show: If True, automatically call show() (default: False, return plotter for manual display)

        Returns:
            PyVista Plotter object
        """
        if self.voxel_grid is None:
            raise ValueError("No voxel grid set. Call set_voxel_grid() first.")

        try:
            grid = self._create_pyvista_grid()
            signal_array = self.voxel_grid.get_signal_array(signal_name, default=0.0)

            # Limit array size to prevent memory issues
            if signal_array.size >= 1e6:  # 1 million voxels or more
                # Downsample for large grids
                import warnings

                warnings.warn(
                    f"Large grid detected ({signal_array.size} voxels). Consider using higher threshold or lower resolution.",
                    UserWarning,
                )
                # Increase threshold to reduce rendered voxels
                threshold = max(threshold, 0.5)

            grid[signal_name] = signal_array.flatten(order="F")

            plotter = pv.Plotter(notebook=True)

            # Apply threshold to show only non-zero voxels
            threshold_mesh = grid.threshold(threshold)

            # Check if mesh is empty
            if threshold_mesh.n_points == 0:
                plotter.add_text("No data above threshold", font_size=12)
            else:
                plotter.add_mesh(
                    threshold_mesh,
                    scalars=signal_name,
                    cmap=colormap,  # type: ignore[arg-type]
                    opacity=opacity,
                    show_edges=False,
                    show_scalar_bar=show_scalar_bar,
                    scalar_bar_args={"title": signal_name.title()},
                )

            plotter.add_axes()  # type: ignore[call-arg]

            if title:
                plotter.add_text(title, font_size=12)

            if auto_show:
                plotter.show(jupyter_backend="static")

            return plotter
        except Exception as e:
            # Return a minimal plotter with error message
            plotter = pv.Plotter(notebook=True)
            plotter.add_text(f"Error: {str(e)}", font_size=12)
            return plotter

    def render_slice(
        self,
        signal_name: str = "power",
        axis: str = "z",
        position: Optional[float] = None,
        colormap: str = "plasma",
        show_scalar_bar: bool = True,
        title: Optional[str] = None,
        auto_show: bool = False,
    ) -> pv.Plotter:
        """
        Render 2D slice through voxel grid.

        Args:
            signal_name: Name of signal to visualize
            axis: Slice axis ('x', 'y', or 'z')
            position: Position along axis (None = center)
            colormap: PyVista colormap name
            show_scalar_bar: Whether to show color bar
            title: Optional title for the plot
            auto_show: If True, automatically call show() (default: False, return plotter for manual display)

        Returns:
            PyVista Plotter object
        """
        if self.voxel_grid is None:
            raise ValueError("No voxel grid set. Call set_voxel_grid() first.")

        grid = self._create_pyvista_grid()
        signal_array = self.voxel_grid.get_signal_array(signal_name, default=0.0)
        grid[signal_name] = signal_array.flatten(order="F")

        # Determine slice position
        if position is None:
            if axis == "x":
                position = (self.voxel_grid.bbox_min[0] + self.voxel_grid.bbox_max[0]) / 2.0
            elif axis == "y":
                position = (self.voxel_grid.bbox_min[1] + self.voxel_grid.bbox_max[1]) / 2.0
            else:  # z
                position = (self.voxel_grid.bbox_min[2] + self.voxel_grid.bbox_max[2]) / 2.0

        plotter = pv.Plotter(notebook=True)

        # Extract slice
        if axis == "x":
            slice_mesh = grid.slice(normal="x", origin=(position, 0, 0))
        elif axis == "y":
            slice_mesh = grid.slice(normal="y", origin=(0, position, 0))
        else:  # z
            slice_mesh = grid.slice(normal="z", origin=(0, 0, position))

        plotter.add_mesh(
            slice_mesh,
            scalars=signal_name,
            cmap=colormap,  # type: ignore[arg-type]
            show_scalar_bar=show_scalar_bar,
            scalar_bar_args={"title": signal_name.title()},
        )

        # Set camera to view slice
        if axis == "z":
            plotter.camera_position = "xy"  # Top view
        elif axis == "y":
            plotter.camera_position = "xz"  # Front view
        else:  # x
            plotter.camera_position = "yz"  # Side view

        plotter.add_axes()  # type: ignore[call-arg]

        if title:
            plotter.add_text(title, font_size=12)

        if auto_show:
            plotter.show(jupyter_backend="static")

        return plotter

    def render_isosurface(
        self,
        signal_name: str = "power",
        isovalue: Optional[float] = None,
        colormap: str = "plasma",
        opacity: float = 0.7,
        show_scalar_bar: bool = True,
        title: Optional[str] = None,
        auto_show: bool = False,
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
            auto_show: If True, automatically call show() (default: False)

        Returns:
            PyVista Plotter object
        """
        if self.voxel_grid is None:
            raise ValueError("No voxel grid set. Call set_voxel_grid() first.")

        grid = self._create_pyvista_grid()
        signal_array = self.voxel_grid.get_signal_array(signal_name, default=0.0)
        grid[signal_name] = signal_array.flatten(order="F")

        # Determine isovalue
        if isovalue is None:
            non_zero = signal_array[signal_array != 0.0]
            isovalue = float(np.mean(non_zero)) if len(non_zero) > 0 else 0.0

        plotter = pv.Plotter(notebook=True)

        # Extract isosurface
        isosurface = grid.contour([isovalue], scalars=signal_name)

        plotter.add_mesh(
            isosurface,
            scalars=signal_name,
            cmap=colormap,  # type: ignore[arg-type]
            opacity=opacity,
            show_edges=True,
            show_scalar_bar=show_scalar_bar,
            scalar_bar_args={"title": signal_name.title()},
        )

        plotter.add_axes()  # type: ignore[call-arg]

        if title:
            plotter.add_text(title, font_size=12)

        if auto_show:
            plotter.show(jupyter_backend="static")

        return plotter

    def render_multi_slice(
        self,
        signal_name: str = "power",
        colormap: str = "plasma",
        show_scalar_bar: bool = True,
        auto_show: bool = False,
    ) -> pv.Plotter:
        """
        Render 3 slice views (X, Y, Z) in a single plotter.

        Args:
            signal_name: Name of signal to visualize
            colormap: PyVista colormap name
            show_scalar_bar: Whether to show color bar
            auto_show: If True, automatically call show() (default: False)

        Returns:
            PyVista Plotter object with 3 subplots
        """
        if self.voxel_grid is None:
            raise ValueError("No voxel grid set. Call set_voxel_grid() first.")

        grid = self._create_pyvista_grid()
        signal_array = self.voxel_grid.get_signal_array(signal_name, default=0.0)
        grid[signal_name] = signal_array.flatten(order="F")

        plotter = pv.Plotter(shape=(2, 2), notebook=True)

        # 3D view (top left)
        plotter.subplot(0, 0)
        threshold = grid.threshold(0.1)
        plotter.add_mesh(threshold, scalars=signal_name, cmap=colormap, show_scalar_bar=False)  # type: ignore[arg-type]
        plotter.add_axes()  # type: ignore[call-arg]
        plotter.add_text("3D View", font_size=10)

        # X slice (top right)
        plotter.subplot(0, 1)
        slice_x = grid.slice(normal="x", origin=(grid.bounds[1], 0, 0))
        plotter.add_mesh(slice_x, scalars=signal_name, cmap=colormap, show_scalar_bar=False)  # type: ignore[arg-type]
        plotter.camera_position = "yz"
        plotter.add_axes()  # type: ignore[call-arg]
        plotter.add_text("X Slice", font_size=10)

        # Y slice (bottom left)
        plotter.subplot(1, 0)
        slice_y = grid.slice(normal="y", origin=(0, grid.bounds[3], 0))
        plotter.add_mesh(slice_y, scalars=signal_name, cmap=colormap, show_scalar_bar=False)  # type: ignore[arg-type]
        plotter.camera_position = "xz"
        plotter.add_axes()  # type: ignore[call-arg]
        plotter.add_text("Y Slice", font_size=10)

        # Z slice (bottom right)
        plotter.subplot(1, 1)
        slice_z = grid.slice(normal="z", origin=(0, 0, grid.bounds[5]))
        plotter.add_mesh(
            slice_z,
            scalars=signal_name,
            cmap=colormap,  # type: ignore[arg-type]
            show_scalar_bar=show_scalar_bar,
            scalar_bar_args={"title": signal_name.title()},
        )
        plotter.camera_position = "xy"
        plotter.add_axes()  # type: ignore[call-arg]
        plotter.add_text("Z Slice", font_size=10)

        if auto_show:
            plotter.show(jupyter_backend="static")

        return plotter
