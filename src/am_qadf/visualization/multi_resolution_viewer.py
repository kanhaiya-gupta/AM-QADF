"""
Multi-Resolution Viewer

LOD-based rendering for multi-resolution voxel grids.
Adaptive resolution selection for performance optimization.
"""

from typing import Optional, Dict, Any, Tuple
import numpy as np

try:
    import pyvista as pv

    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    pv = None


class MultiResolutionViewer:
    """
    Viewer for multi-resolution voxel grids with LOD support.

    Automatically selects appropriate resolution based on:
    - View distance
    - Performance requirements
    - Data density
    - Available memory
    """

    def __init__(self, multi_resolution_grid=None, performance_mode: str = "balanced"):
        """
        Initialize multi-resolution viewer.

        Args:
            multi_resolution_grid: MultiResolutionGrid object
            performance_mode: Performance mode ('fast', 'balanced', 'quality')
        """
        if not PYVISTA_AVAILABLE:
            raise ImportError("PyVista is required. Install with: pip install pyvista")

        self.multi_resolution_grid = multi_resolution_grid

        # Import ResolutionSelector
        try:
            from ..voxelization.multi_resolution import ResolutionSelector
        except ImportError:
            import sys
            from pathlib import Path

            current_file = Path(__file__).resolve()
            mr_path = current_file.parent.parent / "voxelization" / "multi_resolution.py"
            if mr_path.exists():
                import importlib.util

                spec = importlib.util.spec_from_file_location("multi_resolution", mr_path)
                mr_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mr_module)
                ResolutionSelector = mr_module.ResolutionSelector  # type: ignore[assignment]
            else:
                raise ImportError("Could not import ResolutionSelector")

        self.resolution_selector = ResolutionSelector(performance_mode=performance_mode)
        self._current_level = 0
        self._view_parameters: Dict[str, Any] = {}

    def set_view_parameters(
        self,
        distance: Optional[float] = None,
        zoom: Optional[float] = None,
        region_size: Optional[float] = None,
    ):
        """
        Set view parameters for adaptive resolution selection.

        Args:
            distance: View distance (mm)
            zoom: Zoom level
            region_size: Size of viewed region (mm)
        """
        if distance is not None:
            self._view_parameters["distance"] = distance
        if zoom is not None:
            self._view_parameters["zoom"] = zoom
        if region_size is not None:
            self._view_parameters["region_size"] = region_size

    def select_resolution(
        self,
        method: str = "auto",
        target_level: Optional[int] = None,
        view_distance: Optional[float] = None,
    ) -> int:
        """
        Select appropriate resolution level.

        Args:
            method: Selection method ('auto', 'manual', 'performance', 'view')
            target_level: Manual level selection (if method='manual')
            view_distance: View distance for view-based selection

        Returns:
            Selected level index
        """
        if self.multi_resolution_grid is None:
            raise ValueError("MultiResolutionGrid not set")

        if method == "manual" and target_level is not None:
            self._current_level = target_level
        elif method == "view" or (method == "auto" and view_distance is not None):
            if view_distance is None:
                view_distance = self._view_parameters.get("distance", 100.0)
            self._current_level = self.multi_resolution_grid.get_level_for_view_distance(view_distance)
        elif method == "performance":
            # Estimate data density
            grid = self.multi_resolution_grid.get_level(0)
            if grid:
                num_points = len(grid.voxels) if hasattr(grid, "voxels") else 0
                bbox_size = np.prod(
                    [self.multi_resolution_grid.bbox_max[i] - self.multi_resolution_grid.bbox_min[i] for i in range(3)]
                )
                density = num_points / bbox_size if bbox_size > 0 else 0.0

                self._current_level = self.resolution_selector.select_for_data_density(self.multi_resolution_grid, density)
        else:  # auto
            # Use view parameters if available
            if self._view_parameters:
                self._current_level = self.resolution_selector.select_for_view(
                    self.multi_resolution_grid, self._view_parameters
                )
            else:
                # Default to medium level
                self._current_level = self.multi_resolution_grid.num_levels // 2

        return self._current_level

    def render_3d(
        self,
        signal_name: str = "power",
        level: Optional[int] = None,
        colormap: str = "plasma",
        threshold: Optional[float] = None,
        auto_select_level: bool = True,
        view_distance: Optional[float] = None,
        adaptive_threshold: bool = True,
        auto_show: bool = False,
    ) -> pv.Plotter:
        """
        Render 3D visualization with adaptive resolution.

        Args:
            signal_name: Name of signal to visualize
            level: Specific level to use (if None, auto-selects)
            colormap: PyVista colormap name
            threshold: Minimum value to show (if None, uses adaptive threshold)
            auto_select_level: Whether to auto-select level based on view
            view_distance: View distance for level selection
            adaptive_threshold: If True, adjusts threshold based on resolution level
            auto_show: If True, automatically call show() (default: False)

        Returns:
            PyVista Plotter object
        """
        if self.multi_resolution_grid is None:
            raise ValueError("MultiResolutionGrid not set")

        # Select resolution level
        if level is None and auto_select_level:
            if view_distance is not None:
                level = self.select_resolution(method="view", view_distance=view_distance)
            else:
                level = self.select_resolution(method="auto")
        elif level is None:
            level = self._current_level

        # Get grid for selected level
        grid = self.multi_resolution_grid.get_level(level)
        if grid is None:
            raise ValueError(f"Level {level} not found")

        # Adaptive threshold: finer resolutions need lower threshold to show sparse data
        if threshold is None or adaptive_threshold:
            resolution = self.multi_resolution_grid.get_resolution(level)
            stats = grid.get_statistics()
            filled_voxels = stats.get("filled_voxels", 0)
            total_voxels = np.prod(grid.dims)
            fill_ratio = filled_voxels / total_voxels if total_voxels > 0 else 0.0

            if threshold is None:
                # Adaptive threshold based on fill ratio and resolution
                if fill_ratio < 0.1:  # Less than 10% filled (sparse data)
                    # Use very low threshold for sparse data
                    threshold = 0.01
                elif fill_ratio < 0.3:  # Less than 30% filled
                    threshold = 0.05
                else:
                    threshold = 0.1
            elif adaptive_threshold:
                # Adjust provided threshold based on resolution
                # Finer resolutions (smaller voxels) = lower threshold
                base_threshold = threshold
                # Scale threshold inversely with resolution (finer = lower threshold)
                resolution_factor = resolution / self.multi_resolution_grid.base_resolution
                threshold = base_threshold / max(resolution_factor, 1.0)
                threshold = max(threshold, 0.01)  # Don't go below 0.01

        # Use VoxelRenderer for actual rendering
        try:
            from .voxel_renderer import VoxelRenderer
        except ImportError:
            import sys
            from pathlib import Path

            current_file = Path(__file__).resolve()
            renderer_path = current_file.parent / "voxel_renderer.py"
            if renderer_path.exists():
                import importlib.util

                spec = importlib.util.spec_from_file_location("voxel_renderer", renderer_path)
                renderer_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(renderer_module)
                VoxelRenderer = renderer_module.VoxelRenderer  # type: ignore[assignment]
            else:
                raise ImportError("Could not import VoxelRenderer")

        renderer = VoxelRenderer(voxel_grid=grid)

        # Calculate fill ratio for title
        stats = grid.get_statistics()
        filled_voxels = stats.get("filled_voxels", 0)
        total_voxels = np.prod(grid.dims)
        fill_ratio = filled_voxels / total_voxels if total_voxels > 0 else 0.0

        return renderer.render_3d(
            signal_name=signal_name,
            colormap=colormap,
            threshold=threshold,
            title=f"3D View - {signal_name.title()} (Level {level}, Res: {self.multi_resolution_grid.get_resolution(level):.2f} mm, Fill: {fill_ratio*100:.1f}%)",
            auto_show=auto_show,
        )

    def render_slice(
        self,
        signal_name: str = "power",
        axis: str = "z",
        position: Optional[float] = None,
        level: Optional[int] = None,
        colormap: str = "plasma",
        auto_select_level: bool = True,
        auto_show: bool = False,
    ) -> pv.Plotter:
        """
        Render slice view with adaptive resolution.

        Args:
            signal_name: Name of signal to visualize
            axis: Slice axis ('x', 'y', 'z')
            position: Position along axis
            level: Specific level to use (if None, auto-selects)
            colormap: PyVista colormap name
            auto_select_level: Whether to auto-select level
            auto_show: If True, automatically call show() (default: False)

        Returns:
            PyVista Plotter object
        """
        if self.multi_resolution_grid is None:
            raise ValueError("MultiResolutionGrid not set")

        # Select resolution level
        if level is None and auto_select_level:
            level = self.select_resolution(method="auto")
        elif level is None:
            level = self._current_level

        # Get grid for selected level
        grid = self.multi_resolution_grid.get_level(level)
        if grid is None:
            raise ValueError(f"Level {level} not found")

        # Use VoxelRenderer for actual rendering
        try:
            from .voxel_renderer import VoxelRenderer
        except ImportError:
            import sys
            from pathlib import Path

            current_file = Path(__file__).resolve()
            renderer_path = current_file.parent / "voxel_renderer.py"
            if renderer_path.exists():
                import importlib.util

                spec = importlib.util.spec_from_file_location("voxel_renderer", renderer_path)
                renderer_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(renderer_module)
                VoxelRenderer = renderer_module.VoxelRenderer  # type: ignore[assignment]
            else:
                raise ImportError("Could not import VoxelRenderer")

        renderer = VoxelRenderer(voxel_grid=grid)

        return renderer.render_slice(
            signal_name=signal_name,
            axis=axis,
            position=position,
            colormap=colormap,
            title=f"{axis.upper()} Slice - {signal_name.title()} (Level {level})",
            auto_show=auto_show,
        )

    def get_level_info(self) -> Dict[int, Dict[str, Any]]:
        """
        Get information about all resolution levels.

        Returns:
            Dictionary mapping level index to level information
        """
        if self.multi_resolution_grid is None:
            return {}

        info = {}
        for level in range(self.multi_resolution_grid.num_levels):
            grid = self.multi_resolution_grid.get_level(level)
            if grid:
                stats = grid.get_statistics()
                info[level] = {
                    "resolution": self.multi_resolution_grid.get_resolution(level),
                    "dimensions": (grid.dims.tolist() if hasattr(grid.dims, "tolist") else list(grid.dims)),
                    "num_voxels": int(np.prod(grid.dims)),
                    "filled_voxels": stats.get("filled_voxels", 0),
                    "available_signals": (list(grid.available_signals) if hasattr(grid, "available_signals") else []),
                }

        return info
