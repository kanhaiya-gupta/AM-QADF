"""
Spatial Anomaly Visualization

Provides 3D voxel visualization of anomalies including:
- 3D voxel anomaly maps
- Slice views with anomaly overlays
- Anomaly heatmaps
- Geometry overlay on STL models
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

try:
    import pyvista as pv

    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    pv = None

logger = logging.getLogger(__name__)


class SpatialAnomalyVisualizer:
    """
    Visualizer for spatial anomaly detection results in 3D voxel space.
    """

    def __init__(self):
        """Initialize spatial anomaly visualizer."""
        self.voxel_coordinates: Optional[np.ndarray] = None
        self.anomaly_scores: Optional[np.ndarray] = None
        self.anomaly_labels: Optional[np.ndarray] = None

    def set_data(
        self,
        voxel_coordinates: np.ndarray,
        anomaly_scores: np.ndarray,
        anomaly_labels: Optional[np.ndarray] = None,
    ) -> None:
        """
        Set data for visualization.

        Args:
            voxel_coordinates: Array of shape (n_samples, 3) with (x, y, z) coordinates
            anomaly_scores: Array of shape (n_samples,) with anomaly scores
            anomaly_labels: Optional array of shape (n_samples,) with boolean anomaly labels
        """
        self.voxel_coordinates = np.asarray(voxel_coordinates)
        self.anomaly_scores = np.asarray(anomaly_scores)
        if anomaly_labels is not None:
            self.anomaly_labels = np.asarray(anomaly_labels, dtype=bool)
        else:
            self.anomaly_labels = None

    def create_voxel_grid(self, resolution: float = 0.1) -> Optional[np.ndarray]:
        """
        Create a 3D voxel grid from coordinates.

        Args:
            resolution: Voxel resolution (size of each voxel)

        Returns:
            3D array with anomaly scores at each voxel, or None if data not set
        """
        if self.voxel_coordinates is None or self.anomaly_scores is None:
            logger.warning("Data not set. Call set_data() first.")
            return None

        # Get bounds
        min_coords = np.min(self.voxel_coordinates, axis=0)
        max_coords = np.max(self.voxel_coordinates, axis=0)

        # Calculate grid dimensions
        dims = ((max_coords - min_coords) / resolution).astype(int) + 1

        # Create 3D grid
        voxel_grid = np.zeros(dims)
        voxel_count = np.zeros(dims)

        # Map coordinates to grid indices
        for i, coord in enumerate(self.voxel_coordinates):
            idx = ((coord - min_coords) / resolution).astype(int)
            # Clamp to valid range
            idx = np.clip(idx, 0, np.array(dims) - 1)
            voxel_grid[tuple(idx)] += self.anomaly_scores[i]
            voxel_count[tuple(idx)] += 1

        # Average scores where multiple points map to same voxel
        mask = voxel_count > 0
        voxel_grid[mask] /= voxel_count[mask]

        return voxel_grid

    def plot_3d_voxel_map(
        self,
        threshold: Optional[float] = None,
        colormap: str = "Reds",
        opacity: float = 0.8,
        title: str = "3D Anomaly Map",
        show: bool = True,
    ) -> Optional[Any]:
        """
        Create 3D voxel map visualization.

        Args:
            threshold: Optional threshold to show only anomalies above this value
            colormap: Colormap name
            opacity: Opacity of voxels
            title: Plot title
            show: Whether to show the plot immediately

        Returns:
            PyVista plotter if available, matplotlib figure otherwise
        """
        if self.voxel_coordinates is None or self.anomaly_scores is None:
            logger.warning("Data not set. Call set_data() first.")
            return None

        if PYVISTA_AVAILABLE:
            return self._plot_3d_pyvista(threshold, colormap, opacity, title, show)
        else:
            return self._plot_3d_matplotlib(threshold, colormap, title, show)

    def _plot_3d_pyvista(
        self,
        threshold: Optional[float],
        colormap: str,
        opacity: float,
        title: str,
        show: bool,
    ) -> pv.Plotter:
        """Create 3D visualization using PyVista."""
        plotter = pv.Plotter(notebook=True)

        # Create point cloud
        points = self.voxel_coordinates
        scores = self.anomaly_scores

        # Apply threshold if specified
        if threshold is not None:
            mask = scores >= threshold
            points = points[mask]
            scores = scores[mask]

        if len(points) == 0:
            plotter.add_text("No anomalies to display", font_size=12)
            return plotter

        # Create point cloud
        point_cloud = pv.PolyData(points)
        point_cloud["anomaly_score"] = scores

        # Add to plotter
        plotter.add_mesh(
            point_cloud,
            scalars="anomaly_score",
            cmap=colormap,  # type: ignore[arg-type]
            opacity=opacity,
            point_size=5.0,
            render_points_as_spheres=True,
        )

        plotter.add_axes()  # type: ignore[call-arg]
        plotter.add_text(title, font_size=12, position="upper_left")

        if show:
            plotter.show(jupyter_backend="static")

        return plotter

    def _plot_3d_matplotlib(self, threshold: Optional[float], colormap: str, title: str, show: bool) -> plt.Figure:
        """Create 3D visualization using matplotlib."""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        scores = self.anomaly_scores
        coords = self.voxel_coordinates

        # Apply threshold if specified
        if threshold is not None:
            mask = scores >= threshold
            scores = scores[mask]
            coords = coords[mask]

        if len(coords) == 0:
            ax.text(0.5, 0.5, 0.5, "No anomalies to display", transform=ax.transAxes)
            return fig

        # Scatter plot
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            c=scores,
            cmap=colormap,
            s=20,
            alpha=0.6,
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(title)
        plt.colorbar(scatter, ax=ax, label="Anomaly Score")

        if show:
            plt.show()

        return fig

    def plot_slice_view(
        self,
        axis: str = "z",
        position: Optional[float] = None,
        colormap: str = "Reds",
        title: Optional[str] = None,
        show: bool = True,
    ) -> plt.Figure:
        """
        Create 2D slice view with anomaly overlays.

        Args:
            axis: Axis to slice along ('x', 'y', or 'z')
            position: Position along axis (if None, use median)
            colormap: Colormap name
            title: Plot title
            show: Whether to show the plot immediately

        Returns:
            Matplotlib figure
        """
        if self.voxel_coordinates is None or self.anomaly_scores is None:
            logger.warning("Data not set. Call set_data() first.")
            return None

        axis_map = {"x": 0, "y": 1, "z": 2}
        if axis not in axis_map:
            raise ValueError(f"Axis must be 'x', 'y', or 'z', got {axis}")

        axis_idx = axis_map[axis]
        other_axes = [i for i in range(3) if i != axis_idx]

        # Determine slice position
        if position is None:
            position = np.median(self.voxel_coordinates[:, axis_idx])

        # Find points near slice
        tolerance = 0.05  # 5% tolerance
        coord_range = np.ptp(self.voxel_coordinates[:, axis_idx])
        mask = np.abs(self.voxel_coordinates[:, axis_idx] - position) < (tolerance * coord_range)

        if not np.any(mask):
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(
                0.5,
                0.5,
                f"No data at {axis}={position:.2f}",
                transform=ax.transAxes,
                ha="center",
            )
            return fig

        slice_coords = self.voxel_coordinates[mask]
        slice_scores = self.anomaly_scores[mask]

        # Create scatter plot
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(
            slice_coords[:, other_axes[0]],
            slice_coords[:, other_axes[1]],
            c=slice_scores,
            cmap=colormap,
            s=50,
            alpha=0.7,
            edgecolors="black",
            linewidths=0.5,
        )

        ax.set_xlabel(f'{["X", "Y", "Z"][other_axes[0]]} (mm)')
        ax.set_ylabel(f'{["X", "Y", "Z"][other_axes[1]]} (mm)')
        ax.set_title(title or f"Anomaly Slice View: {axis.upper()} = {position:.2f}")
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label="Anomaly Score")

        if show:
            plt.show()

        return fig

    def plot_anomaly_heatmap(
        self,
        resolution: float = 0.1,
        colormap: str = "Reds",
        title: str = "Anomaly Heatmap",
        show: bool = True,
    ) -> plt.Figure:
        """
        Create 2D heatmap of anomaly density.

        Args:
            resolution: Spatial resolution for heatmap
            colormap: Colormap name
            title: Plot title
            show: Whether to show the plot immediately

        Returns:
            Matplotlib figure
        """
        if self.voxel_coordinates is None or self.anomaly_scores is None:
            logger.warning("Data not set. Call set_data() first.")
            return None

        # Project to 2D (use X-Y plane, average over Z)
        x_coords = self.voxel_coordinates[:, 0]
        y_coords = self.voxel_coordinates[:, 1]

        # Create 2D histogram
        x_bins = np.arange(x_coords.min(), x_coords.max() + resolution, resolution)
        y_bins = np.arange(y_coords.min(), y_coords.max() + resolution, resolution)

        heatmap, x_edges, y_edges = np.histogram2d(x_coords, y_coords, bins=[x_bins, y_bins], weights=self.anomaly_scores)

        # Normalize by count
        count_map, _, _ = np.histogram2d(x_coords, y_coords, bins=[x_bins, y_bins])
        mask = count_map > 0
        heatmap[mask] /= count_map[mask]

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(
            heatmap.T,
            origin="lower",
            extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
            cmap=colormap,
            aspect="auto",
            interpolation="bilinear",
        )

        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label="Average Anomaly Score")

        if show:
            plt.show()

        return fig
