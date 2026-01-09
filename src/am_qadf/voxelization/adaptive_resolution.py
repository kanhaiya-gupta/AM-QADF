"""
Adaptive Resolution

Spatially and temporally variable resolution within a voxel domain.
Supports different resolutions in different regions and at different time points.
"""

from typing import Optional, Tuple, Dict, List, Any, Callable
import numpy as np
from dataclasses import dataclass, field


@dataclass
class SpatialResolutionMap:
    """Maps spatial regions to resolution levels."""

    regions: List[Tuple[Tuple[float, float, float], Tuple[float, float, float], float]] = field(default_factory=list)
    # Each region: (bbox_min, bbox_max, resolution)
    default_resolution: float = 1.0


@dataclass
class TemporalResolutionMap:
    """Maps temporal regions to resolution levels."""

    time_ranges: List[Tuple[float, float, float]] = field(default_factory=list)
    # Each range: (time_start, time_end, resolution)
    layer_ranges: List[Tuple[int, int, float]] = field(default_factory=list)
    # Each range: (layer_start, layer_end, resolution)
    default_resolution: float = 1.0


class AdaptiveResolutionGrid:
    """
    Voxel grid with spatially and temporally adaptive resolution.

    Supports:
    - Different resolutions in different spatial regions
    - Different resolutions at different time points/layers
    - Automatic resolution selection based on data density
    - Dynamic resolution adjustment
    """

    def __init__(
        self,
        bbox_min: Tuple[float, float, float],
        bbox_max: Tuple[float, float, float],
        base_resolution: float = 1.0,
        spatial_resolution_map: Optional[SpatialResolutionMap] = None,
        temporal_resolution_map: Optional[TemporalResolutionMap] = None,
    ):
        """
        Initialize adaptive resolution grid.

        Args:
            bbox_min: Minimum bounding box corner (x, y, z)
            bbox_max: Maximum bounding box corner (x, y, z)
            base_resolution: Default resolution (mm)
            spatial_resolution_map: Spatial resolution mapping
            temporal_resolution_map: Temporal resolution mapping
        """
        self.bbox_min = np.array(bbox_min)
        self.bbox_max = np.array(bbox_max)
        self.base_resolution = base_resolution
        self.aggregation = "mean"  # Default aggregation method, matching VoxelGrid

        self.spatial_map = spatial_resolution_map or SpatialResolutionMap(default_resolution=base_resolution)
        self.temporal_map = temporal_resolution_map or TemporalResolutionMap(default_resolution=base_resolution)

        # Import VoxelGrid
        try:
            from .voxel_grid import VoxelGrid
        except ImportError:
            import sys
            from pathlib import Path

            current_file = Path(__file__).resolve()
            grid_path = current_file.parent / "voxel_grid.py"
            if grid_path.exists():
                import importlib.util

                spec = importlib.util.spec_from_file_location("voxel_grid", grid_path)
                grid_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(grid_module)
                VoxelGrid = grid_module.VoxelGrid
            else:
                raise ImportError("Could not import VoxelGrid")

        self.VoxelGrid = VoxelGrid

        # Store data points with their spatial and temporal information
        self.points: List[Tuple[float, float, float]] = []
        self.signals: List[Dict[str, float]] = []
        self.timestamps: List[Optional[float]] = []  # Optional timestamps
        self.layer_indices: List[Optional[int]] = []  # Optional layer indices

        # Track which signals are present in the grid
        self.available_signals: set = set()

        # Grids for different resolution regions
        self.region_grids: Dict[str, VoxelGrid] = {}
        self._finalized = False

    def _world_to_voxel_batch(self, points: np.ndarray) -> np.ndarray:
        """
        Vectorized voxel index calculation for all points.

        This method processes multiple points at once using NumPy vectorization,
        providing significant performance improvement over single-point processing.

        For adaptive grids, this uses the base resolution. For more precise
        resolution handling, points should be processed through the finalized grids.

        Args:
            points: Array of points (N, 3) with (x, y, z) coordinates in mm

        Returns:
            Array of voxel indices (N, 3) as integers
        """
        if points.shape[1] != 3:
            raise ValueError(f"Points must be shape (N, 3), got {points.shape}")

        # Use base resolution for batch conversion
        # For adaptive grids, this is an approximation
        # More precise handling would require checking each point's resolution
        normalized = (points - self.bbox_min) / self.base_resolution

        # Convert to integer indices
        indices = np.floor(normalized).astype(int)

        # Clamp to valid range (vectorized)
        # Calculate approximate dims based on base resolution
        size = self.bbox_max - self.bbox_min
        dims = np.ceil(size / self.base_resolution).astype(int)
        # Ensure at least 1 voxel in each dimension
        dims = np.maximum(dims, [1, 1, 1])
        indices = np.clip(indices, [0, 0, 0], dims - 1)

        return indices

    def add_point(
        self,
        x: float,
        y: float,
        z: float,
        signals: Dict[str, float],
        timestamp: Optional[float] = None,
        layer_index: Optional[int] = None,
    ):
        """
        Add point to grid with adaptive resolution.

        Args:
            x, y, z: Point coordinates
            signals: Signal values dictionary
            timestamp: Optional timestamp (seconds)
            layer_index: Optional layer index
        """
        if self._finalized:
            raise ValueError("Cannot add points after finalization. Create a new grid.")

        self.points.append((x, y, z))
        self.signals.append(signals)
        self.timestamps.append(timestamp)
        self.layer_indices.append(layer_index)

        # Track available signals
        self.available_signals.update(signals.keys())

    def get_resolution_for_point(
        self,
        x: float,
        y: float,
        z: float,
        timestamp: Optional[float] = None,
        layer_index: Optional[int] = None,
    ) -> float:
        """
        Get resolution for a specific point based on spatial and temporal maps.

        Args:
            x, y, z: Point coordinates
            timestamp: Optional timestamp
            layer_index: Optional layer index

        Returns:
            Resolution in mm
        """
        # Check spatial resolution map
        resolution = self.spatial_map.default_resolution

        for bbox_min, bbox_max, region_resolution in self.spatial_map.regions:
            if bbox_min[0] <= x <= bbox_max[0] and bbox_min[1] <= y <= bbox_max[1] and bbox_min[2] <= z <= bbox_max[2]:
                resolution = region_resolution
                break

        # Check temporal resolution map
        if timestamp is not None:
            for time_start, time_end, time_resolution in self.temporal_map.time_ranges:
                if time_start <= timestamp <= time_end:
                    # Use finer of spatial and temporal resolutions
                    resolution = min(resolution, time_resolution)
                    break

        if layer_index is not None:
            for (
                layer_start,
                layer_end,
                layer_resolution,
            ) in self.temporal_map.layer_ranges:
                if layer_start <= layer_index <= layer_end:
                    # Use finer of spatial and temporal resolutions
                    resolution = min(resolution, layer_resolution)
                    break

        return resolution

    def finalize(self, adaptive_density: bool = True):
        """
        Finalize grid by creating region-specific grids.

        Args:
            adaptive_density: If True, adjust resolution based on local data density
        """
        if self._finalized:
            return

        # Group points by resolution
        resolution_groups: Dict[float, List[int]] = {}

        for i, point in enumerate(self.points):
            x, y, z = point
            timestamp = self.timestamps[i]
            layer_idx = self.layer_indices[i]

            resolution = self.get_resolution_for_point(x, y, z, timestamp, layer_idx)

            if adaptive_density:
                # Adjust resolution based on local density
                local_density = self._estimate_local_density(x, y, z)
                if local_density > 1.0:  # High density
                    resolution = resolution * 0.5  # Finer resolution
                elif local_density < 0.1:  # Low density
                    resolution = resolution * 2.0  # Coarser resolution

            # Round to reasonable resolution values
            resolution = self._round_resolution(resolution)

            if resolution not in resolution_groups:
                resolution_groups[resolution] = []
            resolution_groups[resolution].append(i)

        # Create grids for each resolution
        for resolution, point_indices in resolution_groups.items():
            # Find bounding box for this resolution group
            points_array = np.array([self.points[i] for i in point_indices])
            if len(points_array) == 0:
                continue

            group_bbox_min = points_array.min(axis=0)
            group_bbox_max = points_array.max(axis=0)

            # Add padding
            padding = resolution * 2
            group_bbox_min = group_bbox_min - padding
            group_bbox_max = group_bbox_max + padding

            # Clamp to overall bounding box
            group_bbox_min = np.maximum(group_bbox_min, self.bbox_min)
            group_bbox_max = np.minimum(group_bbox_max, self.bbox_max)

            # Create grid for this resolution
            grid = self.VoxelGrid(
                bbox_min=tuple(group_bbox_min),
                bbox_max=tuple(group_bbox_max),
                resolution=resolution,
            )

            # Add points to grid
            for idx in point_indices:
                x, y, z = self.points[idx]
                grid.add_point(x, y, z, self.signals[idx])

            grid.finalize()

            # Store grid with resolution as key
            resolution_key = f"res_{resolution:.3f}"
            self.region_grids[resolution_key] = grid

            # Update available signals from grid
            if hasattr(grid, "available_signals"):
                self.available_signals.update(grid.available_signals)

        self._finalized = True

    def _estimate_local_density(self, x: float, y: float, z: float, radius: float = 2.0) -> float:
        """
        Estimate local data density around a point.

        Args:
            x, y, z: Point coordinates
            radius: Search radius (mm)

        Returns:
            Estimated density (points per mm³)
        """
        if not self.points:
            return 0.0

        points_array = np.array(self.points)
        point = np.array([x, y, z])

        # Calculate distances
        distances = np.linalg.norm(points_array - point, axis=1)

        # Count points within radius
        count = np.sum(distances <= radius)

        # Estimate volume (sphere)
        volume = (4.0 / 3.0) * np.pi * (radius**3)

        return count / volume if volume > 0 else 0.0

    def _round_resolution(self, resolution: float) -> float:
        """
        Round resolution to reasonable values.

        Args:
            resolution: Input resolution

        Returns:
            Rounded resolution
        """
        # Round to nearest power of 2 times 0.1
        if resolution <= 0.1:
            return 0.1
        elif resolution <= 0.2:
            return 0.2
        elif resolution <= 0.5:
            return 0.5
        elif resolution <= 1.0:
            return 1.0
        elif resolution <= 2.0:
            return 2.0
        elif resolution <= 5.0:
            return 5.0
        else:
            return 10.0

    def _build_voxel_grid_batch(self, voxel_data: Dict[Tuple[int, int, int], Dict[str, Any]]):
        """
        Build voxel grid structure from pre-aggregated data.

        For adaptive grids, this adds points to the grid which will be
        processed during finalization. Since adaptive grids use multiple
        region grids, we store the data points and process them during finalize().

        Args:
            voxel_data: Dictionary mapping voxel keys to dictionaries containing:
                - 'signals': Dict[str, float] - aggregated signal values
                - 'count': int - number of points contributing to this voxel
        """
        if self._finalized:
            raise ValueError("Cannot add data to finalized adaptive grid")

        # Convert voxel indices back to world coordinates and add as points
        for voxel_key, data in voxel_data.items():
            i, j, k = voxel_key
            # Convert to world coordinates (voxel center)
            # Use base resolution for approximation
            x = self.bbox_min[0] + (i + 0.5) * self.base_resolution
            y = self.bbox_min[1] + (j + 0.5) * self.base_resolution
            z = self.bbox_min[2] + (k + 0.5) * self.base_resolution

            # Add point with aggregated signals
            self.add_point(x, y, z, data.get("signals", {}))

    def get_signal_array(
        self,
        signal_name: str,
        target_resolution: Optional[float] = None,
        default: float = 0.0,
    ) -> np.ndarray:
        """
        Get signal array at a target resolution.

        Args:
            signal_name: Name of signal
            target_resolution: Target resolution (if None, uses finest available)
            default: Default value for empty voxels

        Returns:
            Signal array (may need to be interpolated from multiple grids)
        """
        if not self._finalized:
            raise ValueError("Grid must be finalized before getting signal arrays")

        if target_resolution is None:
            # Use finest resolution
            resolutions = [float(k.split("_")[1]) for k in self.region_grids.keys()]
            target_resolution = min(resolutions) if resolutions else self.base_resolution

        # Find closest resolution grid
        best_key = None
        best_diff = float("inf")

        for key in self.region_grids.keys():
            grid_resolution = float(key.split("_")[1])
            diff = abs(grid_resolution - target_resolution)
            if diff < best_diff:
                best_diff = diff
                best_key = key

        if best_key is None:
            raise ValueError("No grids available")

        grid = self.region_grids[best_key]
        return grid.get_signal_array(signal_name, default=default)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics for the adaptive grid.

        Returns:
            Statistics dictionary
        """
        if not self._finalized:
            return {"finalized": False}

        stats = {
            "finalized": True,
            "num_regions": len(self.region_grids),
            "resolutions": [],
            "total_points": len(self.points),
            "available_signals": set(),
        }

        for key, grid in self.region_grids.items():
            resolution = float(key.split("_")[1])
            grid_stats = grid.get_statistics()

            stats["resolutions"].append(
                {
                    "resolution": resolution,
                    "filled_voxels": grid_stats.get("filled_voxels", 0),
                    "signals": grid_stats.get("available_signals", set()),
                }
            )

            stats["available_signals"].update(grid_stats.get("available_signals", set()))

        stats["available_signals"] = list(stats["available_signals"])

        return stats

    def add_spatial_region(
        self,
        bbox_min: Tuple[float, float, float],
        bbox_max: Tuple[float, float, float],
        resolution: float,
    ):
        """
        Add a spatial region with specific resolution.

        Args:
            bbox_min: Minimum bounding box corner
            bbox_max: Maximum bounding box corner
            resolution: Resolution for this region (mm)
        """
        self.spatial_map.regions.append((bbox_min, bbox_max, resolution))

    def add_temporal_range(self, time_start: float, time_end: float, resolution: float):
        """
        Add a temporal range with specific resolution.

        Args:
            time_start: Start time (seconds)
            time_end: End time (seconds)
            resolution: Resolution for this time range (mm)
        """
        self.temporal_map.time_ranges.append((time_start, time_end, resolution))

    def add_layer_range(self, layer_start: int, layer_end: int, resolution: float):
        """
        Add a layer range with specific resolution.

        Args:
            layer_start: Start layer index
            layer_end: End layer index
            resolution: Resolution for this layer range (mm)
        """
        self.temporal_map.layer_ranges.append((layer_start, layer_end, resolution))


def create_spatial_only_grid(
    adaptive_grid: "AdaptiveResolutionGrid",
) -> "AdaptiveResolutionGrid":
    """
    Create a spatial-only adaptive grid from an existing adaptive grid.

    This extracts only the spatial resolution regions, ignoring temporal information.
    Useful for comparing spatial vs temporal vs combined adaptive resolution.

    Args:
        adaptive_grid: Source adaptive resolution grid

    Returns:
        New AdaptiveResolutionGrid with only spatial resolution mapping
    """
    # Create new grid with only spatial mapping
    spatial_only = AdaptiveResolutionGrid(
        bbox_min=tuple(adaptive_grid.bbox_min),
        bbox_max=tuple(adaptive_grid.bbox_max),
        base_resolution=adaptive_grid.base_resolution,
        spatial_resolution_map=adaptive_grid.spatial_map,
        temporal_resolution_map=None,  # No temporal mapping
    )

    # Copy all points (but temporal info will be ignored)
    for i, point in enumerate(adaptive_grid.points):
        signals = adaptive_grid.signals[i] if i < len(adaptive_grid.signals) else {}
        # Don't pass timestamp or layer_index to spatial-only grid
        spatial_only.add_point(point[0], point[1], point[2], signals)

    # Finalize
    spatial_only.finalize(adaptive_density=True)

    return spatial_only


def create_temporal_only_grid(
    adaptive_grid: "AdaptiveResolutionGrid",
) -> "AdaptiveResolutionGrid":
    """
    Create a temporal-only adaptive grid from an existing adaptive grid.

    This extracts only the temporal resolution ranges, ignoring spatial regions.
    Useful for comparing spatial vs temporal vs combined adaptive resolution.

    Args:
        adaptive_grid: Source adaptive resolution grid

    Returns:
        New AdaptiveResolutionGrid with only temporal resolution mapping
    """
    # Create new grid with only temporal mapping
    temporal_only = AdaptiveResolutionGrid(
        bbox_min=tuple(adaptive_grid.bbox_min),
        bbox_max=tuple(adaptive_grid.bbox_max),
        base_resolution=adaptive_grid.base_resolution,
        spatial_resolution_map=None,  # No spatial mapping
        temporal_resolution_map=adaptive_grid.temporal_map,
    )

    # Copy all points with temporal information
    for i, point in enumerate(adaptive_grid.points):
        signals = adaptive_grid.signals[i] if i < len(adaptive_grid.signals) else {}
        timestamp = adaptive_grid.timestamps[i] if i < len(adaptive_grid.timestamps) else None
        layer_idx = adaptive_grid.layer_indices[i] if i < len(adaptive_grid.layer_indices) else None
        temporal_only.add_point(
            point[0],
            point[1],
            point[2],
            signals,
            timestamp=timestamp,
            layer_index=layer_idx,
        )

    # Finalize
    temporal_only.finalize(adaptive_density=True)

    return temporal_only
