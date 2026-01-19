"""
Multi-Resolution Voxel Grids

Hierarchical voxel grids with level-of-detail (LOD) support.
Enables variable resolution and efficient memory usage.
"""

from typing import Optional, Tuple, Dict, List, Any, TYPE_CHECKING
import numpy as np
from enum import Enum

if TYPE_CHECKING:
    from .voxel_grid import VoxelGrid


class ResolutionLevel(Enum):
    """Resolution levels for hierarchical grids."""

    COARSE = "coarse"  # Low resolution, fast
    MEDIUM = "medium"  # Medium resolution, balanced
    FINE = "fine"  # High resolution, detailed
    ULTRA_FINE = "ultra_fine"  # Very high resolution, maximum detail


class MultiResolutionGrid:
    """
    Hierarchical voxel grid with multiple resolution levels.

    Supports:
    - Multiple resolution levels (coarse to fine)
    - Level-of-detail (LOD) selection
    - Efficient memory usage
    - Adaptive resolution based on data density
    """

    def __init__(
        self,
        bbox_min: Tuple[float, float, float],
        bbox_max: Tuple[float, float, float],
        base_resolution: float = 1.0,
        num_levels: int = 3,
        level_ratio: float = 2.0,
    ):
        """
        Initialize multi-resolution grid.

        Args:
            bbox_min: Minimum bounding box corner (x, y, z)
            bbox_max: Maximum bounding box corner (x, y, z)
            base_resolution: Base resolution for finest level (mm)
            num_levels: Number of resolution levels
            level_ratio: Resolution ratio between levels (e.g., 2.0 = each level is 2x coarser)
        """
        # Validate inputs
        if base_resolution <= 0:
            raise ValueError("Base resolution must be greater than 0")
        
        if num_levels < 1:
            raise ValueError("Number of levels must be at least 1")
        
        if level_ratio <= 0:
            raise ValueError("Level ratio must be greater than 0")
        
        bbox_min_arr = np.array(bbox_min)
        bbox_max_arr = np.array(bbox_max)
        
        if np.any(bbox_max_arr <= bbox_min_arr):
            raise ValueError("bbox_max must be greater than bbox_min in all dimensions")
        
        self.bbox_min = bbox_min
        self.bbox_max = bbox_max
        self.base_resolution = base_resolution
        self.num_levels = num_levels
        self.level_ratio = level_ratio

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

        # Create grids for each level
        self.grids: Dict[int, VoxelGrid] = {}
        self.resolutions: Dict[int, float] = {}

        for level in range(num_levels):
            resolution = base_resolution * (level_ratio ** (num_levels - 1 - level))
            self.resolutions[level] = resolution

            grid = VoxelGrid(
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                resolution=resolution,
                aggregation="mean",
            )
            self.grids[level] = grid

    def get_resolution(self, level: int) -> float:
        """
        Get resolution for a specific level.

        Args:
            level: Resolution level (0 = coarsest, num_levels-1 = finest)

        Returns:
            Resolution in mm
        """
        return self.resolutions.get(level, self.base_resolution)

    def get_level(self, level: int) -> "VoxelGrid":
        """
        Get voxel grid for a specific level.

        Args:
            level: Resolution level

        Returns:
            VoxelGrid object
        """
        return self.grids.get(level)

    def add_point(
        self,
        x: float,
        y: float,
        z: float,
        signals: Dict[str, float],
        level: Optional[int] = None,
    ):
        """
        Add point to grid(s).

        Args:
            x, y, z: Point coordinates
            signals: Signal values dictionary
            level: Specific level to add to (if None, adds to all levels)
        """
        if level is not None:
            if level in self.grids:
                self.grids[level].add_point(x, y, z, signals)
        else:
            # Add to all levels
            for grid in self.grids.values():
                grid.add_point(x, y, z, signals)

    def finalize(self):
        """Finalize all grids."""
        for grid in self.grids.values():
            grid.finalize()

    def get_signal_array(self, signal_name: str, level: int = 0, default: float = 0.0) -> np.ndarray:
        """
        Get signal array for a specific level.

        Args:
            signal_name: Name of signal
            level: Resolution level
            default: Default value for empty voxels

        Returns:
            Signal array
        """
        grid = self.get_level(level)
        if grid is None:
            raise ValueError(f"Level {level} not found")

        return grid.get_signal_array(signal_name, default=default)

    def get_statistics(self, level: Optional[int] = None) -> Dict[str, Any]:
        """
        Get statistics for a specific level or aggregate across all levels.

        Args:
            level: Resolution level (if None, returns aggregate statistics across all levels)

        Returns:
            Statistics dictionary
        """
        if level is not None:
            # Return statistics for a specific level
            grid = self.get_level(level)
            if grid is None:
                raise ValueError(f"Level {level} not found")
            return grid.get_statistics()
        else:
            # Return aggregate statistics across all levels
            if not self.grids:
                return {
                    "dimensions": None,
                    "total_voxels": 0,
                    "filled_voxels": 0,
                    "num_levels": 0,
                }
            
            # Get finest level for dimensions
            finest_level = max(self.grids.keys())
            finest_grid = self.grids[finest_level]
            finest_stats = finest_grid.get_statistics()
            
            # Sum total voxels and filled voxels across all levels
            total_voxels = 0
            filled_voxels = 0
            
            for grid in self.grids.values():
                grid_stats = grid.get_statistics()
                total_voxels += grid_stats.get("total_voxels", 0)
                filled_voxels += grid_stats.get("filled_voxels", 0)
            
            return {
                "dimensions": finest_stats.get("dimensions"),
                "total_voxels": int(total_voxels),
                "filled_voxels": int(filled_voxels),
                "num_levels": len(self.grids),
                "base_resolution": self.base_resolution,
                "level_ratio": self.level_ratio,
                "resolutions": {level: self.resolutions[level] for level in self.grids.keys()},
            }

    def select_appropriate_level(self, target_resolution: float, prefer_coarse: bool = False) -> int:
        """
        Select appropriate resolution level based on target resolution.

        Args:
            target_resolution: Desired resolution (mm)
            prefer_coarse: If True, prefer coarser level if close

        Returns:
            Best matching level index
        """
        best_level = 0
        best_diff = float("inf")

        # Sort levels by index to ensure consistent ordering
        sorted_levels = sorted(self.resolutions.items())

        for level, resolution in sorted_levels:
            diff = abs(resolution - target_resolution)

            if prefer_coarse and resolution <= target_resolution:
                # Prefer coarser level that's still acceptable
                if diff < best_diff:
                    best_diff = diff
                    best_level = level
            elif not prefer_coarse:
                # Prefer closest resolution, but if tied prefer finer (higher index)
                if diff < best_diff or (diff == best_diff and level > best_level):
                    best_diff = diff
                    best_level = level

        return best_level

    def get_level_for_view_distance(
        self,
        view_distance: float,
        min_resolution: float = 0.1,
        max_resolution: float = 5.0,
    ) -> int:
        """
        Select resolution level based on view distance.

        Closer views use finer resolution, distant views use coarser.

        Args:
            view_distance: Distance from camera to object (mm)
            min_resolution: Minimum resolution to use (mm)
            max_resolution: Maximum resolution to use (mm)

        Returns:
            Appropriate level index
        """
        # Simple heuristic: resolution scales with distance
        # Closer = finer resolution, farther = coarser
        target_resolution = np.clip(view_distance * 0.01, min_resolution, max_resolution)  # Scale factor

        return self.select_appropriate_level(target_resolution, prefer_coarse=True)

    def downsample_from_finer(self, source_level: int, target_level: int):
        """
        Downsample data from finer level to coarser level.

        Args:
            source_level: Source (finer) level
            target_level: Target (coarser) level
        """
        if source_level <= target_level:
            raise ValueError("Source level must be finer (higher index) than target level")

        source_grid = self.get_level(source_level)
        target_grid = self.get_level(target_level)

        if source_grid is None or target_grid is None:
            raise ValueError("Invalid level")

        # Get all signals from source
        for signal_name in source_grid.available_signals:
            source_array = source_grid.get_signal_array(signal_name, default=0.0)

            # Simple downsampling: average over blocks
            # This is a simplified approach - full implementation would use proper spatial averaging
            source_dims = source_grid.dims
            target_dims = target_grid.dims

            # Compute downsampling factors
            factors = [source_dims[i] / target_dims[i] if target_dims[i] > 0 else 1.0 for i in range(3)]

            # Downsample by averaging (simplified)
            # In practice, would use proper spatial averaging
            # For now, just copy the structure - actual downsampling would happen during add_point
            pass  # Placeholder - full implementation would do proper downsampling


class ResolutionSelector:
    """
    Select appropriate resolution based on various criteria.
    """

    def __init__(self, performance_mode: str = "balanced"):  # 'fast', 'balanced', 'quality'
        """
        Initialize resolution selector.

        Args:
            performance_mode: Performance mode ('fast', 'balanced', 'quality')
        """
        self.performance_mode = performance_mode

    def select_for_performance(
        self,
        grid: MultiResolutionGrid,
        num_points: int,
        available_memory: Optional[float] = None,  # GB
    ) -> int:
        """
        Select resolution level based on performance requirements.

        Args:
            grid: MultiResolutionGrid object
            num_points: Number of data points
            available_memory: Available memory in GB (if None, estimates)

        Returns:
            Recommended level index
        """
        # Estimate memory requirements for each level
        memory_per_voxel = 8.0 / (1024**3)  # 8 bytes per voxel (float64)

        best_level = 0
        best_score = float("inf")

        for level in range(grid.num_levels):
            resolution = grid.get_resolution(level)
            level_grid = grid.get_level(level)

            if level_grid is None:
                continue

            num_voxels = np.prod(level_grid.dims)
            estimated_memory = num_voxels * memory_per_voxel

            # Score based on performance mode
            if self.performance_mode == "fast":
                # Prefer coarser (lower memory, faster)
                score = estimated_memory + (level * 0.1)  # Penalize finer levels
            elif self.performance_mode == "quality":
                # Prefer finer (higher quality)
                score = -estimated_memory + ((grid.num_levels - level) * 0.1)  # Prefer finer
            else:  # balanced
                # Balance between memory and quality
                score = estimated_memory + (level * 0.05)

            # Check memory constraint
            if available_memory is not None and estimated_memory > available_memory:
                continue  # Skip if exceeds memory

            if score < best_score:
                best_score = score
                best_level = level

        return best_level

    def select_for_data_density(self, grid: MultiResolutionGrid, data_density: float) -> int:  # points per mm³
        """
        Select resolution based on data density.

        Higher density can support finer resolution.

        Args:
            grid: MultiResolutionGrid object
            data_density: Data density (points per mm³)

        Returns:
            Recommended level index
        """
        # Heuristic: resolution should match data density
        # Higher density = can use finer resolution
        # Lower density = should use coarser resolution

        # Estimate appropriate resolution based on density
        # If density is high (> 1 point/mm³), use fine resolution
        # If density is low (< 0.1 points/mm³), use coarse resolution

        if data_density > 1.0:
            # High density - use fine resolution
            return grid.num_levels - 1
        elif data_density > 0.1:
            # Medium density - use medium resolution
            return grid.num_levels // 2
        else:
            # Low density - use coarse resolution
            return 0

    def select_for_view(self, grid: MultiResolutionGrid, view_parameters: Dict[str, Any]) -> int:
        """
        Select resolution based on view parameters.

        Args:
            grid: MultiResolutionGrid object
            view_parameters: Dictionary with view parameters:
                - 'distance': View distance (mm)
                - 'zoom': Zoom level
                - 'region_size': Size of viewed region (mm)

        Returns:
            Recommended level index
        """
        distance = view_parameters.get("distance", 100.0)
        zoom = view_parameters.get("zoom", 1.0)
        region_size = view_parameters.get("region_size", 100.0)

        # Adjust target resolution based on view
        # Closer view or higher zoom = finer resolution needed
        effective_distance = distance / zoom

        return grid.get_level_for_view_distance(effective_distance)
