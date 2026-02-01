"""
Multi-Resolution Voxel Grids - OpenVDB C++ Implementation

Thin Python wrapper for C++ multi-resolution grid implementation.
All core computation is done in C++ OpenVDB. No Python fallback.
"""

from typing import Optional, Tuple, Dict, List, Any, TYPE_CHECKING
import numpy as np
from enum import Enum

if TYPE_CHECKING:
    from .uniform_resolution import VoxelGrid


class ResolutionLevel(Enum):
    """Resolution levels for hierarchical grids."""
    COARSE = "coarse"
    MEDIUM = "medium"
    FINE = "fine"
    ULTRA_FINE = "ultra_fine"


# C++ OpenVDB bindings - REQUIRED (no Python fallback)
try:
    from am_qadf_native import MultiResolutionVoxelGrid, openvdb_to_numpy
    CPP_AVAILABLE = True
except ImportError as e:
    CPP_AVAILABLE = False
    raise ImportError(
        "C++ OpenVDB bindings are required. "
        "Please build am_qadf_native with pybind11 bindings. "
        f"Original error: {e}"
    )


class MultiResolutionGrid:
    """
    Hierarchical voxel grid with multiple resolution levels - C++ wrapper.
    
    This is a thin wrapper around the C++ MultiResolutionVoxelGrid implementation.
    All core computation is done in C++.
    
    NOTE: This wrapper maintains the Python API but uses C++ internally.
    Some methods may need additional conversion utilities.
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
        if base_resolution <= 0:
            raise ValueError("Base resolution must be greater than 0")
        if num_levels < 1:
            raise ValueError("Number of levels must be at least 1")
        if level_ratio <= 0:
            raise ValueError("Level ratio must be greater than 0")
        
        self.bbox_min = bbox_min
        self.bbox_max = bbox_max
        self.base_resolution = base_resolution
        self.num_levels = num_levels
        self.level_ratio = level_ratio
        
        # Calculate resolutions for each level
        self.resolutions: Dict[int, float] = {}
        resolutions_list = []
        for level in range(num_levels):
            resolution = base_resolution * (level_ratio ** (num_levels - 1 - level))
            self.resolutions[level] = resolution
            resolutions_list.append(resolution)
        
        # Initialize C++ OpenVDB grid (required, no fallback)
        if not CPP_AVAILABLE:
            raise ImportError("C++ OpenVDB bindings are required")
        
        self._cpp_grid = MultiResolutionVoxelGrid(resolutions_list, base_resolution)
        
        # Cache Python VoxelGrid wrappers for API compatibility
        # These are created on-demand from C++ grids
        self.grids: Dict[int, "VoxelGrid"] = {}

    def get_resolution(self, level: int) -> float:
        """Get resolution for a specific level."""
        return self.resolutions.get(level, self.base_resolution)

    def get_level(self, level: int) -> "VoxelGrid":
        """
        Get voxel grid for a specific level.
        
        Returns Python VoxelGrid converted from C++ OpenVDB grid.
        Uses OpenVDB C++ only - no Python fallback.
        """
        if not CPP_AVAILABLE:
            raise ImportError("C++ OpenVDB bindings are required")
        if level < 0 or level >= self.num_levels:
            raise ValueError(f"Level {level} not found")
        if level not in self.grids:
            # Get C++ OpenVDB grid for this level
            cpp_grid = self._cpp_grid.get_grid(level)
            if cpp_grid is None:
                raise ValueError(f"Level {level} not found")
            
            # Get resolution for this level
            resolution = self.get_resolution(level)
            
            # Create Python VoxelGrid (which now uses OpenVDB internally)
            from .uniform_resolution import VoxelGrid
            
            # Create grid with appropriate bounding box
            grid = VoxelGrid(
                bbox_min=self.bbox_min,
                bbox_max=self.bbox_max,
                resolution=resolution,
                aggregation="mean"
            )
            
            # Copy values directly from C++ grid to Python grid (C++ - fast)
            # Get the underlying C++ UniformVoxelGrid and copy from source grid
            # This avoids Python loops and numpy conversion overhead
            grid._get_or_create_grid("value").copy_from_grid(cpp_grid)
            
            self.grids[level] = grid
        
        return self.grids[level]

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
        
        Note: C++ MultiResolutionVoxelGrid doesn't have a direct addPoint method.
        Points should be added to individual level grids via get_level() and then
        use prolongate/restrict to propagate between levels.
        """
        if level is not None:
            # Add to specific level
            grid = self.get_level(level)
            grid.add_point(x, y, z, signals)
        else:
            # Add to all levels
            for level_idx in range(self.num_levels):
                grid = self.get_level(level_idx)
                grid.add_point(x, y, z, signals)

    def finalize(self):
        """Finalize all grids."""
        # C++ grids are finalized automatically, but keep for API compatibility
        pass

    def get_grid(self, level: int, signal_name: str = "value"):
        """
        Get OpenVDB grid directly for a specific level - NO conversion!
        
        This is the preferred method - works directly with OpenVDB sparse storage.
        Use this for all operations except Python visualization.
        
        Args:
            level: Resolution level index
            signal_name: Name of the signal (default: "value")
            
        Returns:
            OpenVDB FloatGrid (or None if level/signal doesn't exist)
        """
        if not CPP_AVAILABLE:
            raise ImportError("C++ OpenVDB bindings are required")
        
        cpp_grid = self._cpp_grid.get_grid(level)
        if cpp_grid is None:
            return None
        
        # Get the Python grid wrapper to access the signal
        grid = self.get_level(level)
        if grid is None:
            return None
        
        return grid.get_grid(signal_name)
    
    def get_signal_array(self, signal_name: str, level: int = 0, default: float = 0.0) -> np.ndarray:
        """
        Get signal array for a specific level - ONLY for Python visualization!
        
        ⚠️ WARNING: This converts sparse OpenVDB grid to dense NumPy array.
        This is SLOW for large grids and defeats OpenVDB's efficiency.
        
        Use get_grid() instead for all operations except visualization.
        For ParaView, export .vdb files directly - no conversion needed!
        """
        grid = self.get_level(level)
        if grid is None:
            raise ValueError(f"Level {level} not found")
        return grid.get_signal_array(signal_name, default=default)

    def prolongate(self, from_level: int, to_level: int):
        """
        Prolongate (interpolate) from coarse to fine level.
        
        Uses C++ OpenVDB implementation.
        """
        if not CPP_AVAILABLE:
            raise ImportError("C++ OpenVDB bindings are required")
        
        self._cpp_grid.prolongate(from_level, to_level)
        # Invalidate cached Python grids for affected levels
        if to_level in self.grids:
            del self.grids[to_level]

    def restrict(self, from_level: int, to_level: int):
        """
        Restrict (downsample) from fine to coarse level.
        
        Uses C++ OpenVDB implementation.
        """
        if not CPP_AVAILABLE:
            raise ImportError("C++ OpenVDB bindings are required")
        
        self._cpp_grid.restrict(from_level, to_level)
        # Invalidate cached Python grids for affected levels
        if to_level in self.grids:
            del self.grids[to_level]

    def get_num_levels(self) -> int:
        """Get number of resolution levels."""
        return self.num_levels


class ResolutionSelector:
    """
    Select resolution level for a MultiResolutionGrid based on performance, data density, or view.
    Python-only helper (not in C++); used for visualization and tuning.
    """

    def __init__(self, performance_mode: str = "balanced"):
        self.performance_mode = performance_mode

    def select_for_performance(
        self,
        grid: MultiResolutionGrid,
        num_points: int = 0,
        available_memory: Optional[float] = None,
    ) -> int:
        """Select level for performance mode (fast=coarse, quality=fine, balanced=mid)."""
        n = grid.num_levels
        if n == 0:
            return 0
        if self.performance_mode == "fast":
            return 0
        if self.performance_mode == "quality":
            return n - 1
        return max(0, n // 2)

    def select_for_data_density(self, grid: MultiResolutionGrid, data_density: float) -> int:
        """Select level based on data density (high density -> finer level)."""
        n = grid.num_levels
        if n == 0:
            return 0
        if data_density >= 1.0:
            return min(n - 1, max(0, n - 1))
        return max(0, n // 2)

    def select_for_view(self, grid: MultiResolutionGrid, view_params: Dict[str, Any]) -> int:
        """Select level based on view parameters (distance, zoom, region_size)."""
        n = grid.num_levels
        if n == 0:
            return 0
        distance = view_params.get("distance", 100.0)
        if distance < 50.0:
            return min(n - 1, n // 2 + 1)
        return max(0, n // 2)
