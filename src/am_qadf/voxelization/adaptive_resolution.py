"""
Adaptive Resolution - OpenVDB C++ Implementation

Thin Python wrapper for C++ adaptive resolution grid implementation.
All core computation is done in C++ OpenVDB. No Python fallback.
"""

from typing import Optional, Tuple, Dict, List, Any
import numpy as np
from dataclasses import dataclass, field

# C++ OpenVDB bindings - REQUIRED (no Python fallback)
try:
    from am_qadf_native import AdaptiveResolutionVoxelGrid, openvdb_to_numpy
    CPP_AVAILABLE = True
except ImportError as e:
    CPP_AVAILABLE = False
    raise ImportError(
        "C++ OpenVDB bindings are required. "
        "Please build am_qadf_native with pybind11 bindings. "
        f"Original error: {e}"
    )


# Keep dataclasses for API compatibility
@dataclass
class SpatialResolutionMap:
    """Maps spatial regions to resolution levels."""
    regions: List[Tuple[Tuple[float, float, float], Tuple[float, float, float], float]] = field(default_factory=list)
    default_resolution: float = 1.0


@dataclass
class TemporalResolutionMap:
    """Maps temporal regions to resolution levels."""
    time_ranges: List[Tuple[float, float, float]] = field(default_factory=list)
    layer_ranges: List[Tuple[int, int, float]] = field(default_factory=list)
    default_resolution: float = 1.0


class AdaptiveResolutionGrid:
    """
    Voxel grid with spatially and temporally adaptive resolution - C++ wrapper.
    
    This is a thin wrapper around the C++ AdaptiveResolutionVoxelGrid implementation.
    All core computation is done in C++.
    
    NOTE: This wrapper maintains the Python API but uses C++ internally.
    Some methods may need additional conversion utilities.
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
        if base_resolution <= 0:
            raise ValueError("Base resolution must be greater than 0")
        
        self.bbox_min = np.array(bbox_min)
        self.bbox_max = np.array(bbox_max)
        
        if np.any(self.bbox_max <= self.bbox_min):
            raise ValueError("bbox_max must be greater than bbox_min in all dimensions")
        
        self.base_resolution = base_resolution
        self.aggregation = "mean"
        
        self.spatial_map = spatial_resolution_map or SpatialResolutionMap(default_resolution=base_resolution)
        self.temporal_map = temporal_resolution_map or TemporalResolutionMap(default_resolution=base_resolution)
        
        # Initialize C++ OpenVDB grid (required, no fallback)
        if not CPP_AVAILABLE:
            raise ImportError("C++ OpenVDB bindings are required")
        
        # C++ OpenVDB storage: one AdaptiveResolutionVoxelGrid per signal
        # For multi-signal support, we'll create one grid per signal
        # Note: C++ AdaptiveResolutionVoxelGrid stores one signal per grid
        # We'll manage multiple grids internally
        self._cpp_grids: Dict[str, AdaptiveResolutionVoxelGrid] = {}
        
        # Add spatial regions to C++ grids (will be added when first signal is added)
        self._spatial_regions = self.spatial_map.regions.copy()
        self._temporal_ranges = self.temporal_map.time_ranges.copy()
        self._layer_ranges = self.temporal_map.layer_ranges.copy()
        
        # Track which signals are present
        self.available_signals: set = set()
        self._finalized = False

    def _get_or_create_grid(self, signal_name: str) -> AdaptiveResolutionVoxelGrid:
        """
        Get or create AdaptiveResolutionVoxelGrid for a signal.
        
        One grid per signal (multi-signal support).
        """
        if signal_name not in self._cpp_grids:
            if not CPP_AVAILABLE:
                raise ImportError("C++ OpenVDB bindings are required")
            
            # Create new AdaptiveResolutionVoxelGrid for this signal
            adaptive_grid = AdaptiveResolutionVoxelGrid(self.base_resolution)
            
            # Add spatial regions (C++ binding expects 7 floats: bbox_min_xyz, bbox_max_xyz, resolution)
            for bbox_min_region, bbox_max_region, resolution in self._spatial_regions:
                bbox_min_arr = np.array(bbox_min_region, dtype=np.float32)
                bbox_max_arr = np.array(bbox_max_region, dtype=np.float32)
                adaptive_grid.add_spatial_region(
                    float(bbox_min_arr[0]), float(bbox_min_arr[1]), float(bbox_min_arr[2]),
                    float(bbox_max_arr[0]), float(bbox_max_arr[1]), float(bbox_max_arr[2]),
                    float(resolution),
                )
            
            # Add temporal ranges
            for time_start, time_end, resolution in self._temporal_ranges:
                adaptive_grid.add_temporal_range(time_start, time_end, 0, 0, resolution)
            
            for layer_start, layer_end, resolution in self._layer_ranges:
                adaptive_grid.add_temporal_range(0.0, 0.0, layer_start, layer_end, resolution)
            
            self._cpp_grids[signal_name] = adaptive_grid
            self.available_signals.add(signal_name)
        
        return self._cpp_grids[signal_name]

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
        Add point to grid with adaptive resolution using OpenVDB C++.

        Args:
            x, y, z: Point coordinates
            signals: Signal values dictionary
            timestamp: Optional timestamp (seconds)
            layer_index: Optional layer index
        """
        if not CPP_AVAILABLE:
            raise ImportError("C++ OpenVDB bindings are required")
        
        if self._finalized:
            raise ValueError("Cannot add points after finalization. Create a new grid.")
        
        # Add to C++ OpenVDB grid (one grid per signal)
        for signal_name, value in signals.items():
            adaptive_grid = self._get_or_create_grid(signal_name)
            adaptive_grid.add_point(
                x, y, z, float(value),
                timestamp if timestamp is not None else 0.0,
                layer_index if layer_index is not None else 0
            )
            self.available_signals.add(signal_name)

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
        
        Uses C++ OpenVDB implementation - no Python fallback.

        Args:
            x, y, z: Point coordinates
            timestamp: Optional timestamp
            layer_index: Optional layer index

        Returns:
            Resolution in mm
        """
        if not CPP_AVAILABLE:
            raise ImportError("C++ OpenVDB bindings are required")
        
        # Use first grid to get resolution (all grids have same resolution maps)
        if len(self._cpp_grids) > 0:
            first_grid = next(iter(self._cpp_grids.values()))
            return first_grid.get_resolution_for_point(
                x, y, z,
                timestamp if timestamp is not None else 0.0,
                layer_index if layer_index is not None else 0
            )
        
        # If no grids created yet, use base resolution
        return self.base_resolution

    def add_spatial_region(self, bbox_min: Tuple[float, float, float], bbox_max: Tuple[float, float, float], resolution: float):
        """Add a spatial region with specific resolution using OpenVDB C++."""
        if not CPP_AVAILABLE:
            raise ImportError("C++ OpenVDB bindings are required")
        
        self.spatial_map.regions.append((bbox_min, bbox_max, resolution))
        self._spatial_regions.append((bbox_min, bbox_max, resolution))
        
        # Add to all existing grids (C++ expects 7 floats)
        bbox_min_arr = np.array(bbox_min, dtype=np.float32)
        bbox_max_arr = np.array(bbox_max, dtype=np.float32)
        for grid in self._cpp_grids.values():
            grid.add_spatial_region(
                float(bbox_min_arr[0]), float(bbox_min_arr[1]), float(bbox_min_arr[2]),
                float(bbox_max_arr[0]), float(bbox_max_arr[1]), float(bbox_max_arr[2]),
                float(resolution),
            )

    def add_temporal_range(self, time_start: float, time_end: float, resolution: float):
        """Add a temporal range with specific resolution using OpenVDB C++."""
        if not CPP_AVAILABLE:
            raise ImportError("C++ OpenVDB bindings are required")
        
        self.temporal_map.time_ranges.append((time_start, time_end, resolution))
        self._temporal_ranges.append((time_start, time_end, resolution))
        
        # Add to all existing grids
        for grid in self._cpp_grids.values():
            grid.add_temporal_range(time_start, time_end, 0, 0, resolution)

    def add_layer_range(self, layer_start: int, layer_end: int, resolution: float):
        """Add a layer range with specific resolution using OpenVDB C++."""
        if not CPP_AVAILABLE:
            raise ImportError("C++ OpenVDB bindings are required")
        
        self.temporal_map.layer_ranges.append((layer_start, layer_end, resolution))
        self._layer_ranges.append((layer_start, layer_end, resolution))
        
        # Add to all existing grids
        for grid in self._cpp_grids.values():
            grid.add_temporal_range(0.0, 0.0, layer_start, layer_end, resolution)

    def finalize(self, adaptive_density: bool = True):
        """
        Finalize grid by creating region-specific grids.
        
        NOTE: C++ grid is already finalized as points are added.
        This method is kept for API compatibility.
        """
        if self._finalized:
            return
        self._finalized = True
        # C++ grid is already finalized, no additional work needed

    def get_grid(self, signal_name: str):
        """
        Get OpenVDB grid directly for a signal - NO conversion!
        
        This is the preferred method - works directly with OpenVDB sparse storage.
        Use this for all operations except Python visualization.
        
        For adaptive resolution, returns the first (finest) grid.
        Use get_all_grids() to get all resolution levels.
        
        Args:
            signal_name: Name of the signal
            
        Returns:
            OpenVDB FloatGrid (or None if signal doesn't exist)
        """
        if not CPP_AVAILABLE:
            raise ImportError("C++ OpenVDB bindings are required")
        
        if signal_name not in self._cpp_grids:
            return None
        
        adaptive_grid = self._cpp_grids[signal_name]
        all_grids = adaptive_grid.get_all_grids()
        
        if len(all_grids) == 0:
            return None
        
        # Return first (finest) grid
        return all_grids[0]
    
    def get_all_grids(self, signal_name: str):
        """
        Get all OpenVDB grids for a signal (all resolution levels) - NO conversion!
        
        Args:
            signal_name: Name of the signal
            
        Returns:
            List of OpenVDB FloatGrids (or empty list if signal doesn't exist)
        """
        if not CPP_AVAILABLE:
            raise ImportError("C++ OpenVDB bindings are required")
        
        if signal_name not in self._cpp_grids:
            return []
        
        adaptive_grid = self._cpp_grids[signal_name]
        return adaptive_grid.get_all_grids()
    
    def get_signal_array(self, signal_name: str, default: float = 0.0) -> np.ndarray:
        """
        Get signal array for entire grid - ONLY for Python visualization!
        
        ⚠️ WARNING: This converts sparse OpenVDB grid to dense NumPy array.
        This is SLOW for large grids and defeats OpenVDB's efficiency.
        
        Use get_grid() or get_all_grids() instead for all operations except visualization.
        For ParaView, export .vdb files directly - no conversion needed!
        
        Combines all region grids into a single array.
        Note: This creates a unified array from multiple resolution grids.
        """
        if not CPP_AVAILABLE:
            raise ImportError("C++ OpenVDB bindings are required")
        
        if signal_name not in self._cpp_grids:
            # Return empty array with base resolution dimensions
            size = self.bbox_max - self.bbox_min
            dims = np.ceil(size / self.base_resolution).astype(int)
            dims = np.maximum(dims, [1, 1, 1])
            return np.full(dims, default, dtype=np.float32)
        
        # Get all grids from C++ adaptive grid for this signal
        adaptive_grid = self._cpp_grids[signal_name]
        all_grids = adaptive_grid.get_all_grids()
        
        if len(all_grids) == 0:
            # Return empty array with base resolution dimensions
            size = self.bbox_max - self.bbox_min
            dims = np.ceil(size / self.base_resolution).astype(int)
            dims = np.maximum(dims, [1, 1, 1])
            return np.full(dims, default, dtype=np.float32)
        
        # ⚠️ ONLY for visualization - converts sparse to dense (SLOW!)
        # Use get_grid() or get_all_grids() for all other operations
        # For adaptive resolution, use the first grid (can be improved to merge all grids)
        # TODO: Implement proper merging of multiple resolution grids
        first_grid = all_grids[0]
        signal_array = openvdb_to_numpy(first_grid)
        return signal_array

    def _world_to_voxel_batch(self, points: np.ndarray) -> np.ndarray:
        """Vectorized voxel index calculation (for compatibility)."""
        if points.shape[1] != 3:
            raise ValueError(f"Points must be shape (N, 3), got {points.shape}")
        normalized = (points - self.bbox_min) / self.base_resolution
        indices = np.floor(normalized).astype(int)
        size = self.bbox_max - self.bbox_min
        dims = np.ceil(size / self.base_resolution).astype(int)
        dims = np.maximum(dims, [1, 1, 1])
        indices = np.clip(indices, [0, 0, 0], dims - 1)
        return indices
