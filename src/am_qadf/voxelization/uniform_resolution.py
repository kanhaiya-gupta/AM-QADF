"""
Uniform Resolution Voxel Grid Data Structure - OpenVDB C++ Implementation

Core uniform resolution voxel grid implementation using OpenVDB C++ for storage.
All core computation is done in C++ OpenVDB. This is a thin Python wrapper.

This is the uniform resolution variant of voxel grids. For multi-resolution or adaptive
resolution grids, see multi_resolution.py and adaptive_resolution.py.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any

# C++ OpenVDB bindings - REQUIRED (no Python fallback)
try:
    from am_qadf_native import (
        numpy_to_openvdb,
        openvdb_to_numpy,
        UniformVoxelGrid,
    )
    CPP_AVAILABLE = True
except ImportError as e:
    CPP_AVAILABLE = False
    raise ImportError(
        "C++ OpenVDB bindings are required. "
        "Please build am_qadf_native with pybind11 bindings. "
        f"Original error: {e}"
    )

# Import VoxelData from core entities (for API compatibility)
from ..core.entities import VoxelData


class VoxelGrid:
    """
    Uniform resolution voxel grid for storing 3D spatial data.
    
    Uses C++ OpenVDB FloatGrid internally for all storage and operations.
    One FloatGrid per signal (multi-signal support).
    
    All coordinate transformations and sparse storage are handled by OpenVDB.
    """

    def __init__(
        self,
        bbox_min: Tuple[float, float, float],
        bbox_max: Tuple[float, float, float],
        resolution: float,
        aggregation: str = "mean",
    ):
        """
        Initialize voxel grid using OpenVDB C++.

        Args:
            bbox_min: Minimum bounding box coordinates (x_min, y_min, z_min) in mm
            bbox_max: Maximum bounding box coordinates (x_max, y_max, z_max) in mm
            resolution: Voxel size in mm (cubic voxels)
            aggregation: How to aggregate multiple values per voxel ('mean', 'max', 'min', 'sum')
        """
        if not CPP_AVAILABLE:
            raise ImportError("C++ OpenVDB bindings are required")
        
        # Validate inputs
        if resolution <= 0:
            raise ValueError("Resolution must be greater than 0")
        
        self.bbox_min = np.array(bbox_min, dtype=np.float64)
        self.bbox_max = np.array(bbox_max, dtype=np.float64)
        
        if np.any(self.bbox_max <= self.bbox_min):
            raise ValueError("bbox_max must be greater than bbox_min in all dimensions")
        
        self.resolution = float(resolution)
        self.aggregation = aggregation

        # Calculate grid dimensions (for API compatibility)
        self.size = self.bbox_max - self.bbox_min
        self.dims = np.ceil(self.size / self.resolution).astype(int)
        self.dims = np.maximum(self.dims, [1, 1, 1])
        self.actual_size = self.dims * self.resolution

        # C++ OpenVDB storage: one UniformVoxelGrid per signal
        # Using UniformVoxelGrid wrapper (which manages FloatGrid internally)
        self._cpp_grids: Dict[str, UniformVoxelGrid] = {}
        
        # Track which signals are present
        self.available_signals: set = set()
        
        # For aggregation: store multiple values per voxel temporarily
        # Key: (signal_name, voxel_idx) -> list of values
        self._pending_values: Dict[Tuple[str, Tuple[int, int, int]], List[float]] = {}

    def _get_or_create_grid(self, signal_name: str) -> UniformVoxelGrid:
        """
        Get or create UniformVoxelGrid for a signal.
        
        OpenVDB Transform handles bbox_min offset automatically.
        """
        if signal_name not in self._cpp_grids:
            # Create new UniformVoxelGrid with bbox_min offset
            # OpenVDB will handle coordinate transformation automatically
            uniform_grid = UniformVoxelGrid(
                self.resolution,
                float(self.bbox_min[0]),
                float(self.bbox_min[1]),
                float(self.bbox_min[2])
            )
            uniform_grid.set_signal_name(signal_name)
            self._cpp_grids[signal_name] = uniform_grid
            self.available_signals.add(signal_name)
        
        return self._cpp_grids[signal_name]

    def _world_to_voxel(self, x: float, y: float, z: float) -> Tuple[int, int, int]:
        """
        Convert world coordinates to voxel indices.
        
        Note: Kept for API compatibility only. OpenVDB handles this in C++.
        """
        normalized = (np.array([x, y, z]) - self.bbox_min) / self.resolution
        indices = np.floor(normalized).astype(int)
        indices = np.clip(indices, [0, 0, 0], self.dims - 1)
        return tuple(indices)

    def _world_to_voxel_batch(self, points: np.ndarray) -> np.ndarray:
        """
        Vectorized voxel index calculation for all points.
        
        Note: Kept for API compatibility only. OpenVDB handles this in C++.
        """
        if points.shape[1] != 3:
            raise ValueError(f"Points must be shape (N, 3), got {points.shape}")
        
        normalized = (points - self.bbox_min) / self.resolution
        indices = np.floor(normalized).astype(int)
        indices = np.clip(indices, [0, 0, 0], self.dims - 1)
        return indices

    def add_point(self, x: float, y: float, z: float, signals: Dict[str, float]):
        """
        Add a data point to the voxel grid using OpenVDB C++.

        Args:
            x, y, z: World coordinates in mm (OpenVDB handles transformation)
            signals: Dictionary of signal names to values
        """
        if not CPP_AVAILABLE:
            raise ImportError("C++ OpenVDB bindings are required")
        
        # Get voxel index for this point (for aggregation tracking only)
        voxel_idx = self._world_to_voxel(x, y, z)
        
        # Collect all values for aggregation (will be processed in finalize())
        # OpenVDB handles coordinate transformation automatically
        for signal_name, value in signals.items():
            self.available_signals.add(signal_name)
            key = (signal_name, voxel_idx)
            if key not in self._pending_values:
                self._pending_values[key] = []
            self._pending_values[key].append(float(value))

    def finalize(self):
        """
        Finalize all voxels by aggregating multiple values.
        Should be called after all points have been added.
        
        Aggregates all collected values based on aggregation mode and sets final values in OpenVDB grids.
        """
        if not CPP_AVAILABLE:
            raise ImportError("C++ OpenVDB bindings are required")
        
        # Aggregate all collected values and set in OpenVDB grids
        for (signal_name, voxel_idx), values in self._pending_values.items():
            if len(values) == 0:
                continue
            
            uniform_grid = self._get_or_create_grid(signal_name)
            i, j, k = voxel_idx
            
            # Aggregate in C++ (fast, no Python loops)
            uniform_grid.aggregate_at_voxel(i, j, k, values, self.aggregation)
        
        # Clear pending values
        self._pending_values.clear()

    def get_voxel(self, i: int, j: int, k: int) -> Optional[VoxelData]:
        """
        Get voxel data at given indices.
        
        Converts from OpenVDB grids to VoxelData for API compatibility.

        Args:
            i, j, k: Voxel indices

        Returns:
            VoxelData if voxel exists, None otherwise
        """
        if not CPP_AVAILABLE:
            raise ImportError("C++ OpenVDB bindings are required")
        
        # Check if any signal has a value at this voxel location
        voxel_data = VoxelData()
        has_data = False
        
        for signal_name in self.available_signals:
            if signal_name in self._cpp_grids:
                uniform_grid = self._cpp_grids[signal_name]
                # UniformVoxelGrid.get_value() takes voxel coordinates (i, j, k)
                value = uniform_grid.get_value(i, j, k)
                if value != 0.0:  # OpenVDB default value
                    voxel_data.signals[signal_name] = float(value)
                    has_data = True
        
        if has_data:
            return voxel_data
        return None

    def get_grid(self, signal_name: str):
        """
        Get OpenVDB grid directly for a signal - NO conversion!
        
        This is the preferred method - works directly with OpenVDB sparse storage.
        Use this for all operations except Python visualization.
        
        Args:
            signal_name: Name of the signal
            
        Returns:
            OpenVDB FloatGrid (or None if signal doesn't exist)
        """
        if not CPP_AVAILABLE:
            raise ImportError("C++ OpenVDB bindings are required")
        
        if signal_name not in self._cpp_grids:
            return None
        
        uniform_grid = self._cpp_grids[signal_name]
        return uniform_grid.get_grid()
    
    def get_signal_array(self, signal_name: str, default: float = 0.0) -> np.ndarray:
        """
        Get a 3D numpy array for a specific signal - ONLY for Python visualization!
        
        ⚠️ WARNING: This converts sparse OpenVDB grid to dense NumPy array.
        This is SLOW for large grids and defeats OpenVDB's efficiency.
        
        Use get_grid() instead for all operations except visualization.
        For ParaView, export .vdb files directly - no conversion needed!
        
        Args:
            signal_name: Name of the signal
            default: Default value for empty voxels

        Returns:
            3D numpy array with signal values
        """
        if not CPP_AVAILABLE:
            raise ImportError("C++ OpenVDB bindings are required")
        
        if signal_name not in self._cpp_grids:
            # Return empty array with correct dimensions
            return np.full(self.dims, default, dtype=np.float32)
        
        # Get FloatGrid from UniformVoxelGrid
        uniform_grid = self._cpp_grids[signal_name]
        grid = uniform_grid.get_grid()
        
        # ⚠️ ONLY for visualization - converts sparse to dense (SLOW!)
        # Use get_grid() for all other operations
        array = openvdb_to_numpy(grid)
        
        # Note: openvdb_to_numpy returns array with shape based on active voxel bbox
        # We may need to pad/trim to match self.dims
        # For now, return as-is (can be enhanced later to match exact dims)
        return array

    def get_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the bounding box of the grid.

        Returns:
            Tuple of (bbox_min, bbox_max) as numpy arrays
        """
        return self.bbox_min.copy(), self.bbox_max.copy()

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the voxel grid.

        Returns:
            Dictionary with grid statistics
        """
        if not CPP_AVAILABLE:
            raise ImportError("C++ OpenVDB bindings are required")
        
        total_voxels = int(np.prod(self.dims))

        stats = {
            "dimensions": tuple(self.dims),
            "resolution_mm": self.resolution,
            "bounding_box_min": tuple(self.bbox_min),
            "bounding_box_max": tuple(self.bbox_max),
            "total_voxels": total_voxels,
            "available_signals": sorted(list(self.available_signals)),
        }

        # Get signal statistics from C++ (fast, no Python loops)
        for signal_name in self.available_signals:
            if signal_name in self._cpp_grids:
                uniform_grid = self._cpp_grids[signal_name]
                cpp_stats = uniform_grid.get_statistics()
                
                stats[f"{signal_name}_filled_voxels"] = cpp_stats.filled_voxels
                stats[f"{signal_name}_fill_ratio"] = cpp_stats.fill_ratio
                stats[f"{signal_name}_mean"] = cpp_stats.mean
                stats[f"{signal_name}_min"] = cpp_stats.min
                stats[f"{signal_name}_max"] = cpp_stats.max
                stats[f"{signal_name}_std"] = cpp_stats.std
        
        # Overall statistics (use first signal as representative)
        if len(self._cpp_grids) > 0:
            first_stats = next(iter(self._cpp_grids.values())).get_statistics()
            stats["filled_voxels"] = first_stats.filled_voxels
            stats["fill_ratio"] = first_stats.fill_ratio
        else:
            stats["filled_voxels"] = 0
            stats["fill_ratio"] = 0.0

        return stats
