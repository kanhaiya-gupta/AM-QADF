"""
Voxel Grid Data Structure

Core voxel grid implementation for storing and managing 3D voxel data.
Supports multiple signal types per voxel and configurable spatial resolution.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any

# Import VoxelData from core entities
from ..core.entities import VoxelData


class VoxelGrid:
    """
    Voxel grid for storing 3D spatial data.

    The grid is defined by a bounding box and resolution, creating a regular
    3D grid of voxels. Each voxel can store multiple signal types.
    """

    def __init__(
        self,
        bbox_min: Tuple[float, float, float],
        bbox_max: Tuple[float, float, float],
        resolution: float,
        aggregation: str = "mean",
    ):
        """
        Initialize voxel grid.

        Args:
            bbox_min: Minimum bounding box coordinates (x_min, y_min, z_min) in mm
            bbox_max: Maximum bounding box coordinates (x_max, y_max, z_max) in mm
            resolution: Voxel size in mm (cubic voxels)
            aggregation: How to aggregate multiple values per voxel ('mean', 'max', 'min', 'sum')
        """
        self.bbox_min = np.array(bbox_min, dtype=np.float64)
        self.bbox_max = np.array(bbox_max, dtype=np.float64)
        self.resolution = float(resolution)
        self.aggregation = aggregation

        # Calculate grid dimensions
        self.size = self.bbox_max - self.bbox_min
        self.dims = np.ceil(self.size / self.resolution).astype(int)

        # Ensure at least 1 voxel in each dimension
        self.dims = np.maximum(self.dims, [1, 1, 1])

        # Actual grid size (may be slightly larger than bbox due to rounding)
        self.actual_size = self.dims * self.resolution

        # Initialize voxel data structure
        # Using dictionary for sparse storage (only store non-empty voxels)
        self.voxels: Dict[Tuple[int, int, int], VoxelData] = {}

        # Track which signals are present in the grid
        self.available_signals: set = set()

    def _world_to_voxel(self, x: float, y: float, z: float) -> Tuple[int, int, int]:
        """
        Convert world coordinates to voxel indices.

        Args:
            x, y, z: World coordinates in mm

        Returns:
            Voxel indices (i, j, k)
        """
        # Normalize to grid space
        normalized = (np.array([x, y, z]) - self.bbox_min) / self.resolution

        # Convert to integer indices
        indices = np.floor(normalized).astype(int)

        # Clamp to valid range
        indices = np.clip(indices, [0, 0, 0], self.dims - 1)

        return tuple(indices)

    def _world_to_voxel_batch(self, points: np.ndarray) -> np.ndarray:
        """
        Vectorized voxel index calculation for all points.

        This method processes multiple points at once using NumPy vectorization,
        providing significant performance improvement over single-point processing.

        Args:
            points: Array of points (N, 3) with (x, y, z) coordinates in mm

        Returns:
            Array of voxel indices (N, 3) as integers
        """
        if points.shape[1] != 3:
            raise ValueError(f"Points must be shape (N, 3), got {points.shape}")

        # Vectorized normalization to grid space
        normalized = (points - self.bbox_min) / self.resolution

        # Convert to integer indices
        indices = np.floor(normalized).astype(int)

        # Clamp to valid range (vectorized)
        indices = np.clip(indices, [0, 0, 0], self.dims - 1)

        return indices

    def _build_voxel_grid_batch(self, voxel_data: Dict[Tuple[int, int, int], Dict[str, Any]]):
        """
        Build voxel grid structure from pre-aggregated data.

        This method efficiently builds the voxel grid from pre-computed
        aggregated data, avoiding per-point dictionary operations.

        Args:
            voxel_data: Dictionary mapping voxel keys to dictionaries containing:
                - 'signals': Dict[str, float] - aggregated signal values
                - 'count': int - number of points contributing to this voxel
        """
        for voxel_key, data in voxel_data.items():
            if voxel_key not in self.voxels:
                self.voxels[voxel_key] = VoxelData()

            voxel = self.voxels[voxel_key]

            # Directly assign aggregated signals (already computed)
            for signal_name, value in data["signals"].items():
                voxel.signals[signal_name] = value
                self.available_signals.add(signal_name)

            # Set count
            voxel.count = data.get("count", 1)

    def _voxel_to_world(self, i: int, j: int, k: int) -> Tuple[float, float, float]:
        """
        Convert voxel indices to world coordinates (center of voxel).

        Args:
            i, j, k: Voxel indices

        Returns:
            World coordinates (x, y, z) in mm
        """
        indices = np.array([i, j, k], dtype=float)
        world = self.bbox_min + (indices + 0.5) * self.resolution
        return tuple(world)

    def add_point(self, x: float, y: float, z: float, signals: Dict[str, float]):
        """
        Add a data point to the voxel grid.

        Args:
            x, y, z: World coordinates in mm
            signals: Dictionary of signal names to values
        """
        voxel_idx = self._world_to_voxel(x, y, z)

        if voxel_idx not in self.voxels:
            self.voxels[voxel_idx] = VoxelData()

        voxel = self.voxels[voxel_idx]

        for signal_name, value in signals.items():
            voxel.add_signal(signal_name, value, self.aggregation)
            self.available_signals.add(signal_name)

    def finalize(self):
        """
        Finalize all voxels by aggregating multiple values.
        Should be called after all points have been added.
        """
        for voxel in self.voxels.values():
            voxel.finalize(self.aggregation)

    def get_voxel(self, i: int, j: int, k: int) -> Optional[VoxelData]:
        """
        Get voxel data at given indices.

        Args:
            i, j, k: Voxel indices

        Returns:
            VoxelData if voxel exists, None otherwise
        """
        if (i, j, k) in self.voxels:
            return self.voxels[(i, j, k)]
        return None

    def get_signal_array(self, signal_name: str, default: float = 0.0) -> np.ndarray:
        """
        Get a 3D numpy array for a specific signal.

        OPTIMIZED: Uses vectorized operations for better performance.

        Args:
            signal_name: Name of the signal
            default: Default value for empty voxels

        Returns:
            3D numpy array with signal values
        """
        array = np.full(self.dims, default, dtype=np.float32)

        if len(self.voxels) == 0:
            return array

        # OPTIMIZED: Extract indices and values in batch, then assign vectorized
        indices_list = []
        values_list = []

        for (i, j, k), voxel in self.voxels.items():
            if signal_name in voxel.signals:
                indices_list.append([i, j, k])
                signal_value = voxel.signals[signal_name]
                # Handle both float and list types
                if isinstance(signal_value, list):
                    # If it's a list, take the mean (voxel should be finalized first)
                    values_list.append(float(np.mean(signal_value)))
                else:
                    values_list.append(float(signal_value))

        if len(indices_list) > 0:
            # Vectorized assignment (much faster than per-voxel assignment)
            indices_array = np.array(indices_list, dtype=np.int32)
            values_array = np.array(values_list, dtype=np.float32)
            array[indices_array[:, 0], indices_array[:, 1], indices_array[:, 2]] = values_array

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
        total_voxels = np.prod(self.dims)
        filled_voxels = len(self.voxels)
        fill_ratio = filled_voxels / total_voxels if total_voxels > 0 else 0.0

        stats = {
            "dimensions": tuple(self.dims),
            "resolution_mm": self.resolution,
            "bounding_box_min": tuple(self.bbox_min),
            "bounding_box_max": tuple(self.bbox_max),
            "total_voxels": int(total_voxels),
            "filled_voxels": filled_voxels,
            "fill_ratio": fill_ratio,
            "available_signals": sorted(list(self.available_signals)),
        }

        # Add signal statistics
        for signal_name in self.available_signals:
            signal_array = self.get_signal_array(signal_name)
            non_zero = signal_array[signal_array != 0.0]
            if len(non_zero) > 0:
                stats[f"{signal_name}_mean"] = float(np.mean(non_zero))
                stats[f"{signal_name}_min"] = float(np.min(non_zero))
                stats[f"{signal_name}_max"] = float(np.max(non_zero))
                stats[f"{signal_name}_std"] = float(np.std(non_zero))

        return stats
