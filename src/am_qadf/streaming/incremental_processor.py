"""
Incremental Processor

Incremental processing of streaming data (voxel grid updates, signal mapping).
Processes streaming data incrementally to minimize memory usage and latency.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
from collections import deque

logger = logging.getLogger(__name__)


class IncrementalProcessor:
    """
    Process streaming data incrementally.

    Provides:
    - Incremental voxel grid updates
    - Signal batch processing
    - Tracking of updated regions
    - State management for incremental processing
    """

    def __init__(self, voxel_grid: Optional[Any] = None):
        """
        Initialize incremental processor.

        Args:
            voxel_grid: Optional VoxelGrid instance for incremental updates
        """
        self.voxel_grid = voxel_grid

        # Track updated regions (bounding boxes of updated areas)
        self._updated_regions: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = []

        # Track last update timestamp
        self._last_update_time: Optional[datetime] = None

        # Statistics
        self._stats = {
            "total_points_processed": 0,
            "total_updates": 0,
            "regions_updated": 0,
        }

        logger.info("IncrementalProcessor initialized")

    def update_voxel_grid(self, new_data: np.ndarray, coordinates: np.ndarray) -> Any:  # Returns VoxelGrid
        """
        Update voxel grid with new data incrementally.

        Args:
            new_data: New data values (N points) or signal data (N x M signals)
            coordinates: Coordinates array (N x 3) of (x, y, z) positions

        Returns:
            Updated VoxelGrid
        """
        if self.voxel_grid is None:
            raise ValueError("VoxelGrid not initialized. Provide voxel_grid in __init__ or set it manually.")

        if coordinates is None or len(coordinates) == 0:
            logger.warning("No coordinates provided for voxel grid update")
            return self.voxel_grid

        # Ensure coordinates is 2D (N x 3)
        coordinates = np.asarray(coordinates)
        if coordinates.ndim == 1:
            coordinates = coordinates.reshape(1, -1)

        if coordinates.shape[1] != 3:
            raise ValueError(f"Coordinates must be N x 3, got shape {coordinates.shape}")

        # Prepare new_data
        if new_data is None or len(new_data) == 0:
            logger.warning("No data provided for voxel grid update")
            return self.voxel_grid

        new_data = np.asarray(new_data)
        if new_data.ndim == 1:
            # Single signal, convert to dict format
            signal_dict = {"value": new_data}
        elif new_data.ndim == 2:
            # Multiple signals (N x M), convert to dict
            signal_dict = {f"signal_{i}": new_data[:, i] for i in range(new_data.shape[1])}
        else:
            raise ValueError(f"Invalid data shape: {new_data.shape}, expected 1D or 2D array")

        # Track bounding box of updated region
        min_coords = np.min(coordinates, axis=0)
        max_coords = np.max(coordinates, axis=0)
        updated_region = (tuple(min_coords), tuple(max_coords))

        # Update voxel grid incrementally
        try:
            num_points = len(coordinates)

            # Add points to voxel grid
            for i in range(num_points):
                x, y, z = coordinates[i]

                # Extract signals for this point
                signals = {name: values[i] for name, values in signal_dict.items()}

                # Add point to voxel grid
                self.voxel_grid.add_point(x, y, z, signals)

            # Finalize voxel grid after batch (may want to defer finalize for performance)
            # For incremental processing, we might finalize less frequently
            # For now, finalize after each batch to ensure consistency
            # self.voxel_grid.finalize()

            # Track update
            self._updated_regions.append(updated_region)
            self._last_update_time = datetime.now()

            # Update statistics
            self._stats["total_points_processed"] += num_points
            self._stats["total_updates"] += 1
            self._stats["regions_updated"] = len(self._updated_regions)

            logger.debug(f"Updated voxel grid with {num_points} points, region: {updated_region}")

        except Exception as e:
            logger.error(f"Error updating voxel grid: {e}")
            raise

        return self.voxel_grid

    def process_signal_batch(self, signal_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Process batch of signal data.

        Args:
            signal_data: Dictionary mapping signal names to data arrays.
                        Expected format: {
                            'coordinates': np.ndarray (N x 3),
                            'signal1': np.ndarray (N,),
                            'signal2': np.ndarray (N,),
                            ...
                        }

        Returns:
            Dictionary with processing results
        """
        if "coordinates" not in signal_data:
            raise ValueError("signal_data must contain 'coordinates' key")

        coordinates = signal_data["coordinates"]
        signal_names = [k for k in signal_data.keys() if k != "coordinates"]

        if len(signal_names) == 0:
            logger.warning("No signals found in signal_data")
            return {"processed": False, "reason": "no_signals"}

        try:
            # Prepare data array (stack signals)
            signal_arrays = [signal_data[name] for name in signal_names]
            data_array = np.column_stack(signal_arrays) if len(signal_arrays) > 1 else signal_arrays[0]

            # Update voxel grid if available
            if self.voxel_grid:
                self.update_voxel_grid(data_array, coordinates)

            # Prepare result
            result = {
                "processed": True,
                "points_processed": len(coordinates),
                "signals_processed": signal_names,
                "timestamp": datetime.now().isoformat(),
                "voxel_grid_updated": self.voxel_grid is not None,
            }

            if self._updated_regions:
                result["last_updated_region"] = self._updated_regions[-1]

            return result

        except Exception as e:
            logger.error(f"Error processing signal batch: {e}")
            return {"processed": False, "error": str(e), "timestamp": datetime.now().isoformat()}

    def get_updated_regions(self, clear: bool = False) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        """
        Get regions that were updated in last processing cycle.

        Args:
            clear: If True, clear the updated regions list after returning

        Returns:
            List of (bbox_min, bbox_max) tuples for updated regions
        """
        regions = self._updated_regions.copy()

        if clear:
            self._updated_regions.clear()
            self._stats["regions_updated"] = 0

        return regions

    def get_combined_updated_region(self) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        """
        Get combined bounding box of all updated regions.

        Returns:
            (bbox_min, bbox_max) tuple for combined region, or None if no updates
        """
        if not self._updated_regions:
            return None

        all_min_coords = np.array([region[0] for region in self._updated_regions])
        all_max_coords = np.array([region[1] for region in self._updated_regions])

        combined_min = tuple(np.min(all_min_coords, axis=0))
        combined_max = tuple(np.max(all_max_coords, axis=0))

        return (combined_min, combined_max)

    def reset(self) -> None:
        """Reset processor state."""
        self._updated_regions.clear()
        self._last_update_time = None
        self._stats = {
            "total_points_processed": 0,
            "total_updates": 0,
            "regions_updated": 0,
        }
        logger.info("IncrementalProcessor state reset")

    def set_voxel_grid(self, voxel_grid: Any) -> None:
        """
        Set or replace voxel grid.

        Args:
            voxel_grid: VoxelGrid instance
        """
        self.voxel_grid = voxel_grid
        logger.info("VoxelGrid set for IncrementalProcessor")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get processing statistics.

        Returns:
            Dictionary with statistics
        """
        stats = self._stats.copy()
        stats["last_update_time"] = self._last_update_time.isoformat() if self._last_update_time else None
        stats["num_regions_tracked"] = len(self._updated_regions)
        stats["has_voxel_grid"] = self.voxel_grid is not None
        return stats
