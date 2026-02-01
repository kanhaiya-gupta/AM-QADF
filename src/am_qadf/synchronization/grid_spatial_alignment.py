"""
Grid spatial alignment - thin C++ wrapper.

Thin Python wrapper for C++ SpatialAlignment.
Converts VoxelGrid <-> FloatGrid and calls C++.
"""

from typing import List, Optional, Any

try:
    try:
        from am_qadf_native.synchronization import SpatialAlignment
    except ImportError:
        from am_qadf_native import SpatialAlignment
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
    SpatialAlignment = None

from ..voxelization.uniform_resolution import VoxelGrid

from .grid_synchronizer import _float_grids_to_voxel_grids


class GridSpatialAlignment:
    """
    Thin wrapper around C++ SpatialAlignment.

    Aligns a source grid to a target grid's coordinate system (resample).
    """

    def __init__(self):
        """Initialize (requires am_qadf_native)."""
        if not CPP_AVAILABLE:
            raise ImportError(
                "C++ bindings not available. "
                "Please build am_qadf_native with pybind11 bindings."
            )
        self._aligner = SpatialAlignment()

    def align(
        self,
        source_grid: VoxelGrid,
        target_grid: VoxelGrid,
        signal_name: str,
        method: str = "trilinear",
    ) -> VoxelGrid:
        """
        Align source grid to target grid's coordinate system (C++ SpatialAlignment.align).

        Args:
            source_grid: Source VoxelGrid.
            target_grid: Target VoxelGrid (reference transform).
            signal_name: Signal to align (must exist in both grids).
            method: Interpolation method ('nearest', 'trilinear', 'triquadratic').

        Returns:
            New VoxelGrid with aligned data (same bbox/resolution as target_grid).
        """
        source_float = source_grid.get_grid(signal_name)
        target_float = target_grid.get_grid(signal_name)
        if source_float is None:
            raise ValueError(f"Source grid does not have signal '{signal_name}'.")
        if target_float is None:
            raise ValueError(f"Target grid does not have signal '{signal_name}'.")
        aligned_float = self._aligner.align(source_float, target_float, method)
        result_list = _float_grids_to_voxel_grids([aligned_float], target_grid, signal_name)
        return result_list[0]
