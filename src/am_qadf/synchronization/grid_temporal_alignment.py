"""
Grid temporal alignment - thin C++ wrapper.

Thin Python wrapper for temporal alignment of grids.
Uses C++ GridSynchronizer.synchronize_temporal (FloatGrid + timestamps + layer_indices).
Converts VoxelGrid <-> FloatGrid and calls C++.
"""

from typing import List, Optional

try:
    try:
        from am_qadf_native.synchronization import GridSynchronizer
    except ImportError:
        from am_qadf_native import GridSynchronizer
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
    GridSynchronizer = None

from ..voxelization.uniform_resolution import VoxelGrid

from .grid_synchronizer import _voxel_grids_to_float_grids, _float_grids_to_voxel_grids


class GridTemporalAlignment:
    """
    Thin wrapper for temporal alignment of multiple grids.

    Uses C++ GridSynchronizer.synchronize_temporal (expects FloatGrid + timestamps + layer_indices).
    """

    def __init__(self):
        """Initialize (requires am_qadf_native)."""
        if not CPP_AVAILABLE:
            raise ImportError(
                "C++ bindings not available. "
                "Please build am_qadf_native with pybind11 bindings."
            )
        self._synchronizer = GridSynchronizer()

    def synchronize_temporal(
        self,
        grids: List[VoxelGrid],
        signal_name: str,
        temporal_window: float = 0.1,
        layer_tolerance: int = 1,
        timestamps: Optional[List[List[float]]] = None,
        layer_indices: Optional[List[List[int]]] = None,
    ) -> List[VoxelGrid]:
        """
        Align grids temporally (C++ GridSynchronizer.synchronize_temporal).

        Args:
            grids: List of VoxelGrid objects (must share signal_name).
            signal_name: Signal to align.
            temporal_window: Time window for alignment.
            layer_tolerance: Layer tolerance.
            timestamps: Per-grid timestamps; if None, empty lists are used.
            layer_indices: Per-grid layer indices; if None, empty lists are used.

        Returns:
            List of temporally aligned VoxelGrids.
        """
        float_grids = _voxel_grids_to_float_grids(grids, signal_name)
        if len(float_grids) != len(grids):
            raise ValueError("Some grids do not have the chosen signal.")
        n = len(grids)
        # Per-grid metadata only (length n); point data stays in float_grids — no per-point copy.
        ts = [[] for _ in range(n)] if timestamps is None else list(timestamps)[:n]
        if len(ts) < n:
            ts = ts + [[] for _ in range(n - len(ts))]
        li = [[] for _ in range(n)] if layer_indices is None else list(layer_indices)[:n]
        if len(li) < n:
            li = li + [[] for _ in range(n - len(li))]
        # C++ creates no bins when all timestamps are empty; pass one distinct timestamp per grid
        # so each grid gets its own bin and C++ returns one grid per input (no Python bypass).
        if all(not t for t in ts) and all(not elem for elem in li):
            ts = [[float(i)] for i in range(n)]
        aligned = self._synchronizer.synchronize_temporal(
            float_grids, ts, li, float(temporal_window), int(layer_tolerance)
        )
        return _float_grids_to_voxel_grids(aligned, grids[0], signal_name)
