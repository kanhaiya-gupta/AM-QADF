"""
Grid synchronizer - thin C++ wrapper.

Thin Python wrapper for C++ GridSynchronizer.
All core computation is done in C++; this module only converts VoxelGrid <-> FloatGrid and calls C++.
"""

from typing import List, Dict, Optional, Tuple, Any
import numpy as np

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


def _first_common_signal(grids: List[VoxelGrid], required: Optional[VoxelGrid] = None) -> Optional[str]:
    """Return first signal name common to all grids (and required if given)."""
    if not grids:
        return None
    common = set(grids[0].available_signals)
    for g in grids[1:]:
        common &= g.available_signals
    if required is not None:
        common &= required.available_signals
    return next(iter(common), None)


def _voxel_grids_to_float_grids(
    grids: List[VoxelGrid],
    signal_name: str,
) -> List[Any]:
    """Extract FloatGrid for signal_name from each VoxelGrid; skip grids missing the signal."""
    out: List[Any] = []
    for g in grids:
        fg = g.get_grid(signal_name)
        if fg is not None:
            out.append(fg)
    return out


def _float_grids_to_voxel_grids(
    float_grids: List[Any],
    template: VoxelGrid,
    signal_name: str,
) -> List[VoxelGrid]:
    """Wrap each C++ FloatGrid into a VoxelGrid with same bbox/resolution as template (thin wrapper)."""
    result: List[VoxelGrid] = []
    for fg in float_grids:
        vg = VoxelGrid(
            bbox_min=tuple(template.bbox_min),
            bbox_max=tuple(template.bbox_max),
            resolution=template.resolution,
            aggregation=template.aggregation,
        )
        vg._get_or_create_grid(signal_name).copy_from_grid(fg)
        vg.available_signals.add(signal_name)
        result.append(vg)
    return result


class SynchronizationClient:
    """
    Synchronization client - thin C++ wrapper around GridSynchronizer.

    All alignment work is done in C++ (SpatialAlignment, TemporalAlignment, GridSynchronizer).
    This client only converts VoxelGrid <-> OpenVDB FloatGrid and calls C++.
    """

    def __init__(self):
        """Initialize synchronization client (requires am_qadf_native)."""
        if not CPP_AVAILABLE:
            raise ImportError(
                "C++ bindings not available. "
                "Please build am_qadf_native with pybind11 bindings."
            )
        self._synchronizer = GridSynchronizer()

    def align_temporal(
        self,
        grids: List[VoxelGrid],
        timestamps: List[List[float]],
        temporal_window: float = 0.1,
        layer_tolerance: int = 1,
        signal_name: Optional[str] = None,
        layer_indices: Optional[List[List[int]]] = None,
    ) -> List[VoxelGrid]:
        """
        Align grids temporally (C++ TemporalAlignment via GridSynchronizer).

        Args:
            grids: List of VoxelGrid objects (must share at least one signal).
            timestamps: Per-grid list of timestamps (one per layer/voxel as expected by C++).
            temporal_window: Time window for alignment (seconds).
            layer_tolerance: Layer tolerance for alignment.
            signal_name: Signal to align; if None, first common signal is used.
            layer_indices: Per-grid layer indices; if None, empty lists are used.

        Returns:
            List of temporally aligned VoxelGrids (same order as input where possible).
        """
        sig = signal_name or _first_common_signal(grids)
        if not sig:
            raise ValueError("No common signal across grids; specify signal_name or add a shared signal.")
        float_grids = _voxel_grids_to_float_grids(grids, sig)
        if len(float_grids) != len(grids):
            raise ValueError("Some grids do not have the chosen signal.")
        ts = timestamps if timestamps else [[] for _ in grids]
        li = layer_indices if layer_indices is not None else [[] for _ in grids]
        # Pad to match grid count
        while len(ts) < len(grids):
            ts.append([])
        while len(li) < len(grids):
            li.append([])
        aligned = self._synchronizer.synchronize_temporal(
            float_grids, ts, li, float(temporal_window), int(layer_tolerance)
        )
        template = grids[0]
        return _float_grids_to_voxel_grids(aligned, template, sig)

    def align_spatial(
        self,
        source_grids: List[VoxelGrid],
        target_grid: VoxelGrid,
        method: str = "trilinear",
        signal_name: Optional[str] = None,
    ) -> List[VoxelGrid]:
        """
        Align grids spatially to target grid's coordinate system (C++ SpatialAlignment).

        Args:
            source_grids: List of source VoxelGrid objects.
            target_grid: Target VoxelGrid (reference coordinate system).
            method: Interpolation method ('nearest', 'trilinear', 'triquadratic').
            signal_name: Signal to align; if None, first common signal across sources and target.

        Returns:
            List of spatially aligned VoxelGrids.
        """
        sig = signal_name or _first_common_signal(source_grids, required=target_grid)
        if not sig:
            raise ValueError("No common signal between source grids and target; specify signal_name.")
        target_float = target_grid.get_grid(sig)
        if target_float is None:
            raise ValueError("Target grid does not have the chosen signal.")
        source_floats = _voxel_grids_to_float_grids(source_grids, sig)
        if not source_floats:
            raise ValueError("No source grid has the chosen signal.")
        aligned = self._synchronizer.synchronize_spatial(source_floats, target_float, method)
        return _float_grids_to_voxel_grids(aligned, target_grid, sig)

    def synchronize(
        self,
        grids: List[VoxelGrid],
        timestamps: List[List[float]],
        reference_grid: VoxelGrid,
        temporal_window: float = 0.1,
        layer_tolerance: int = 1,
        signal_name: Optional[str] = None,
        layer_indices: Optional[List[List[int]]] = None,
    ) -> List[VoxelGrid]:
        """
        Synchronize grids both spatially and temporally (C++ GridSynchronizer.synchronize).

        Args:
            grids: List of VoxelGrid objects to synchronize.
            timestamps: Per-grid timestamp arrays for temporal alignment.
            reference_grid: Reference grid for spatial alignment.
            temporal_window: Time window for temporal alignment.
            layer_tolerance: Layer tolerance.
            signal_name: Signal to synchronize; if None, first common signal is used.
            layer_indices: Per-grid layer indices; if None, empty lists are used.

        Returns:
            List of synchronized VoxelGrids.
        """
        sig = signal_name or _first_common_signal(grids, required=reference_grid)
        if not sig:
            raise ValueError("No common signal across grids and reference; specify signal_name.")
        ref_float = reference_grid.get_grid(sig)
        if ref_float is None:
            raise ValueError("Reference grid does not have the chosen signal.")
        source_floats = _voxel_grids_to_float_grids(grids, sig)
        if len(source_floats) != len(grids):
            raise ValueError("Some grids do not have the chosen signal.")
        ts = timestamps if timestamps else [[] for _ in grids]
        li = layer_indices if layer_indices is not None else [[] for _ in grids]
        while len(ts) < len(grids):
            ts.append([])
        while len(li) < len(grids):
            li.append([])
        synchronized = self._synchronizer.synchronize(
            source_floats, ref_float, ts, li, float(temporal_window), int(layer_tolerance)
        )
        return _float_grids_to_voxel_grids(synchronized, reference_grid, sig)
