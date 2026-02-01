"""
Voxel-Based Data Fusion - C++ Wrapper

Thin Python wrapper for C++ grid fusion implementation.
All core computation is done in C++.
"""

from typing import List, Dict, Optional, Union, Any
import numpy as np

try:
    from am_qadf_native.fusion import GridFusion
    from am_qadf_native import numpy_to_openvdb, openvdb_to_numpy
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
    GridFusion = None

from ..voxelization.uniform_resolution import VoxelGrid
from ..core.entities import VoxelData


class VoxelFusion:
    """
    Voxel-based grid fusion - C++ wrapper.
    
    This is a thin wrapper around the C++ GridFusion implementation.
    All core computation is done in C++.
    
    NOTE: This wrapper requires C++ bindings and conversion utilities.
    """

    def __init__(self):
        """Initialize voxel fusion."""
        if not CPP_AVAILABLE:
            raise ImportError(
                "C++ bindings not available. "
                "Please build am_qadf_native with pybind11 bindings."
            )
        self._fusion = GridFusion()

    def fuse_grids(
        self,
        grids: List[VoxelGrid],
        strategy: str = "weighted_average",
        weights: Optional[List[float]] = None,
        signal_name: Optional[str] = None,
    ) -> VoxelGrid:
        """
        Fuse multiple voxel grids into a single grid.

        Args:
            grids: List of VoxelGrid objects to fuse
            strategy: Fusion strategy ('weighted_average', 'average', 'max', 'min', 'median')
            weights: Optional weights for each grid (for weighted_average)
            signal_name: Optional signal name to fuse (if None, fuses all signals)

        Returns:
            Fused VoxelGrid
        """
        if len(grids) == 0:
            raise ValueError("No grids provided for fusion")
        
        if len(grids) == 1:
            return grids[0]

        # Use first grid as template for output
        output_grid = VoxelGrid(
            bbox_min=tuple(grids[0].bbox_min),
            bbox_max=tuple(grids[0].bbox_max),
            resolution=grids[0].resolution,
            aggregation=grids[0].aggregation
        )

        # Determine which signals to fuse
        if signal_name:
            signals_to_fuse = [signal_name]
        else:
            # Fuse all common signals
            common_signals = set(grids[0].available_signals)
            for grid in grids[1:]:
                common_signals &= grid.available_signals
            signals_to_fuse = list(common_signals)

        if len(signals_to_fuse) == 0:
            return output_grid

        # Fuse each signal separately
        for signal in signals_to_fuse:
            # Convert each grid's signal to OpenVDB FloatGrid
            openvdb_grids = []
            for grid in grids:
                signal_array = grid.get_signal_array(signal, default=0.0)
                openvdb_grid = numpy_to_openvdb(signal_array, grid.resolution)
                openvdb_grids.append(openvdb_grid)

            # Fuse grids using C++
            if weights is not None and strategy == "weighted_average":
                weights_cpp = [float(w) for w in weights]
                fused_grid = self._fusion.fuse_weighted(openvdb_grids, weights_cpp)
            else:
                fused_grid = self._fusion.fuse(openvdb_grids, strategy)

            # Convert fused grid back to numpy
            fused_array = openvdb_to_numpy(fused_grid)

            # Update output_grid with fused signal using C++ (fast, no Python loops)
            # Use populateFromArray to populate grid directly from numpy array
            output_grid._get_or_create_grid(signal).populate_from_array(fused_array)
            output_grid.available_signals.add(signal)

        return output_grid
