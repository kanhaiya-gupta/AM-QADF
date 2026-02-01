"""
ParaView Exporter - C++ Wrapper

Thin Python wrapper for C++ VDBWriter to export voxel grids to .vdb files.
All core computation is done in C++.

ParaView is the PRIMARY visualization method - it can open .vdb files directly.
"""

from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging

# C++ OpenVDB bindings - REQUIRED (no fallback)
from am_qadf_native.io import VDBWriter
# FloatGridPtr is a type alias, may not be directly importable
# Use Any for type hints if not available
try:
    from am_qadf_native import FloatGridPtr
except ImportError:
    FloatGridPtr = Any  # Fallback for type hints

logger = logging.getLogger(__name__)


def export_voxel_grid_to_paraview(
    voxel_grid: Any,  # VoxelGrid or AdaptiveResolutionGrid
    output_path: str,
    signal_names: Optional[List[str]] = None,
) -> str:
    """
    Export voxel grid to ParaView .vdb file.
    
    This is the PRIMARY visualization method. ParaView can open .vdb files directly
    and provides superior 3D visualization, slice views, isosurfaces, etc.
    
    Args:
        voxel_grid: VoxelGrid or AdaptiveResolutionGrid instance
        output_path: Path to output .vdb file (will be created)
                     If no extension provided, .vdb will be added automatically
        signal_names: Optional list of signal names to export (all if None)
    
    Returns:
        Path to created .vdb file (with .vdb extension)
    
    Example:
        >>> from am_qadf.visualization import export_voxel_grid_to_paraview
        >>> export_voxel_grid_to_paraview(voxel_grid, "output.vdb")
        'output.vdb'
        >>> # Or without extension (auto-adds .vdb)
        >>> export_voxel_grid_to_paraview(voxel_grid, "output")
        'output.vdb'
    """
    output_path_obj = Path(output_path)
    
    # Ensure .vdb extension
    if output_path_obj.suffix.lower() != '.vdb':
        output_path_obj = output_path_obj.with_suffix('.vdb')
    
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    # Get all signals or specified ones
    if signal_names is None:
        signal_names = list(voxel_grid.available_signals)
    
    if len(signal_names) == 0:
        raise ValueError("No signals to export")
    
    # Convert each signal to FloatGrid
    grids: List[FloatGridPtr] = []
    grid_names: List[str] = []
    
    for signal_name in signal_names:
        if signal_name not in voxel_grid.available_signals:
            logger.warning(f"Signal {signal_name} not found in grid, skipping")
            continue

        # Get FloatGrid via wrapper API (VoxelGrid and AdaptiveResolutionGrid both have get_grid(signal_name))
        float_grid = voxel_grid.get_grid(signal_name)
        if float_grid is None:
            logger.warning(f"Signal {signal_name} returned no grid, skipping")
            continue

        grids.append(float_grid)
        grid_names.append(signal_name)
    
    if len(grids) == 0:
        raise ValueError("No valid signals to export")
    
    # Write to .vdb file using C++ VDBWriter
    writer = VDBWriter()
    writer.write_multiple_with_names(grids, grid_names, str(output_path_obj))
    
    logger.info(f"Exported {len(grids)} grids to ParaView: {output_path_obj}")
    return str(output_path_obj)


def export_multiple_grids_to_paraview(
    grids: Dict[str, Any],  # Dict of {name: VoxelGrid}
    output_path: str,
    signal_names: Optional[List[str]] = None,
) -> str:
    """
    Export multiple voxel grids to a single ParaView .vdb file.
    
    Args:
        grids: Dictionary mapping grid names to VoxelGrid instances
        output_path: Path to output .vdb file
        signal_names: Optional list of signal names to export from each grid
    
    Returns:
        Path to created .vdb file
    """
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    all_float_grids: List[FloatGridPtr] = []
    all_grid_names: List[str] = []
    
    for grid_name, voxel_grid in grids.items():
        # Get signals for this grid
        if signal_names is None:
            signals = list(voxel_grid.available_signals)
        else:
            signals = [s for s in signal_names if s in voxel_grid.available_signals]
        
        for signal_name in signals:
            float_grid = voxel_grid.get_grid(signal_name)
            if float_grid is None:
                continue
            # Name format: "grid_name_signal_name"
            full_name = f"{grid_name}_{signal_name}"
            all_float_grids.append(float_grid)
            all_grid_names.append(full_name)
    
    if len(all_float_grids) == 0:
        raise ValueError("No grids to export")
    
    # Write all grids to single .vdb file
    writer = VDBWriter()
    writer.write_multiple_with_names(all_float_grids, all_grid_names, str(output_path_obj))
    
    logger.info(f"Exported {len(all_float_grids)} grids to ParaView: {output_path_obj}")
    return str(output_path_obj)
