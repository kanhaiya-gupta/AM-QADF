"""
STL-based Voxel Grid Creation

Creates voxel grids directly from STL files using PyVista's voxelization.
This is more accurate than bounding box-based grids as it only creates
voxels that intersect with the actual geometry.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
import pyvista as pv

logger = logging.getLogger(__name__)

from .voxel_grid import VoxelGrid


def create_voxel_grid_from_stl(
    stl_path: str,
    resolution: float,
    aggregation: str = "mean",
    padding: float = 0.0,
) -> VoxelGrid:
    """
    Create a VoxelGrid directly from an STL file.
    
    This method uses PyVista to voxelize the STL mesh, then creates a VoxelGrid
    that only contains voxels that intersect with the geometry. This is more
    accurate than bounding box-based grids.
    
    Args:
        stl_path: Path to STL file
        resolution: Voxel size in mm (cubic voxels)
        aggregation: How to aggregate multiple values per voxel ('mean', 'max', 'min', 'sum')
        padding: Additional padding around the mesh in mm (default: 0.0)
    
    Returns:
        VoxelGrid object with voxels only where geometry exists
    
    Raises:
        FileNotFoundError: If STL file doesn't exist
        ValueError: If resolution is invalid
    """
    stl_path_obj = Path(stl_path)
    if not stl_path_obj.exists():
        raise FileNotFoundError(f"STL file not found: {stl_path}")
    
    if resolution <= 0:
        raise ValueError("Resolution must be greater than 0")
    
    # Load STL mesh
    logger.info(f"Loading STL file: {stl_path}")
    stl_mesh = pv.read(str(stl_path))
    
    # Get bounding box from mesh
    bbox = stl_mesh.bounds
    bbox_min = np.array([bbox[0], bbox[2], bbox[4]])
    bbox_max = np.array([bbox[1], bbox[3], bbox[5]])
    
    # Add padding if specified
    if padding > 0:
        bbox_min = bbox_min - padding
        bbox_max = bbox_max + padding
    
    # Voxelize the mesh using PyVista
    logger.info(f"Voxelizing STL mesh with resolution: {resolution} mm")
    voxelized_mesh = stl_mesh.voxelize(spacing=resolution)
    
    # Get voxel centers from voxelized mesh
    # PyVista's voxelize returns an UnstructuredGrid with cell centers
    voxel_centers = voxelized_mesh.cell_centers().points
    
    logger.info(f"Voxelized mesh contains {len(voxel_centers)} voxels")
    
    # Create VoxelGrid with the bounding box
    # The grid will be sized to fit the bbox, but we'll only populate voxels that exist
    grid = VoxelGrid(
        bbox_min=tuple(bbox_min),
        bbox_max=tuple(bbox_max),
        resolution=resolution,
        aggregation=aggregation
    )
    
    # Populate grid with voxel centers from the voxelized mesh
    # Convert voxel centers to voxel indices and mark them as existing
    for center in voxel_centers:
        x, y, z = center
        # Use the grid's method to convert world coordinates to voxel indices
        voxel_idx = grid._world_to_voxel(x, y, z)
        
        # Create a VoxelData entry for this voxel (empty signals for now)
        # Signals can be added later via add_point() or set_signal()
        if voxel_idx not in grid.voxels:
            from ..core.entities import VoxelData
            grid.voxels[voxel_idx] = VoxelData()
            grid.voxels[voxel_idx].count = 1  # Mark as existing voxel
    
    logger.info(f"Created VoxelGrid with {len(grid.voxels)} populated voxels out of {np.prod(grid.dims)} total possible")
    
    return grid


def create_voxel_grid_from_stl_with_signals(
    stl_path: str,
    resolution: float,
    points: np.ndarray,
    signals: List[Dict[str, float]],
    aggregation: str = "mean",
    padding: float = 0.0,
) -> VoxelGrid:
    """
    Create a VoxelGrid from STL file and populate it with signal data.
    
    This method creates a voxel grid from the STL geometry, then maps
    signal data points to the appropriate voxels. This ensures that
    signals are only mapped to voxels that actually contain geometry.
    
    Args:
        stl_path: Path to STL file
        resolution: Voxel size in mm
        points: Array of points (N, 3) with signal data coordinates
        signals: List of signal dictionaries (one per point)
        aggregation: How to aggregate multiple values per voxel
        padding: Additional padding around the mesh in mm
    
    Returns:
        VoxelGrid object with geometry-based voxels and signal data
    """
    # First create the grid from STL
    grid = create_voxel_grid_from_stl(
        stl_path=stl_path,
        resolution=resolution,
        aggregation=aggregation,
        padding=padding
    )
    
    # Now add signal data points
    # Only points that fall within existing voxels will be added
    for i, point in enumerate(points):
        x, y, z = point
        voxel_idx = grid._world_to_voxel(x, y, z)
        
        # Only add if this voxel exists (was created from STL geometry)
        if voxel_idx in grid.voxels:
            grid.add_point(x, y, z, signals[i] if i < len(signals) else {})
    
    logger.info(f"Added {len(points)} signal points to STL-based voxel grid")
    
    return grid


def get_stl_bounding_box(stl_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get bounding box from STL file without creating a full voxel grid.
    
    Args:
        stl_path: Path to STL file
    
    Returns:
        Tuple of (bbox_min, bbox_max) as numpy arrays
    """
    stl_path_obj = Path(stl_path)
    if not stl_path_obj.exists():
        raise FileNotFoundError(f"STL file not found: {stl_path}")
    
    stl_mesh = pv.read(str(stl_path))
    bbox = stl_mesh.bounds
    bbox_min = np.array([bbox[0], bbox[2], bbox[4]])
    bbox_max = np.array([bbox[1], bbox[3], bbox[5]])
    
    return bbox_min, bbox_max
