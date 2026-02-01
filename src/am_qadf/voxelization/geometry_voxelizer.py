"""
Unified Geometry Voxelizer - C++ Wrapper

Voxelizes both STL geometry and hatching paths together.
This is the primary interface for creating voxel grids with both geometry and process parameters.

This module uses C++ OpenVDB for core voxelization operations.
All core computation is done in C++.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

try:
    # Import from main module (not submodule - bindings are registered directly in am_qadf_native)
    from am_qadf_native import STLVoxelizer, openvdb_to_numpy, numpy_to_openvdb
    try:
        from am_qadf_native import VDBWriter
    except ImportError:
        VDBWriter = None
    # HatchingVoxelizer will be available after C++ implementation
    try:
        from am_qadf_native import HatchingVoxelizer
        HATCHING_VOXELIZER_AVAILABLE = True
    except ImportError:
        HATCHING_VOXELIZER_AVAILABLE = False
        HatchingVoxelizer = None
except ImportError:
    CPP_AVAILABLE = False
    STLVoxelizer = None
    openvdb_to_numpy = None
    numpy_to_openvdb = None
    VDBWriter = None
    HATCHING_VOXELIZER_AVAILABLE = False
    HatchingVoxelizer = None
else:
    CPP_AVAILABLE = True

# Import VoxelGrid
from .uniform_resolution import VoxelGrid


def calculate_adaptive_resolution(
    stl_path: str,
    resolution_ratio: float = 1.0 / 1000.0,
    min_resolution: float = 0.01,
    max_resolution: float = 1.0,
) -> float:
    """
    Calculate adaptive voxel resolution based on geometry size.
    
    Default strategy: Use 1/1000 of the largest dimension of the geometry.
    This provides fine resolution for small parts and reasonable resolution for large parts.
    
    Based on research: Adaptive resolution is standard practice in AM, with resolution
    typically scaling with geometry size to balance quality and performance.
    
    Args:
        stl_path: Path to STL file
        resolution_ratio: Ratio of geometry size to use (default: 1/1000 = 0.001)
        min_resolution: Minimum resolution in mm (default: 0.01 mm = 10 microns)
        max_resolution: Maximum resolution in mm (default: 1.0 mm)
    
    Returns:
        Calculated resolution in mm (clamped between min and max)
    
    Examples:
        Small part (10 mm): 10 * 0.001 = 0.01 mm (10 microns) - very fine
        Medium part (100 mm): 100 * 0.001 = 0.1 mm - good balance  
        Large part (500 mm): 500 * 0.001 = 0.5 mm - reasonable for large parts
    
    References:
        - UpNano Adaptive Resolution technology
        - NIST research on voxel-level control in LPBF
    """
    if not CPP_AVAILABLE or STLVoxelizer is None:
        raise ImportError("C++ bindings required for adaptive resolution calculation")
    
    voxelizer = STLVoxelizer()
    bbox_min_array, bbox_max_array = voxelizer.get_stl_bounding_box(str(stl_path))
    bbox_min = list(bbox_min_array)
    bbox_max = list(bbox_max_array)
    
    # Calculate dimensions
    mesh_size = [bbox_max[i] - bbox_min[i] for i in range(3)]
    max_dimension = max(mesh_size) if mesh_size else 0.0
    
    if max_dimension <= 0:
        raise ValueError(f"Invalid bounding box dimensions: {mesh_size}")
    
    # Calculate adaptive resolution: 1/1000 of largest dimension
    calculated_resolution = max_dimension * resolution_ratio
    
    # Clamp to reasonable bounds (10 microns to 1 mm)
    resolution = max(min_resolution, min(calculated_resolution, max_resolution))
    
    logger.info(
        f"Adaptive resolution: max_dim={max_dimension:.2f} mm, "
        f"ratio={resolution_ratio}, calculated={calculated_resolution:.4f} mm, "
        f"final={resolution:.4f} mm"
    )
    
    return resolution


# Internal helper: Voxelize STL geometry only
def _voxelize_stl_geometry(
    stl_path: str,
    resolution: float,
    aggregation: str = "mean",
    padding: float = 0.0,
    bbox_min: Optional[Tuple[float, float, float]] = None,
    bbox_max: Optional[Tuple[float, float, float]] = None,
) -> VoxelGrid:
    """
    Internal helper: Voxelize STL geometry to get occupancy grid.

    When bbox_min/bbox_max are provided (e.g. union bounding box from
    query_and_transform_points), the grid extent uses that bbox; otherwise
    the extent is taken from the STL.
    """
    stl_path_obj = Path(stl_path)
    if not stl_path_obj.exists():
        raise FileNotFoundError(f"STL file not found: {stl_path}")
    
    if resolution <= 0:
        raise ValueError("Resolution must be greater than 0")
    
    if not CPP_AVAILABLE:
        raise ImportError(
            "C++ bindings not available. "
            "Please build am_qadf_native with pybind11 bindings."
        )
    
    # Use C++ STL voxelizer
    logger.info(f"Voxelizing STL geometry using OpenVDB: {stl_path}")
    voxelizer = STLVoxelizer()
    
    # Grid extent: use provided bbox (e.g. union bounds) or STL bbox
    if bbox_min is not None and bbox_max is not None:
        bbox_min = list(bbox_min)
        bbox_max = list(bbox_max)
        logger.info("Using provided bounding box (e.g. union bounds) for grid extent")
    else:
        bbox_min_array, bbox_max_array = voxelizer.get_stl_bounding_box(str(stl_path))
        bbox_min = list(bbox_min_array)
        bbox_max = list(bbox_max_array)
    
    # Add padding if specified
    if padding > 0:
        bbox_min = [b - padding for b in bbox_min]
        bbox_max = [b + padding for b in bbox_max]
    
    # Voxelize STL using OpenVDB
    # Calculate appropriate half_width to ensure interior is fully voxelized
    # half_width should be large enough to cover the entire mesh interior
    # Rule of thumb: half_width >= max(mesh_dimension) / voxel_size
    mesh_size = [bbox_max[i] - bbox_min[i] for i in range(3)]
    max_dimension = max(mesh_size) if mesh_size else 0.0
    # Use at least 3 voxels, but scale up for large meshes
    calculated_half_width = max(3.0, max_dimension / resolution) if resolution > 0 else 3.0
    # Cap at reasonable maximum (100 voxels) to avoid excessive memory
    half_width = min(calculated_half_width, 100.0)
    
    logger.info(f"Voxelizing STL mesh with resolution: {resolution} mm, half_width: {half_width:.1f} voxels")
    logger.info(f"Mesh dimensions: {mesh_size}, max dimension: {max_dimension:.2f} mm")
    logger.info(f"⚠️ Starting C++ STL voxelization - this may take time for large meshes at fine resolution...")
    
    import time
    voxel_start = time.time()
    openvdb_grid = voxelizer.voxelize_stl(
        str(stl_path),
        resolution,
        half_width=half_width,
        unsigned_distance=False
    )
    voxel_time = time.time() - voxel_start
    logger.info(f"✅ C++ STL voxelization completed in {voxel_time:.2f} seconds ({voxel_time/60:.2f} minutes)")
    
    # Create VoxelGrid with the bounding box
    grid = VoxelGrid(
        bbox_min=tuple(bbox_min),
        bbox_max=tuple(bbox_max),
        resolution=resolution,
        aggregation=aggregation
    )
    
    # Copy OpenVDB grid directly to UniformVoxelGrid - NO NumPy conversion!
    # This preserves OpenVDB's sparse storage and is extremely fast
    logger.info(f"Copying OpenVDB grid to UniformVoxelGrid (direct C++ copy, no NumPy conversion)...")
    copy_start = time.time()
    occupancy_grid = grid._get_or_create_grid("occupancy")
    occupancy_grid.copy_from_grid(openvdb_grid)
    grid.available_signals.add("occupancy")
    copy_time = time.time() - copy_start
    logger.info(f"✅ Grid copy completed in {copy_time:.2f} seconds")
    
    # Get statistics from C++ (no NumPy conversion needed)
    stats = occupancy_grid.get_statistics()
    logger.info(f"Created VoxelGrid with {stats.filled_voxels:,} occupied voxels")
    
    return grid


def create_voxel_grid_from_stl_and_hatching(
    stl_path: str,
    hatching_result: Any,  # QueryResult from hatching_client.query()
    resolution: float,
    line_width: float = 0.1,
    aggregation: str = "mean",
    padding: float = 0.0,
    bbox_min: Optional[Tuple[float, float, float]] = None,
    bbox_max: Optional[Tuple[float, float, float]] = None,
) -> Tuple[VoxelGrid, Dict[str, Any]]:
    """
    Create a VoxelGrid from STL geometry and hatching paths with process parameters.

    When bbox_min/bbox_max are provided (e.g. union bounding box from
    query_and_transform_points), the grid extent uses that bbox so the grid
    covers all aligned sources (hatching + laser_monitoring + ISPM, etc.).
    
    Args:
        stl_path: Path to STL file
        hatching_result: QueryResult from hatching_client.query() with points and signals
        resolution: Voxel size in mm (cubic voxels)
        line_width: Width of hatching line in mm (laser beam width)
        aggregation: How to aggregate multiple values per voxel
        padding: Additional padding around the mesh in mm
        bbox_min: Optional (x_min, y_min, z_min) for grid extent (e.g. union bounds)
        bbox_max: Optional (x_max, y_max, z_max) for grid extent (e.g. union bounds)
    
    Returns:
        Tuple of (VoxelGrid, grids_dict) where:
        - VoxelGrid: Python VoxelGrid with geometry and signals
        - grids_dict: Dict of OpenVDB grids {"power": grid, "velocity": grid, "energy": grid}
    
    Raises:
        FileNotFoundError: If STL file doesn't exist
        ValueError: If resolution is invalid
        ImportError: If C++ bindings not available
    """
    if not CPP_AVAILABLE:
        raise ImportError(
            "C++ bindings not available. "
            "Please build am_qadf_native with pybind11 bindings."
        )
    
    if not HATCHING_VOXELIZER_AVAILABLE:
        raise NotImplementedError(
            "HatchingVoxelizer C++ implementation not yet available. "
            "Please use create_voxel_grid_from_stl_with_signals() for now."
        )
    
    stl_path_obj = Path(stl_path)
    if not stl_path_obj.exists():
        raise FileNotFoundError(f"STL file not found: {stl_path}")
    
    if resolution <= 0:
        raise ValueError("Resolution must be greater than 0")
    
    logger.info(f"Voxelizing STL geometry and hatching paths: {stl_path}")
    
    # 1. Voxelize STL to get geometry (occupancy)
    logger.info("Step 1: Voxelizing STL geometry...")
    geometry_grid = _voxelize_stl_geometry(
        stl_path=stl_path,
        resolution=resolution,
        aggregation=aggregation,
        padding=padding,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
    )
    
    # 2. Voxelize hatching paths to get signals
    logger.info("Step 2: Voxelizing hatching paths with process parameters...")
    
    # Check if we have vectors (new format) or points (old format)
    # New format: vectors from MongoDB with {x1, y1, x2, y2, z, dataindex}
    # Old format: points array with signals
    has_vectors = False
    vectors_data = None
    vectordata_list = None
    
    # Check metadata for vector format indicator
    metadata = getattr(hatching_result, 'metadata', {})
    if isinstance(metadata, dict):
        # Check if we have vector-based data
        format_type = metadata.get('format', 'point-based')
        if format_type == 'vector-based' or 'vectors' in metadata:
            has_vectors = True
            # Try to get vectors from metadata
            if 'vectors' in metadata:
                vectors_data = metadata['vectors']
            elif hasattr(hatching_result, 'vectors'):
                vectors_data = hatching_result.vectors
            
            # Get vectordata
            if 'vectordata' in metadata:
                vectordata_list = metadata['vectordata']
            elif hasattr(hatching_result, 'vectordata'):
                vectordata_list = hatching_result.vectordata
    
    # Get bounding box from geometry grid - pass numpy array directly to C++ (no list conversion needed)
    # C++ can accept numpy arrays directly, avoiding unnecessary Python list creation
    bbox_min = geometry_grid.bbox_min  # Already numpy array, C++ accepts it
    bbox_max = geometry_grid.bbox_max  # Already numpy array, C++ accepts it
    
    hatching_voxelizer = HatchingVoxelizer()
    
    if has_vectors and vectors_data:
        # NEW FORMAT: Vector-based (accurate, preserves structure)
        logger.info("Using vector-based format for voxelization")
        
        # Pass raw Python data structures directly to C++ - ALL parsing happens in C++
        # No Python loops, no array building, no calculations - everything in C++ for performance
        # This is critical for billions of voxels at high resolution
        if len(vectors_data) == 0:
            logger.warning("No hatching vectors found. Returning geometry-only grid.")
            return geometry_grid, {}
        
        # C++ will do ALL the parsing, case-insensitive lookup, array building, and voxelization
        signal_grids = hatching_voxelizer.voxelize_vectors_from_python_data(
            vectors_data,
            vectordata_list if vectordata_list else [],
            resolution,
            line_width,
            bbox_min,
            bbox_max
        )
        
        # Get direction vector grid for arrow visualization in ParaView
        direction_grid = hatching_voxelizer.get_direction_grid()
        if direction_grid is not None:
            # Add direction grid to signal_grids for export (ParaView can visualize Vec3fGrid as arrows)
            signal_grids["direction"] = direction_grid
        
    else:
        # No vectors found - return geometry-only grid
        logger.warning("No hatching vectors found in result. Returning geometry-only grid.")
        return geometry_grid, {}
    
    # 3. Combine geometry and signals into Python VoxelGrid
    logger.info("Step 3: Combining geometry and signals...")
    
    # Copy signal grids directly to VoxelGrid using C++ - NO NumPy conversion!
    # This preserves OpenVDB's sparse storage and is extremely fast
    # Note: direction grid is Vec3fGrid, not FloatGrid, so we skip it for VoxelGrid population
    # It will be exported separately for ParaView visualization
    for signal_name, openvdb_grid in signal_grids.items():
        if signal_name == "direction":
            # Skip direction grid - it's Vec3fGrid, not compatible with FloatGrid VoxelGrid
            continue
        
        # Copy OpenVDB grid directly - preserves sparse storage, no dense array conversion!
        signal_grid = geometry_grid._get_or_create_grid(signal_name)
        signal_grid.copy_from_grid(openvdb_grid)
        geometry_grid.available_signals.add(signal_name)
    
    # Get statistics from C++ (no NumPy conversion needed)
    if "occupancy" in geometry_grid.available_signals:
        occupancy_stats = geometry_grid._get_or_create_grid("occupancy").get_statistics()
        logger.info(f"Created unified voxel grid: {occupancy_stats.filled_voxels:,} occupied voxels with {len(signal_grids)} signal types")
    else:
        logger.info(f"Created unified voxel grid with {len(signal_grids)} signal types")
    
    return geometry_grid, signal_grids


def export_to_paraview(
    stl_path: str,
    hatching_result: Any,
    output_path: str,
    resolution: float = 0.1,
    line_width: float = 0.1,
    padding: float = 0.0,
) -> str:
    """
    Voxelize STL and hatching paths, then export to Paraview (.vdb file).
    
    This is a convenience function that:
    1. Voxelizes STL geometry
    2. Voxelizes hatching paths with signals
    3. Exports all grids to a single .vdb file for Paraview
    
    Args:
        stl_path: Path to STL file
        hatching_result: QueryResult from hatching_client.query()
        output_path: Path to output .vdb file
        resolution: Voxel size in mm
        line_width: Width of hatching line in mm
        padding: Additional padding around the mesh in mm
    
    Returns:
        Path to created .vdb file
    
    The .vdb file will contain:
    - "geometry": Occupancy grid (1.0 = inside, 0.0 = outside)
    - "power": Power signal grid
    - "velocity": Velocity signal grid
    - "energy": Energy signal grid
    - "direction": Direction vector grid (Vec3fGrid) for arrow visualization showing laser scan direction
    """
    if not CPP_AVAILABLE:
        raise ImportError("C++ bindings not available")
    
    # Voxelize both
    geometry_grid, signal_grids = create_voxel_grid_from_stl_and_hatching(
        stl_path=stl_path,
        hatching_result=hatching_result,
        resolution=resolution,
        line_width=line_width,
        padding=padding
    )

    # Get geometry occupancy grid directly from geometry_grid - NO re-voxelization!
    # Use the OpenVDB grid that was already created - direct access, no conversion
    geometry_openvdb = None
    if "occupancy" in geometry_grid.available_signals:
        geometry_openvdb = geometry_grid.get_grid("occupancy")
    
    if geometry_openvdb is None:
        logger.warning("No occupancy grid found in geometry_grid, creating new one...")
        # Fallback: create new grid (shouldn't happen, but handle gracefully)
        stl_voxelizer = STLVoxelizer()
        bbox_min_export_array, bbox_max_export_array = stl_voxelizer.get_stl_bounding_box(str(stl_path))
        bbox_min_export = list(bbox_min_export_array)
        bbox_max_export = list(bbox_max_export_array)
        mesh_size_export = [bbox_max_export[i] - bbox_min_export[i] for i in range(3)]
        max_dimension_export = max(mesh_size_export) if mesh_size_export else 0.0
        calculated_half_width_export = max(3.0, max_dimension_export / resolution) if resolution > 0 else 3.0
        half_width_export = min(calculated_half_width_export, 100.0)
        geometry_openvdb = stl_voxelizer.voxelize_stl(
            str(stl_path),
            resolution,
            half_width=half_width_export,
            unsigned_distance=False
        )
    
    # Combine all grids - all are OpenVDB grids, no conversion needed!
    all_grids = {"geometry": geometry_openvdb}
    all_grids.update(signal_grids)
    
    # Write to .vdb file
    writer = VDBWriter()
    writer.write_multiple_with_names(output_path, all_grids)
    
    logger.info(f"Exported to Paraview: {output_path} with {len(all_grids)} grids")
    
    return output_path


def get_stl_bounding_box(stl_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get bounding box from STL file without creating a full voxel grid.
    
    Utility function for getting STL bounding box.
    Uses C++ OpenVDB implementation for fast STL parsing.
    
    Args:
        stl_path: Path to STL file
    
    Returns:
        Tuple of (bbox_min, bbox_max) as numpy arrays
    """
    stl_path_obj = Path(stl_path)
    if not stl_path_obj.exists():
        raise FileNotFoundError(f"STL file not found: {stl_path}")
    
    if not CPP_AVAILABLE:
        raise ImportError(
            "C++ bindings not available. "
            "Please build am_qadf_native with pybind11 bindings."
        )
    
    # Use C++ STL voxelizer to get bounding box
    voxelizer = STLVoxelizer()
    bbox_min_array, bbox_max_array = voxelizer.get_stl_bounding_box(str(stl_path))
    
    return np.array(list(bbox_min_array)), np.array(list(bbox_max_array))
