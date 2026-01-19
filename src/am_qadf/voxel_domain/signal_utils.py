"""
Utility functions for handling signal arrays in voxel grids.

This module provides functions to handle sparse and dense signal arrays,
reconstructing them to proper 3D grid formats for consistent use across notebooks.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


def reconstruct_sparse_signal(
    signal_array: np.ndarray,
    grid_dims: Tuple[int, int, int],
    filled_voxel_indices: Optional[np.ndarray] = None,
    default_value: float = 0.0
) -> np.ndarray:
    """
    Reconstruct a sparse 1D signal array to a full 3D grid array.
    
    Args:
        signal_array: Input signal array (can be 1D sparse or already 3D)
        grid_dims: Expected grid dimensions (nx, ny, nz)
        filled_voxel_indices: Optional array of shape (N, 3) with filled voxel indices.
                            If None, assumes signal_array contains values for all voxels in order.
        default_value: Default value for unfilled voxels
        
    Returns:
        3D numpy array with shape matching grid_dims
        
    Raises:
        ValueError: If signal_array cannot be reconstructed
    """
    if not isinstance(signal_array, np.ndarray):
        signal_array = np.array(signal_array)
    
    expected_size = np.prod(grid_dims)
    
    # If already 3D and correct shape, return as-is
    if signal_array.ndim == 3:
        if signal_array.shape == grid_dims:
            return signal_array
        else:
            logger.warning(
                f"Signal array shape {signal_array.shape} doesn't match expected {grid_dims}. "
                f"Attempting to reshape or reconstruct."
            )
    
    # If 1D and size matches expected, reshape directly
    if signal_array.ndim == 1:
        if signal_array.size == expected_size:
            # Direct reshape - all voxels have values
            return signal_array.reshape(grid_dims)
        elif filled_voxel_indices is not None:
            # Sparse format with known indices
            reconstructed = np.full(grid_dims, default_value, dtype=signal_array.dtype)
            if len(filled_voxel_indices) == signal_array.size:
                # Assign values to filled voxels
                indices = filled_voxel_indices.astype(int)
                reconstructed[indices[:, 0], indices[:, 1], indices[:, 2]] = signal_array
                return reconstructed
            else:
                raise ValueError(
                    f"Number of filled voxel indices ({len(filled_voxel_indices)}) "
                    f"doesn't match signal array size ({signal_array.size})"
                )
        else:
            # Sparse format but no indices provided - cannot reconstruct properly
            # This is the problematic case - we'll log a warning and return a zero-filled array
            logger.warning(
                f"Signal array is sparse (size {signal_array.size}) but doesn't match "
                f"expected grid size ({expected_size}). Cannot reconstruct without filled voxel indices. "
                f"Returning zero-filled array."
            )
            return np.zeros(grid_dims, dtype=signal_array.dtype)
    
    # If 2D, try to interpret as a slice or flatten and reshape
    if signal_array.ndim == 2:
        if signal_array.size == expected_size:
            return signal_array.reshape(grid_dims)
        else:
            logger.warning(
                f"2D signal array size {signal_array.size} doesn't match expected {expected_size}. "
                f"Returning zero-filled array."
            )
            return np.zeros(grid_dims, dtype=signal_array.dtype)
    
    raise ValueError(
        f"Cannot reconstruct signal array with shape {signal_array.shape} "
        f"to grid dimensions {grid_dims}"
    )


def ensure_3d_signals(
    signal_arrays: Dict[str, np.ndarray],
    grid_dims: Tuple[int, int, int],
    filled_voxel_indices: Optional[np.ndarray] = None,
    default_value: float = 0.0
) -> Dict[str, np.ndarray]:
    """
    Ensure all signal arrays in a dictionary are 3D arrays with correct dimensions.
    
    Args:
        signal_arrays: Dictionary mapping signal names to signal arrays
        grid_dims: Expected grid dimensions (nx, ny, nz)
        filled_voxel_indices: Optional array of shape (N, 3) with filled voxel indices
        default_value: Default value for unfilled voxels
        
    Returns:
        Dictionary with all signals as 3D arrays
    """
    reconstructed = {}
    
    for signal_name, signal_array in signal_arrays.items():
        try:
            reconstructed[signal_name] = reconstruct_sparse_signal(
                signal_array,
                grid_dims,
                filled_voxel_indices,
                default_value
            )
        except Exception as e:
            logger.error(
                f"Failed to reconstruct signal '{signal_name}': {e}. "
                f"Signal shape: {signal_array.shape if hasattr(signal_array, 'shape') else 'unknown'}"
            )
            # Return zero-filled array as fallback
            reconstructed[signal_name] = np.zeros(grid_dims, dtype=np.float32)
    
    return reconstructed


def extract_filled_voxel_indices(
    grid_data: Dict,
    metadata: Optional[Dict] = None
) -> Optional[np.ndarray]:
    """
    Extract filled voxel indices from grid data or metadata.
    
    Args:
        grid_data: Grid data dictionary (from voxel_storage.load_voxel_grid)
        metadata: Optional metadata dictionary
        
    Returns:
        Array of shape (N, 3) with filled voxel indices, or None if not available
    """
    if metadata is None:
        metadata = grid_data.get('metadata', {})
    
    # Try to get from metadata
    filled_indices = metadata.get('filled_voxel_indices')
    if filled_indices is not None:
        return np.array(filled_indices)
    
    # Try to get from grid_data
    if 'filled_voxel_indices' in grid_data:
        return np.array(grid_data['filled_voxel_indices'])
    
    # Try to reconstruct from voxel grid if available
    voxel_grid = grid_data.get('voxel_grid')
    if voxel_grid is not None and hasattr(voxel_grid, 'voxels'):
        # Extract indices from voxel grid
        indices = list(voxel_grid.voxels.keys())
        if indices:
            return np.array(indices)
    
    return None


def prepare_signal_arrays_for_processing(
    loaded_grid_data: Dict,
    grid_dims: Tuple[int, int, int],
    default_value: float = 0.0
) -> Dict[str, np.ndarray]:
    """
    Prepare signal arrays from loaded grid data for processing.
    
    This is the main utility function to use in notebooks. It handles:
    - Loading signals from grid data
    - Reconstructing sparse signals to 3D
    - Ensuring all signals have correct dimensions
    
    Args:
        loaded_grid_data: Grid data dictionary from voxel_storage.load_voxel_grid()
        grid_dims: Expected grid dimensions (nx, ny, nz)
        default_value: Default value for unfilled voxels
        
    Returns:
        Dictionary mapping signal names to 3D numpy arrays
    """
    # Get signal arrays
    signal_arrays = loaded_grid_data.get('signal_arrays', {})
    
    if not signal_arrays:
        # Try to get from metadata
        metadata = loaded_grid_data.get('metadata', {})
        available_signals = metadata.get('available_signals', [])
        if available_signals:
            logger.warning(
                f"Signal arrays not found in grid data, but {len(available_signals)} "
                f"signals are listed in metadata. Cannot reconstruct without signal data."
            )
        return {}
    
    # Try to extract filled voxel indices
    filled_indices = extract_filled_voxel_indices(loaded_grid_data)
    
    # Reconstruct all signals to 3D
    reconstructed = ensure_3d_signals(
        signal_arrays,
        grid_dims,
        filled_indices,
        default_value
    )
    
    return reconstructed
