"""
AM-QADF Voxel Domain Module

Voxel domain client and storage.
Main orchestrator for creating unified voxel domain representations.
"""

from .voxel_domain_client import (
    VoxelDomainClient,
)

from .voxel_storage import (
    VoxelGridStorage,
)

from .grid_naming import (
    GridNaming,
    GridSource,
    GridType,
    GridStage,
)

from .signal_utils import (
    reconstruct_sparse_signal,
    ensure_3d_signals,
    extract_filled_voxel_indices,
    prepare_signal_arrays_for_processing,
)

__all__ = [
    "VoxelDomainClient",
    "VoxelGridStorage",
    "GridNaming",
    "GridSource",
    "GridType",
    "GridStage",
    "reconstruct_sparse_signal",
    "ensure_3d_signals",
    "extract_filled_voxel_indices",
    "prepare_signal_arrays_for_processing",
]
