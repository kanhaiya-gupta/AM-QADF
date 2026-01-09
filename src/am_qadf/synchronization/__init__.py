"""
AM-QADF Synchronization Module

Temporal and spatial alignment of multi-source data.
Handles time-to-layer mapping, coordinate transformations, and data fusion preparation.
"""

from .temporal_alignment import (
    TimePoint,
    LayerTimeMapper,
    TemporalAligner,
)

from .spatial_transformation import (
    TransformationMatrix,
    SpatialTransformer,
    TransformationManager,
)

from .data_fusion import (
    FusionStrategy,
    DataFusion,
)

from .alignment_storage import (
    AlignmentStorage,
)

__all__ = [
    # Temporal alignment
    "TimePoint",
    "LayerTimeMapper",
    "TemporalAligner",
    # Spatial transformation
    "TransformationMatrix",
    "SpatialTransformer",
    "TransformationManager",
    # Data fusion
    "FusionStrategy",
    "DataFusion",
    # Storage
    "AlignmentStorage",
]
