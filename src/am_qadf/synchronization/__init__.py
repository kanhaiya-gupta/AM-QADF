"""
AM-QADF Synchronization Module

Temporal and spatial alignment of multi-source data.
Handles time-to-layer mapping, coordinate transformations, and alignment storage.

Note: DataFusion has been moved to the fusion module as it deals with
signal combination (fusion), not temporal/spatial alignment (synchronization).
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
    # Storage
    "AlignmentStorage",
]
