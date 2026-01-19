"""
AM-QADF Fusion Module

Multi-modal data fusion for voxel domain.
Handles combining multiple signals using various fusion strategies.
"""

from .data_fusion import FusionStrategy, DataFusion

from .voxel_fusion import VoxelFusion

from .fusion_quality import (
    FusionQualityMetrics,
    FusionQualityAssessor,
)

from .fusion_methods import (
    FusionMethod,
    WeightedAverageFusion,
    MedianFusion,
    QualityBasedFusion,
    AverageFusion,
    MaxFusion,
    MinFusion,
    get_fusion_method,
)

# Note: FusionStrategy and DataFusion are imported from .data_fusion at the top

from .multi_source_fusion import MultiSourceFusion

__all__ = [
    # Data fusion (base classes)
    "FusionStrategy",
    "DataFusion",
    # Voxel fusion
    "VoxelFusion",
    # Fusion quality
    "FusionQualityMetrics",
    "FusionQualityAssessor",
    # Fusion methods
    "FusionMethod",
    "WeightedAverageFusion",
    "MedianFusion",
    "QualityBasedFusion",
    "AverageFusion",
    "MaxFusion",
    "MinFusion",
    "get_fusion_method",
    # Multi-source fusion
    "MultiSourceFusion",
]
