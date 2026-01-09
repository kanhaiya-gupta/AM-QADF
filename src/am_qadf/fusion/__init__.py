"""
AM-QADF Fusion Module

Multi-modal data fusion for voxel domain.
Handles combining multiple signals using various fusion strategies.
"""

from .voxel_fusion import VoxelFusion

from .fusion_quality import (
    FusionQualityMetrics,
    FusionQualityAssessor,
)

from .fusion_methods import (
    FusionStrategy,
    FusionMethod,
    WeightedAverageFusion,
    MedianFusion,
    QualityBasedFusion,
    AverageFusion,
    MaxFusion,
    MinFusion,
    get_fusion_method,
)

__all__ = [
    # Voxel fusion
    "VoxelFusion",
    # Fusion quality
    "FusionQualityMetrics",
    "FusionQualityAssessor",
    # Fusion methods
    "FusionStrategy",
    "FusionMethod",
    "WeightedAverageFusion",
    "MedianFusion",
    "QualityBasedFusion",
    "AverageFusion",
    "MaxFusion",
    "MinFusion",
    "get_fusion_method",
]
