"""
AM-QADF Fusion Module

Multi-modal data fusion for voxel domain.
Handles combining multiple signals using various fusion strategies.
All core fusion operations are now C++ wrappers.
"""

from .data_fusion import FusionStrategy, DataFusion

# C++ wrappers (may be None if C++ bindings not available)
try:
    from .voxel_fusion import VoxelFusion
except (ImportError, NotImplementedError):
    VoxelFusion = None

try:
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
except (ImportError, NotImplementedError):
    FusionMethod = None
    WeightedAverageFusion = None
    MedianFusion = None
    QualityBasedFusion = None
    AverageFusion = None
    MaxFusion = None
    MinFusion = None
    get_fusion_method = None

from .fusion_quality import (
    FusionQualityMetrics,
    FusionQualityAssessor,
)

# Note: FusionStrategy and DataFusion are imported from .data_fusion at the top

from .multi_source_fusion import MultiSourceFusion

__all__ = [
    # Data fusion (base classes - Python orchestration)
    "FusionStrategy",
    "DataFusion",
    # Fusion quality (Python quality metrics)
    "FusionQualityMetrics",
    "FusionQualityAssessor",
    # Multi-source fusion (Python orchestration)
    "MultiSourceFusion",
]

# Add C++ wrappers if available
if VoxelFusion is not None:
    __all__.append("VoxelFusion")

if FusionMethod is not None:
    __all__.extend([
        "FusionMethod",
        "WeightedAverageFusion",
        "MedianFusion",
        "QualityBasedFusion",
        "AverageFusion",
        "MaxFusion",
        "MinFusion",
        "get_fusion_method",
    ])
