"""
Fusion Methods

Provides fusion method implementations for voxel domain data.
Includes weighted average, median, quality-based, and other fusion strategies.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Callable
from enum import Enum

# Import FusionStrategy from synchronization module
try:
    from ..synchronization.data_fusion import FusionStrategy, DataFusion
except ImportError:
    # Fallback definition
    class FusionStrategy(Enum):
        AVERAGE = "average"
        WEIGHTED_AVERAGE = "weighted_average"
        MEDIAN = "median"
        MAX = "max"
        MIN = "min"
        FIRST = "first"
        LAST = "last"
        QUALITY_BASED = "quality_based"

    class DataFusion:
        def __init__(self, *args, **kwargs):
            pass


class FusionMethod:
    """
    Base class for fusion method implementations.

    Provides a common interface for different fusion strategies
    that can be applied to voxel domain data.
    """

    def __init__(self, strategy: FusionStrategy):
        """
        Initialize fusion method.

        Args:
            strategy: Fusion strategy to use
        """
        self.strategy = strategy

    def fuse(
        self,
        signals: Dict[str, np.ndarray],
        weights: Optional[Dict[str, float]] = None,
        quality_scores: Optional[Dict[str, float]] = None,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Fuse multiple signals using the configured strategy.

        Args:
            signals: Dictionary mapping signal names to arrays
            weights: Optional weights for each signal
            quality_scores: Optional quality scores for each signal
            mask: Optional mask for valid voxels

        Returns:
            Fused signal array
        """
        if not signals:
            raise ValueError("At least one signal must be provided")

        # Use DataFusion for standard strategies
        fusion_engine = DataFusion(default_strategy=self.strategy)

        # Register quality scores if provided
        if quality_scores:
            for signal_name, score in quality_scores.items():
                fusion_engine.register_source_quality(signal_name, score)

        # Compute weights if quality scores are provided
        if quality_scores and not weights:
            source_names = list(signals.keys())
            weight_array = fusion_engine.compute_weights(source_names, use_quality=True)
            weights = {name: weight_array[i] for i, name in enumerate(source_names)}

        # Fuse using DataFusion
        return fusion_engine.fuse_signals(signals=signals, strategy=self.strategy, weights=weights, mask=mask)


class WeightedAverageFusion(FusionMethod):
    """
    Weighted average fusion method.

    Combines signals using weighted averaging, where weights can be
    based on quality scores, confidence, or user-provided values.
    """

    def __init__(self, default_weights: Optional[Dict[str, float]] = None):
        """
        Initialize weighted average fusion.

        Args:
            default_weights: Default weights for each signal
        """
        super().__init__(FusionStrategy.WEIGHTED_AVERAGE)
        self.default_weights = default_weights or {}

    def fuse(
        self,
        signals: Dict[str, np.ndarray],
        weights: Optional[Dict[str, float]] = None,
        quality_scores: Optional[Dict[str, float]] = None,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Fuse signals using weighted average.

        Args:
            signals: Dictionary mapping signal names to arrays
            weights: Optional weights for each signal (if None, uses quality_scores or equal weights)
            quality_scores: Optional quality scores for each signal
            mask: Optional mask for valid voxels

        Returns:
            Fused signal array
        """
        # Use default weights if not provided
        if weights is None:
            weights = self.default_weights.copy()

        return super().fuse(signals, weights, quality_scores, mask)


class MedianFusion(FusionMethod):
    """
    Median fusion method.

    Combines signals by taking the median value at each voxel.
    Robust to outliers.
    """

    def __init__(self):
        """Initialize median fusion."""
        super().__init__(FusionStrategy.MEDIAN)

    def fuse(
        self,
        signals: Dict[str, np.ndarray],
        weights: Optional[Dict[str, float]] = None,
        quality_scores: Optional[Dict[str, float]] = None,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Fuse signals using median.

        Args:
            signals: Dictionary mapping signal names to arrays
            weights: Ignored for median fusion
            quality_scores: Ignored for median fusion
            mask: Optional mask for valid voxels

        Returns:
            Fused signal array (median at each voxel)
        """
        return super().fuse(signals, None, None, mask)


class QualityBasedFusion(FusionMethod):
    """
    Quality-based fusion method.

    Selects the signal with the highest quality score at each voxel.
    Falls back to weighted average if quality scores are not available.
    """

    def __init__(self):
        """Initialize quality-based fusion."""
        super().__init__(FusionStrategy.QUALITY_BASED)

    def fuse(
        self,
        signals: Dict[str, np.ndarray],
        weights: Optional[Dict[str, float]] = None,
        quality_scores: Optional[Dict[str, float]] = None,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Fuse signals using quality-based selection.

        Args:
            signals: Dictionary mapping signal names to arrays
            weights: Ignored for quality-based fusion
            quality_scores: Required for quality-based fusion
            mask: Optional mask for valid voxels

        Returns:
            Fused signal array (highest quality signal at each voxel)
        """
        if not quality_scores:
            # Fallback to weighted average if no quality scores
            return WeightedAverageFusion().fuse(signals, weights, quality_scores, mask)

        return super().fuse(signals, None, quality_scores, mask)


class AverageFusion(FusionMethod):
    """
    Simple average fusion method.

    Combines signals by taking the simple average at each voxel.
    """

    def __init__(self):
        """Initialize average fusion."""
        super().__init__(FusionStrategy.AVERAGE)

    def fuse(
        self,
        signals: Dict[str, np.ndarray],
        weights: Optional[Dict[str, float]] = None,
        quality_scores: Optional[Dict[str, float]] = None,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Fuse signals using simple average.

        Args:
            signals: Dictionary mapping signal names to arrays
            weights: Ignored for average fusion
            quality_scores: Ignored for average fusion
            mask: Optional mask for valid voxels

        Returns:
            Fused signal array (average at each voxel)
        """
        return super().fuse(signals, None, None, mask)


class MaxFusion(FusionMethod):
    """
    Maximum fusion method.

    Combines signals by taking the maximum value at each voxel.
    """

    def __init__(self):
        """Initialize max fusion."""
        super().__init__(FusionStrategy.MAX)

    def fuse(
        self,
        signals: Dict[str, np.ndarray],
        weights: Optional[Dict[str, float]] = None,
        quality_scores: Optional[Dict[str, float]] = None,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Fuse signals using maximum.

        Args:
            signals: Dictionary mapping signal names to arrays
            weights: Ignored for max fusion
            quality_scores: Ignored for max fusion
            mask: Optional mask for valid voxels

        Returns:
            Fused signal array (maximum at each voxel)
        """
        return super().fuse(signals, None, None, mask)


class MinFusion(FusionMethod):
    """
    Minimum fusion method.

    Combines signals by taking the minimum value at each voxel.
    """

    def __init__(self):
        """Initialize min fusion."""
        super().__init__(FusionStrategy.MIN)

    def fuse(
        self,
        signals: Dict[str, np.ndarray],
        weights: Optional[Dict[str, float]] = None,
        quality_scores: Optional[Dict[str, float]] = None,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Fuse signals using minimum.

        Args:
            signals: Dictionary mapping signal names to arrays
            weights: Ignored for min fusion
            quality_scores: Ignored for min fusion
            mask: Optional mask for valid voxels

        Returns:
            Fused signal array (minimum at each voxel)
        """
        return super().fuse(signals, None, None, mask)


def get_fusion_method(strategy: FusionStrategy) -> FusionMethod:
    """
    Get a fusion method instance for the given strategy.

    Args:
        strategy: Fusion strategy

    Returns:
        FusionMethod instance
    """
    strategy_map = {
        FusionStrategy.WEIGHTED_AVERAGE: WeightedAverageFusion,
        FusionStrategy.MEDIAN: MedianFusion,
        FusionStrategy.QUALITY_BASED: QualityBasedFusion,
        FusionStrategy.AVERAGE: AverageFusion,
        FusionStrategy.MAX: MaxFusion,
        FusionStrategy.MIN: MinFusion,
    }

    fusion_class = strategy_map.get(strategy, AverageFusion)
    return fusion_class()


__all__ = [
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
