"""
Fusion Methods - C++ Wrapper

Thin Python wrapper for C++ fusion strategy implementations.
All core computation is done in C++.
"""

from typing import List, Optional
from abc import ABC, abstractmethod

try:
    from am_qadf_native.fusion import (
        FusionStrategy,
        WeightedAverageStrategy,
        MaxStrategy,
        MinStrategy,
        MedianStrategy,
    )
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
    FusionStrategy = None
    WeightedAverageStrategy = None
    MaxStrategy = None
    MinStrategy = None
    MedianStrategy = None


class FusionMethod(ABC):
    """
    Base class for fusion methods - C++ wrapper.
    
    This is a thin wrapper around C++ fusion strategy implementations.
    """

    @abstractmethod
    def fuse_values(self, values: List[float], weights: Optional[List[float]] = None) -> float:
        """Fuse a list of values into a single value."""
        pass


class WeightedAverageFusion(FusionMethod):
    """Weighted average fusion - C++ wrapper."""

    def __init__(self, weights: Optional[List[float]] = None):
        """
        Initialize weighted average fusion.

        Args:
            weights: Optional weights for values (if None, equal weights)
        """
        if not CPP_AVAILABLE:
            raise ImportError("C++ bindings not available")
        
        weights_cpp = weights if weights else [1.0] * 10  # Default weights
        self._strategy = WeightedAverageStrategy(weights_cpp)
        self.weights = weights

    def fuse_values(self, values: List[float], weights: Optional[List[float]] = None) -> float:
        """Fuse values using weighted average."""
        if weights is not None:
            self._strategy = WeightedAverageStrategy(weights)
        return self._strategy.fuse_values(values)


class AverageFusion(FusionMethod):
    """Simple average fusion - C++ wrapper."""

    def __init__(self):
        """Initialize average fusion."""
        if not CPP_AVAILABLE:
            raise ImportError("C++ bindings not available")
        # Use weighted average with equal weights
        self._strategy = WeightedAverageStrategy([1.0] * 10)

    def fuse_values(self, values: List[float], weights: Optional[List[float]] = None) -> float:
        """Fuse values using simple average."""
        if weights is None:
            weights = [1.0] * len(values)
        self._strategy = WeightedAverageStrategy(weights)
        return self._strategy.fuse_values(values)


class MaxFusion(FusionMethod):
    """Maximum value fusion - C++ wrapper."""

    def __init__(self):
        """Initialize max fusion."""
        if not CPP_AVAILABLE:
            raise ImportError("C++ bindings not available")
        self._strategy = MaxStrategy()

    def fuse_values(self, values: List[float], weights: Optional[List[float]] = None) -> float:
        """Fuse values using maximum."""
        return self._strategy.fuse_values(values)


class MinFusion(FusionMethod):
    """Minimum value fusion - C++ wrapper."""

    def __init__(self):
        """Initialize min fusion."""
        if not CPP_AVAILABLE:
            raise ImportError("C++ bindings not available")
        self._strategy = MinStrategy()

    def fuse_values(self, values: List[float], weights: Optional[List[float]] = None) -> float:
        """Fuse values using minimum."""
        return self._strategy.fuse_values(values)


class MedianFusion(FusionMethod):
    """Median value fusion - C++ wrapper."""

    def __init__(self):
        """Initialize median fusion."""
        if not CPP_AVAILABLE:
            raise ImportError("C++ bindings not available")
        self._strategy = MedianStrategy()

    def fuse_values(self, values: List[float], weights: Optional[List[float]] = None) -> float:
        """Fuse values using median."""
        return self._strategy.fuse_values(values)


class QualityBasedFusion(FusionMethod):
    """
    Quality-based fusion - Python implementation.
    
    NOTE: This is not in C++ yet, so uses Python logic.
    """

    def __init__(self, quality_scores: Optional[List[float]] = None):
        """
        Initialize quality-based fusion.

        Args:
            quality_scores: Quality scores for each source (higher is better)
        """
        self.quality_scores = quality_scores

    def fuse_values(self, values: List[float], weights: Optional[List[float]] = None) -> float:
        """Fuse values using quality-based selection."""
        if self.quality_scores and len(self.quality_scores) == len(values):
            # Use source with highest quality
            best_idx = max(range(len(self.quality_scores)), key=lambda i: self.quality_scores[i])
            return values[best_idx]
        else:
            # Fallback to average
            return sum(values) / len(values) if values else 0.0


def get_fusion_method(method_name: str, **kwargs) -> FusionMethod:
    """
    Get fusion method by name.

    Args:
        method_name: Method name ('weighted_average', 'average', 'max', 'min', 'median', 'quality_based')
        **kwargs: Additional arguments for specific methods

    Returns:
        FusionMethod instance
    """
    if method_name == "weighted_average":
        return WeightedAverageFusion(weights=kwargs.get("weights"))
    elif method_name == "average":
        return AverageFusion()
    elif method_name == "max":
        return MaxFusion()
    elif method_name == "min":
        return MinFusion()
    elif method_name == "median":
        return MedianFusion()
    elif method_name == "quality_based":
        return QualityBasedFusion(quality_scores=kwargs.get("quality_scores"))
    else:
        raise ValueError(f"Unknown fusion method: {method_name}")
