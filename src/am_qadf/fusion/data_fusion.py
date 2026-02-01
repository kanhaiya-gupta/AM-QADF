"""
Data Fusion - C++ Wrapper

Combine multiple signal sources into unified representation.
Handles conflicting data and weighted averaging strategies.

This module uses C++ GridFusion for core fusion operations.
All core computation is done in C++.

Note: This module was moved from synchronization to fusion as it deals with
signal combination (fusion), not temporal/spatial alignment (synchronization).
"""

from typing import Optional, Dict, List, Tuple, Any, Callable
import numpy as np
from enum import Enum

from am_qadf_native.fusion import GridFusion
from am_qadf_native import numpy_to_openvdb, openvdb_to_numpy


class FusionStrategy(Enum):
    """Strategies for fusing multiple data sources."""

    AVERAGE = "average"  # Simple average
    WEIGHTED_AVERAGE = "weighted_average"  # Weighted by confidence/quality
    MEDIAN = "median"  # Median value
    MAX = "max"  # Maximum value
    MIN = "min"  # Minimum value
    FIRST = "first"  # First available value
    LAST = "last"  # Last available value
    QUALITY_BASED = "quality_based"  # Use highest quality source


class DataFusion:
    """
    Fuse multiple signal sources into unified representation.

    Handles:
    - Combining signals from different sources
    - Resolving conflicts
    - Weighted averaging based on quality/confidence
    """

    def __init__(
        self,
        default_strategy: FusionStrategy = FusionStrategy.WEIGHTED_AVERAGE,
        default_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize data fusion.

        Args:
            default_strategy: Default fusion strategy
            default_weights: Default weights for each source (if None, equal weights)
        """
        self.default_strategy = default_strategy
        self.default_weights = default_weights or {}
        self._source_qualities: Dict[str, float] = {}  # source_name -> quality score
        
        # Initialize C++ fusion engine (required)
        if not CPP_AVAILABLE:
            raise ImportError(
                "C++ bindings not available. "
                "Please build am_qadf_native with pybind11 bindings."
            )
        self._cpp_fusion = GridFusion()

    def register_source_quality(self, source_name: str, quality_score: float):
        """
        Register quality score for a data source.

        Args:
            source_name: Name of data source
            quality_score: Quality score (0.0 to 1.0, higher is better)
        """
        self._source_qualities[source_name] = max(0.0, min(1.0, quality_score))

    def compute_weights(self, source_names: List[str], use_quality: bool = True) -> np.ndarray:
        """
        Compute weights for data sources.

        Args:
            source_names: List of source names
            use_quality: Whether to use quality scores for weighting

        Returns:
            Array of weights (normalized to sum to 1.0)
        """
        if use_quality and self._source_qualities:
            # Use quality scores as weights
            weights = np.array([self._source_qualities.get(name, 0.5) for name in source_names])
        else:
            # Use default weights or equal weights
            weights = np.array([self.default_weights.get(name, 1.0) for name in source_names])

        # Normalize
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(len(source_names)) / len(source_names)

        return weights

    def fuse_signals(
        self,
        signals: Dict[str, np.ndarray],
        strategy: Optional[FusionStrategy] = None,
        weights: Optional[Dict[str, float]] = None,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Fuse multiple signals into single signal.

        Args:
            signals: Dictionary mapping source names to signal arrays
            strategy: Fusion strategy (if None, uses default)
            weights: Custom weights for sources (if None, computed automatically)
            mask: Optional mask for valid voxels (True = valid)

        Returns:
            Fused signal array
        """
        if len(signals) == 0:
            raise ValueError("No signals provided")

        if len(signals) == 1:
            return list(signals.values())[0]

        strategy = strategy or self.default_strategy

        # Get all signal arrays
        source_names = list(signals.keys())
        signal_arrays = [signals[name] for name in source_names]
        
        # Check if arrays have compatible shapes
        if len(signal_arrays) == 0:
            raise ValueError("No signal arrays provided")
        
        first_shape = signal_arrays[0].shape
        if not all(arr.shape == first_shape for arr in signal_arrays):
            raise ValueError("All signal arrays must have the same shape")

        # Check if strategy is supported by C++ GridFusion
        supported_strategies = {
            FusionStrategy.AVERAGE,
            FusionStrategy.WEIGHTED_AVERAGE,
            FusionStrategy.MEDIAN,
            FusionStrategy.MAX,
            FusionStrategy.MIN,
        }
        
        if strategy not in supported_strategies:
            raise ValueError(
                f"Strategy {strategy} is not supported. "
                f"Supported strategies: {[s.value for s in supported_strategies]}"
            )
        
        # Convert NumPy arrays to OpenVDB FloatGrid
        openvdb_grids = []
        for arr in signal_arrays:
            # Replace NaN with 0 for OpenVDB (OpenVDB doesn't support NaN)
            arr_clean = np.nan_to_num(arr, nan=0.0)
            # Use a default resolution (1.0) - actual resolution doesn't matter for fusion
            grid = numpy_to_openvdb(arr_clean, resolution=1.0)
            openvdb_grids.append(grid)
        
        # Fuse using C++
        if strategy == FusionStrategy.WEIGHTED_AVERAGE and weights is not None:
            # Use weighted fusion
            weight_list = [weights.get(name, 1.0) for name in source_names]
            fused_grid = self._cpp_fusion.fuse_weighted(openvdb_grids, weight_list)
        else:
            # Map strategy to C++ string
            strategy_map = {
                FusionStrategy.AVERAGE: "weighted_average",  # C++ uses weighted_average for average
                FusionStrategy.WEIGHTED_AVERAGE: "weighted_average",
                FusionStrategy.MEDIAN: "median",
                FusionStrategy.MAX: "max",
                FusionStrategy.MIN: "min",
            }
            cpp_strategy = strategy_map.get(strategy, "weighted_average")
            fused_grid = self._cpp_fusion.fuse(openvdb_grids, cpp_strategy)
        
        # Convert back to NumPy
        fused = openvdb_to_numpy(fused_grid)
        
        # Ensure shape matches input
        if fused.shape != first_shape:
            # Reshape if needed (shouldn't happen, but safety check)
            if fused.size == np.prod(first_shape):
                fused = fused.reshape(first_shape)
            else:
                raise ValueError(
                    f"Shape mismatch: fused shape {fused.shape} != input shape {first_shape}"
                )
        
        # Apply mask if provided
        if mask is not None:
            fused = np.where(mask, fused, 0.0)
        else:
            # Replace any remaining NaN/0 with 0
            fused = np.nan_to_num(fused, nan=0.0)
        
        return fused

    def fuse_multiple_signals(
        self,
        signal_dicts: List[Dict[str, np.ndarray]],
        signal_names: List[str],
        strategy: Optional[FusionStrategy] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Fuse multiple signals from multiple sources.

        Args:
            signal_dicts: List of signal dictionaries, one per source
            signal_names: List of signal names to fuse
            strategy: Fusion strategy (if None, uses default)

        Returns:
            Dictionary of fused signals
        """
        fused_signals = {}

        for signal_name in signal_names:
            # Collect signal from each source
            signals = {}
            for i, signal_dict in enumerate(signal_dicts):
                if signal_name in signal_dict:
                    signals[f"source_{i}"] = signal_dict[signal_name]

            if len(signals) > 0:
                fused_signals[signal_name] = self.fuse_signals(signals, strategy=strategy)

        return fused_signals

    def handle_conflicts(
        self,
        signals: Dict[str, np.ndarray],
        conflict_threshold: float = 0.1,
        method: str = "weighted_average",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect and handle conflicting data between sources.

        Args:
            signals: Dictionary mapping source names to signal arrays
            conflict_threshold: Relative difference threshold for conflict detection
            method: Method to resolve conflicts ('weighted_average', 'quality_based', 'median')

        Returns:
            Tuple of (fused_signal, conflict_mask)
        """
        if len(signals) < 2:
            fused = list(signals.values())[0] if signals else np.array([])
            conflict_mask = np.zeros_like(fused, dtype=bool)
            return fused, conflict_mask

        source_names = list(signals.keys())
        signal_arrays = [signals[name] for name in source_names]
        stacked = np.stack(signal_arrays, axis=0)

        # Compute statistics
        mean_signal = np.nanmean(stacked, axis=0)
        std_signal = np.nanstd(stacked, axis=0)

        # Detect conflicts: high standard deviation relative to mean
        relative_std = np.where(np.abs(mean_signal) > 1e-10, std_signal / np.abs(mean_signal), std_signal)
        conflict_mask = relative_std > conflict_threshold

        # Resolve conflicts
        if method == "weighted_average":
            weights = self.compute_weights(source_names, use_quality=True)
            weight_shape = (len(source_names),) + (1,) * (stacked.ndim - 1)
            weights = weights.reshape(weight_shape)
            fused = np.nansum(stacked * weights, axis=0) / np.nansum(weights * ~np.isnan(stacked), axis=0)
        elif method == "quality_based":
            quality_scores = np.array([self._source_qualities.get(name, 0.5) for name in source_names])
            best_idx = np.argmax(quality_scores)
            fused = stacked[best_idx]
        elif method == "median":
            fused = np.nanmedian(stacked, axis=0)
        else:
            fused = np.nanmean(stacked, axis=0)

        # Replace NaN
        fused = np.nan_to_num(fused, nan=0.0)

        return fused, conflict_mask

    def compute_fusion_quality(self, signals: Dict[str, np.ndarray], fused: np.ndarray) -> Dict[str, float]:
        """
        Compute quality metrics for fused signal.

        Args:
            signals: Dictionary of source signals
            fused: Fused signal array

        Returns:
            Dictionary of quality metrics
        """
        metrics = {}

        # Agreement between sources
        if len(signals) > 1:
            signal_arrays = list(signals.values())
            stacked = np.stack(signal_arrays, axis=0)
            std = np.nanstd(stacked, axis=0)
            mean = np.nanmean(stacked, axis=0)

            # Coefficient of variation (lower is better)
            cv = np.where(np.abs(mean) > 1e-10, std / np.abs(mean), np.inf)
            metrics["coefficient_of_variation"] = float(np.nanmean(cv))
            metrics["agreement"] = float(1.0 / (1.0 + np.nanmean(cv)))  # Normalized agreement

        # Coverage (fraction of non-zero/non-NaN voxels)
        valid_mask = ~(np.isnan(fused) | (fused == 0))
        metrics["coverage"] = float(np.sum(valid_mask) / fused.size)

        # Signal strength
        metrics["mean"] = float(np.nanmean(fused))
        metrics["std"] = float(np.nanstd(fused))

        return metrics


__all__ = ['FusionStrategy', 'DataFusion']
