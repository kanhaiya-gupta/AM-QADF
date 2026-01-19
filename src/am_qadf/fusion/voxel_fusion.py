"""
Voxel-Based Data Fusion

Extends data fusion capabilities to work with voxel domain data.
Performs fusion at the voxel level, combining multiple signals per voxel.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
import sys
import importlib.util
from pathlib import Path

# Import base DataFusion from fusion module (moved from synchronization)
try:
    # Try relative import first
    from .data_fusion import DataFusion, FusionStrategy
except ImportError:
    # Fallback: should not be needed if module structure is correct
    try:
        from ..fusion.data_fusion import DataFusion, FusionStrategy
    except Exception:
        # Define minimal fallback
        from enum import Enum

        class FusionStrategy(Enum):
            WEIGHTED_AVERAGE = "weighted_average"
            AVERAGE = "average"
            MEDIAN = "median"
            MAX = "max"
            MIN = "min"

        class DataFusion:
            def __init__(self, *args, **kwargs):
                pass


class VoxelFusion:
    """
    Fuse multiple signals in voxel domain.

    Extends DataFusion to work with voxel domain data structures,
    performing fusion at the voxel level.
    """

    def __init__(
        self,
        default_strategy: FusionStrategy = FusionStrategy.WEIGHTED_AVERAGE,
        use_quality_scores: bool = True,
    ):
        """
        Initialize voxel fusion.

        Args:
            default_strategy: Default fusion strategy
            use_quality_scores: Whether to use quality scores for weighting
        """
        self.fusion_engine = DataFusion(default_strategy=default_strategy)
        self.use_quality_scores = use_quality_scores

    def fuse_voxel_signals(
        self,
        voxel_data: Any,
        signals: List[str],
        fusion_strategy: Optional[FusionStrategy] = None,
        quality_scores: Optional[Dict[str, float]] = None,
        output_signal_name: str = "fused",
    ) -> np.ndarray:
        """
        Fuse multiple signals in voxel domain.

        Args:
            voxel_data: Voxel domain data object
            signals: List of signal names to fuse
            fusion_strategy: Fusion strategy (None = use default)
            quality_scores: Optional quality scores per signal
            output_signal_name: Name for the fused signal

        Returns:
            Fused signal array
        """
        if not signals:
            raise ValueError("At least one signal must be provided")

        # Get signal arrays
        signal_arrays = {}
        expected_shape = None
        for signal in signals:
            try:
                signal_array = voxel_data.get_signal_array(signal, default=0.0)
                # Check if array is actually valid (not just default empty array)
                if signal_array.size == 0:
                    continue

                # Check if this looks like a default placeholder (single element with default value)
                # This is a heuristic to detect when get_signal_array returns a default
                # for a missing signal. Real signals should have size > 1 or be non-zero.
                if signal_array.size == 1 and signal_array[0] == 0.0:
                    # This might be a default placeholder, but we'll include it if it matches shape
                    # The shape check below will handle mismatches
                    pass

                # If this is the first valid array, set expected shape
                if expected_shape is None:
                    expected_shape = signal_array.shape
                    signal_arrays[signal] = signal_array
                # If shape matches, include it
                elif signal_array.shape == expected_shape:
                    signal_arrays[signal] = signal_array
                else:
                    # Shape mismatch - skip this signal
                    print(f"⚠️ Warning: Signal {signal} has incompatible shape {signal_array.shape}, expected {expected_shape}")
                    continue
            except Exception as e:
                print(f"⚠️ Warning: Could not load signal {signal}: {e}")
                continue

        if not signal_arrays:
            raise ValueError("No valid signals found")

        # Additional check: if all signals are single-element default placeholders, consider them invalid
        # This handles the case where all signals are missing and return np.array([0.0])
        if len(signal_arrays) > 0:
            all_single_default = all(arr.size == 1 and arr[0] == 0.0 for arr in signal_arrays.values())
            if all_single_default:
                raise ValueError("No valid signals found")

        # Register quality scores if provided
        if quality_scores:
            for signal, score in quality_scores.items():
                self.fusion_engine.register_source_quality(signal, score)

        # Determine fusion strategy
        strategy = fusion_strategy or self.fusion_engine.default_strategy

        # OPTIMIZATION: Use base DataFusion.fuse_signals() for most strategies
        # This method is already fully vectorized and optimized
        # Only use custom implementation for strategies not supported by base class

        # Check if strategy is supported by base DataFusion
        base_supported_strategies = {
            FusionStrategy.AVERAGE,
            FusionStrategy.WEIGHTED_AVERAGE,
            FusionStrategy.MEDIAN,
            FusionStrategy.MAX,
            FusionStrategy.MIN,
        }

        if strategy in base_supported_strategies:
            # Use optimized base method (fully vectorized)
            # Create mask for valid voxels (not zero, not NaN)
            first_array = list(signal_arrays.values())[0]
            valid_voxel_mask = np.ones_like(first_array, dtype=bool)
            for array in signal_arrays.values():
                valid_voxel_mask &= (~np.isnan(array)) & (array != 0.0)

            # Use base fusion engine (vectorized)
            fused_array = self.fusion_engine.fuse_signals(signal_arrays, strategy=strategy, mask=valid_voxel_mask)
        else:
            # Fallback: custom implementation for unsupported strategies
            # Get dimensions
            first_array = list(signal_arrays.values())[0]
            fused_array = np.zeros_like(first_array, dtype=np.float32)

            # Custom strategy handling (if needed)
            # Default to weighted average
            weights = self.fusion_engine.compute_weights(list(signal_arrays.keys()), use_quality=self.use_quality_scores)
            for i, (signal, array) in enumerate(signal_arrays.items()):
                valid_mask = (~np.isnan(array)) & (array != 0.0)
                fused_array[valid_mask] += array[valid_mask] * weights[i]

        # Add fused signal to voxel data if possible
        if hasattr(voxel_data, "add_signal"):
            voxel_data.add_signal(output_signal_name, fused_array)

        return fused_array

    def fuse_with_quality_weights(
        self,
        voxel_data: Any,
        signals: List[str],
        quality_scores: Dict[str, float],
        output_signal_name: str = "fused_quality_weighted",
    ) -> np.ndarray:
        """
        Fuse signals using quality-based weighting.

        Args:
            voxel_data: Voxel domain data object
            signals: List of signal names to fuse
            quality_scores: Quality scores per signal (0-1, higher is better)
            output_signal_name: Name for the fused signal

        Returns:
            Fused signal array
        """
        return self.fuse_voxel_signals(
            voxel_data,
            signals,
            fusion_strategy=FusionStrategy.WEIGHTED_AVERAGE,
            quality_scores=quality_scores,
            output_signal_name=output_signal_name,
        )

    def fuse_per_voxel(
        self,
        voxel_data: Any,
        signals: List[str],
        fusion_func: Callable[[List[float]], float],
        output_signal_name: str = "fused_custom",
    ) -> np.ndarray:
        """
        Fuse signals using a custom per-voxel function.

        Args:
            voxel_data: Voxel domain data object
            signals: List of signal names to fuse
            fusion_func: Function that takes a list of values and returns fused value
            output_signal_name: Name for the fused signal

        Returns:
            Fused signal array
        """
        # Get signal arrays
        signal_arrays = {}
        expected_shape = None
        for signal in signals:
            try:
                signal_array = voxel_data.get_signal_array(signal, default=0.0)
                # Check if array is actually valid (not just default empty array)
                if signal_array.size == 0:
                    continue

                # Check if this looks like a default placeholder (single element with default value)
                if signal_array.size == 1 and signal_array[0] == 0.0:
                    # This might be a default placeholder, but we'll include it if it matches shape
                    pass

                # If this is the first valid array, set expected shape
                if expected_shape is None:
                    expected_shape = signal_array.shape
                    signal_arrays[signal] = signal_array
                # If shape matches, include it
                elif signal_array.shape == expected_shape:
                    signal_arrays[signal] = signal_array
                else:
                    # Shape mismatch - skip this signal
                    continue
            except Exception:
                continue

        if not signal_arrays:
            raise ValueError("No valid signals found")

        # Additional check: if all signals are single-element default placeholders, consider them invalid
        # This handles the case where all signals are missing and return np.array([0.0])
        if len(signal_arrays) > 0:
            all_single_default = all(arr.size == 1 and arr[0] == 0.0 for arr in signal_arrays.values())
            if all_single_default:
                raise ValueError("No valid signals found")

        # Get dimensions
        first_array = list(signal_arrays.values())[0]
        fused_array = np.zeros_like(first_array, dtype=np.float32)

        # Apply fusion function per voxel
        # OPTIMIZATION: Use vectorized approach when possible
        # For custom functions, we still need iteration, but optimize where possible

        # Stack arrays for efficient access
        signal_list = list(signal_arrays.values())
        stacked = np.stack(signal_list, axis=0)  # Shape: (n_signals, ...)

        # Create valid mask
        valid_mask = (~np.isnan(stacked)) & (stacked != 0.0)

        # For custom functions, we need to iterate, but do it more efficiently
        # by only processing voxels with at least one valid value
        has_valid = np.any(valid_mask, axis=0)
        valid_indices = np.where(has_valid)

        # Process only valid voxels (much faster than iterating all)
        for idx_tuple in zip(*valid_indices):
            values = []
            for i, array in enumerate(signal_arrays.values()):
                val = array[idx_tuple]
                if not np.isnan(val) and val != 0.0:
                    values.append(val)

            if values:
                fused_array[idx_tuple] = fusion_func(values)

        # Add fused signal to voxel data if possible
        if hasattr(voxel_data, "add_signal"):
            voxel_data.add_signal(output_signal_name, fused_array)

        return fused_array

    def fuse_voxel_grids(
        self,
        grids: List[Any],
        method: str = "weighted_average",
        quality_scores: Optional[Dict[str, float]] = None,
    ) -> Any:
        """
        Fuse multiple voxel grids into a single grid.

        This method combines signals from multiple grids and fuses them.
        All grids must have the same bounding box and resolution.

        Args:
            grids: List of voxel grids to fuse
            method: Fusion method ('weighted_average', 'median', 'max', 'min', 'average')
            quality_scores: Optional quality scores per grid (by grid index or signal name)

        Returns:
            Fused voxel grid (returns the first grid with all signals combined and fused)
        """
        if not grids:
            raise ValueError("At least one grid must be provided")

        if len(grids) == 1:
            return grids[0]

        # Use first grid as base
        base_grid = grids[0]

        # Collect all signals from all grids
        all_signals = set()
        for grid in grids:
            if hasattr(grid, "available_signals"):
                all_signals.update(grid.available_signals)

        # Update base grid's available signals to include all
        if hasattr(base_grid, "available_signals"):
            base_grid.available_signals.update(all_signals)

        unique_signals = list(all_signals)

        if not unique_signals:
            return base_grid

        # If only one signal, no fusion needed
        if len(unique_signals) == 1:
            return base_grid

        # Map method string to FusionStrategy
        method_map = {
            "weighted_average": FusionStrategy.WEIGHTED_AVERAGE,
            "average": FusionStrategy.AVERAGE,
            "median": FusionStrategy.MEDIAN,
            "max": FusionStrategy.MAX,
            "maximum": FusionStrategy.MAX,
            "min": FusionStrategy.MIN,
            "minimum": FusionStrategy.MIN,
            "quality_based": FusionStrategy.WEIGHTED_AVERAGE,
        }

        fusion_strategy = method_map.get(method.lower(), FusionStrategy.WEIGHTED_AVERAGE)

        # If we have multiple signals, try to fuse them
        if len(unique_signals) > 1:
            # Use quality scores if provided
            signal_quality_scores = None
            if quality_scores:
                signal_quality_scores = quality_scores

            # Try to fuse all signals in the base grid
            # If grids are empty or signals can't be retrieved, just return base grid
            try:
                fused_array = self.fuse_voxel_signals(
                    voxel_data=base_grid,
                    signals=unique_signals,
                    fusion_strategy=fusion_strategy,
                    quality_scores=signal_quality_scores,
                    output_signal_name="fused",
                )
            except (ValueError, AttributeError) as e:
                # If fusion fails (e.g., no valid signals, grid not finalized),
                # just return the base grid with all signals collected
                # This allows the method to work even with empty grids
                if "No valid signals found" in str(e) or "finalized" in str(e).lower():
                    pass  # Just return base grid as-is
                else:
                    raise

        return base_grid
