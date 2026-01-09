"""
Fusion Quality Assessment

Assesses the quality of fused voxel signals:
- Fusion Accuracy: Compare fused signal with individual signals
- Signal Consistency: Check consistency between fused and source signals
- Fusion Completeness: Assess coverage of fused signal
- Quality Metrics: Overall fusion quality metrics
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class FusionQualityMetrics:
    """Fusion quality metrics."""

    fusion_accuracy: float  # Accuracy of fusion (0-1, higher is better)
    signal_consistency: float  # Consistency with source signals (0-1)
    fusion_completeness: float  # Coverage of fused signal (0-1)
    quality_score: float  # Overall quality score (0-1)

    # Detailed metrics
    per_signal_accuracy: Dict[str, float]  # Accuracy per source signal
    coverage_ratio: float  # Ratio of voxels with fused data
    residual_errors: Optional[np.ndarray] = None  # Residual errors per voxel

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        result = {
            "fusion_accuracy": self.fusion_accuracy,
            "signal_consistency": self.signal_consistency,
            "fusion_completeness": self.fusion_completeness,
            "quality_score": self.quality_score,
            "per_signal_accuracy": self.per_signal_accuracy,
            "coverage_ratio": self.coverage_ratio,
        }
        if self.residual_errors is not None:
            result["residual_errors_shape"] = self.residual_errors.shape
        return result


class FusionQualityAssessor:
    """Assesses quality of fused voxel signals."""

    def __init__(self):
        """Initialize the fusion quality assessor."""
        pass

    def assess_fusion_quality(
        self,
        fused_array: np.ndarray,
        source_arrays: Dict[str, np.ndarray],
        fusion_weights: Optional[Dict[str, float]] = None,
    ) -> FusionQualityMetrics:
        """
        Assess quality of fused signal.

        Args:
            fused_array: Fused signal array
            source_arrays: Dictionary mapping signal names to source arrays
            fusion_weights: Optional weights used for fusion

        Returns:
            FusionQualityMetrics object
        """
        # Calculate coverage
        valid_mask = (~np.isnan(fused_array)) & (fused_array != 0.0)
        coverage_ratio = np.sum(valid_mask) / fused_array.size if fused_array.size > 0 else 0.0

        # Calculate per-signal accuracy
        per_signal_accuracy = {}
        for signal_name, source_array in source_arrays.items():
            # Check if arrays have compatible shapes
            if fused_array.shape != source_array.shape:
                per_signal_accuracy[signal_name] = 0.0
                continue

            # Compare fused with source where both are valid
            both_valid = valid_mask & (~np.isnan(source_array)) & (source_array != 0.0)

            if np.sum(both_valid) > 0:
                fused_vals = fused_array[both_valid]
                source_vals = source_array[both_valid]

                # Calculate correlation as accuracy measure
                if np.std(fused_vals) > 0 and np.std(source_vals) > 0:
                    corr = np.corrcoef(fused_vals, source_vals)[0, 1]
                    per_signal_accuracy[signal_name] = abs(corr) if not np.isnan(corr) else 0.0
                else:
                    per_signal_accuracy[signal_name] = 0.0
            else:
                per_signal_accuracy[signal_name] = 0.0

        # Overall fusion accuracy (mean of per-signal accuracies)
        fusion_accuracy = np.mean(list(per_signal_accuracy.values())) if per_signal_accuracy else 0.0

        # Calculate signal consistency
        # Consistency is how well fused signal represents all sources
        consistency_scores = []
        for signal_name, source_array in source_arrays.items():
            # Check if arrays have compatible shapes
            if fused_array.shape != source_array.shape:
                continue

            both_valid = valid_mask & (~np.isnan(source_array)) & (source_array != 0.0)

            if np.sum(both_valid) > 0:
                fused_vals = fused_array[both_valid]
                source_vals = source_array[both_valid]

                # Normalize for comparison
                if np.max(fused_vals) > 0 and np.max(source_vals) > 0:
                    fused_norm = fused_vals / np.max(fused_vals)
                    source_norm = source_vals / np.max(source_vals)

                    # Calculate consistency (1 - normalized RMSE)
                    rmse = np.sqrt(np.mean((fused_norm - source_norm) ** 2))
                    consistency = max(0.0, 1.0 - rmse)
                    consistency_scores.append(consistency)

        signal_consistency = np.mean(consistency_scores) if consistency_scores else 0.0

        # Calculate residual errors
        residual_errors = None
        if source_arrays:
            # Calculate weighted residual
            residual_errors = np.zeros_like(fused_array)
            total_weight = 0.0

            for signal_name, source_array in source_arrays.items():
                # Check if arrays have compatible shapes
                if fused_array.shape != source_array.shape:
                    continue

                weight = fusion_weights.get(signal_name, 1.0) if fusion_weights else 1.0
                both_valid = valid_mask & (~np.isnan(source_array)) & (source_array != 0.0)

                if np.sum(both_valid) > 0:
                    residual_errors[both_valid] += weight * np.abs(fused_array[both_valid] - source_array[both_valid])
                    total_weight += weight

            if total_weight > 0:
                residual_errors[valid_mask] /= total_weight
            residual_errors[~valid_mask] = np.nan

        # Overall quality score (weighted combination)
        quality_score = 0.4 * fusion_accuracy + 0.3 * signal_consistency + 0.3 * coverage_ratio

        return FusionQualityMetrics(
            fusion_accuracy=fusion_accuracy,
            signal_consistency=signal_consistency,
            fusion_completeness=coverage_ratio,
            quality_score=quality_score,
            per_signal_accuracy=per_signal_accuracy,
            coverage_ratio=coverage_ratio,
            residual_errors=residual_errors,
        )

    def compare_fusion_strategies(
        self,
        voxel_data: Any,
        signals: List[str],
        strategies: List[str],
        quality_scores: Optional[Dict[str, float]] = None,
    ) -> Dict[str, FusionQualityMetrics]:
        """
        Compare different fusion strategies.

        Args:
            voxel_data: Voxel domain data object
            signals: List of signal names to fuse
            strategies: List of fusion strategy names to compare
            quality_scores: Optional quality scores per signal

        Returns:
            Dictionary mapping strategy names to FusionQualityMetrics
        """
        from .voxel_fusion import VoxelFusion, FusionStrategy

        results = {}

        # Get source arrays
        source_arrays = {}
        for signal in signals:
            try:
                array = voxel_data.get_signal_array(signal, default=0.0)
                # Check if array is actually valid (not just default empty array)
                # An array is considered valid if it has non-zero size and contains non-default values
                if array.size > 0:
                    # Check if it's not just a single default value
                    if array.size > 1 or (array.size == 1 and array[0] != 0.0):
                        source_arrays[signal] = array
            except Exception:
                continue

        if not source_arrays:
            return results

        # Try each strategy
        strategy_map = {
            "weighted_average": FusionStrategy.WEIGHTED_AVERAGE,
            "average": FusionStrategy.AVERAGE,
            "median": FusionStrategy.MEDIAN,
            "max": FusionStrategy.MAX,
            "min": FusionStrategy.MIN,
        }

        fusion_engine = VoxelFusion(use_quality_scores=quality_scores is not None)

        for strategy_name in strategies:
            if strategy_name not in strategy_map:
                continue

            try:
                # Fuse with this strategy
                fused_array = fusion_engine.fuse_voxel_signals(
                    voxel_data,
                    signals,
                    fusion_strategy=strategy_map[strategy_name],
                    quality_scores=quality_scores,
                )

                # Assess quality
                metrics = self.assess_fusion_quality(fused_array, source_arrays, fusion_weights=quality_scores)

                results[strategy_name] = metrics
            except Exception as e:
                print(f"⚠️ Error evaluating strategy {strategy_name}: {e}")
                continue

        return results
