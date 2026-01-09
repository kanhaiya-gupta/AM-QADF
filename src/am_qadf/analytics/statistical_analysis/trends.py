"""
Trend Analysis

Analyzes trends in voxel domain data:
- Temporal Trends: Trends over time/layers
- Spatial Trends: Trends across spatial dimensions
- Build Progression: Analyze build progression
- Quality Evolution: Track quality over build
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from scipy import stats


@dataclass
class TrendResults:
    """Trend analysis results."""

    temporal_trends: Dict[str, Dict[str, float]]  # Per-signal temporal trends
    spatial_trends: Dict[str, Dict[str, float]]  # Per-signal spatial trends
    build_progression: Optional[Dict[str, Any]] = None  # Build progression metrics
    quality_evolution: Optional[Dict[str, Any]] = None  # Quality evolution over build

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        result = {
            "temporal_trends": self.temporal_trends,
            "spatial_trends": self.spatial_trends,
        }
        if self.build_progression:
            result["build_progression"] = self.build_progression
        if self.quality_evolution:
            result["quality_evolution"] = self.quality_evolution
        return result


class TrendAnalyzer:
    """Analyzes trends in voxel domain data."""

    def __init__(self):
        """Initialize the trend analyzer."""
        pass

    def analyze_temporal_trend(self, signal_array: np.ndarray, signal_name: str = "unknown") -> Dict[str, float]:
        """
        Analyze temporal trend (along z-axis/layers).

        Args:
            signal_array: Signal array (z-axis represents time/layers)
            signal_name: Name of the signal

        Returns:
            Dictionary with trend metrics (slope, intercept, r_value, p_value, trend_direction)
        """
        if signal_array.shape[2] < 2:
            return {
                "slope": 0.0,
                "intercept": 0.0,
                "r_value": 0.0,
                "p_value": 1.0,
                "trend_direction": "none",
            }

        # Aggregate values per layer
        layer_means = []
        layer_indices = []

        for z in range(signal_array.shape[2]):
            layer_slice = signal_array[:, :, z]
            valid_values = layer_slice[(~np.isnan(layer_slice)) & (layer_slice != 0.0)]
            if len(valid_values) > 0:
                layer_means.append(np.mean(valid_values))
                layer_indices.append(z)

        if len(layer_means) < 2:
            return {
                "slope": 0.0,
                "intercept": 0.0,
                "r_value": 0.0,
                "p_value": 1.0,
                "trend_direction": "none",
            }

        # Linear regression
        x = np.array(layer_indices)
        y = np.array(layer_means)

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        # Determine trend direction
        if abs(slope) < 1e-6:
            trend_direction = "none"
        elif slope > 0:
            trend_direction = "increasing"
        else:
            trend_direction = "decreasing"

        return {
            "slope": slope,
            "intercept": intercept,
            "r_value": r_value,
            "p_value": p_value,
            "std_err": std_err,
            "trend_direction": trend_direction,
        }

    def analyze_spatial_trend(
        self,
        signal_array: np.ndarray,
        axis: int = 0,  # 0=x, 1=y, 2=z
        signal_name: str = "unknown",
    ) -> Dict[str, float]:
        """
        Analyze spatial trend along a specific axis.

        Args:
            signal_array: Signal array
            axis: Axis to analyze (0=x, 1=y, 2=z)
            signal_name: Name of the signal

        Returns:
            Dictionary with trend metrics
        """
        if axis >= len(signal_array.shape):
            return {
                "slope": 0.0,
                "intercept": 0.0,
                "r_value": 0.0,
                "p_value": 1.0,
                "trend_direction": "none",
            }

        # Aggregate values along the axis
        axis_means = []
        axis_indices = []

        for i in range(signal_array.shape[axis]):
            if axis == 0:
                slice_data = signal_array[i, :, :]
            elif axis == 1:
                slice_data = signal_array[:, i, :]
            else:  # axis == 2
                slice_data = signal_array[:, :, i]

            valid_values = slice_data[(~np.isnan(slice_data)) & (slice_data != 0.0)]
            if len(valid_values) > 0:
                axis_means.append(np.mean(valid_values))
                axis_indices.append(i)

        if len(axis_means) < 2:
            return {
                "slope": 0.0,
                "intercept": 0.0,
                "r_value": 0.0,
                "p_value": 1.0,
                "trend_direction": "none",
            }

        # Linear regression
        x = np.array(axis_indices)
        y = np.array(axis_means)

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        # Determine trend direction
        if abs(slope) < 1e-6:
            trend_direction = "none"
        elif slope > 0:
            trend_direction = "increasing"
        else:
            trend_direction = "decreasing"

        return {
            "slope": slope,
            "intercept": intercept,
            "r_value": r_value,
            "p_value": p_value,
            "std_err": std_err,
            "trend_direction": trend_direction,
            "axis": axis,
        }

    def analyze_build_progression(self, voxel_data: Any, signals: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze build progression over layers.

        Args:
            voxel_data: Voxel domain data object
            signals: List of signal names to analyze (None = all signals)

        Returns:
            Dictionary with build progression metrics
        """
        if signals is None:
            signals = list(voxel_data.available_signals) if hasattr(voxel_data, "available_signals") else []

        if not signals:
            return {}

        # Get layer-wise statistics
        layer_stats = {}

        for signal in signals:
            try:
                signal_array = voxel_data.get_signal_array(signal, default=0.0)
                n_layers = signal_array.shape[2]

                layer_means = []
                layer_stds = []
                layer_counts = []

                for z in range(n_layers):
                    layer_slice = signal_array[:, :, z]
                    valid_values = layer_slice[(~np.isnan(layer_slice)) & (layer_slice != 0.0)]

                    if len(valid_values) > 0:
                        layer_means.append(np.mean(valid_values))
                        layer_stds.append(np.std(valid_values))
                        layer_counts.append(len(valid_values))
                    else:
                        layer_means.append(0.0)
                        layer_stds.append(0.0)
                        layer_counts.append(0)

                layer_stats[signal] = {
                    "layer_means": layer_means,
                    "layer_stds": layer_stds,
                    "layer_counts": layer_counts,
                    "num_layers": n_layers,
                }
            except Exception:
                continue

        # Calculate progression metrics
        progression_metrics = {}
        for signal, signal_stats in layer_stats.items():
            means = np.array(signal_stats["layer_means"])
            if len(means) > 1:
                # Overall trend
                x = np.arange(len(means))
                slope, intercept, r_value, p_value, _ = stats.linregress(x, means)

                progression_metrics[signal] = {
                    "overall_slope": slope,
                    "overall_trend": ("increasing" if slope > 0 else ("decreasing" if slope < 0 else "stable")),
                    "r_value": r_value,
                    "p_value": p_value,
                    "mean_value": np.mean(means),
                    "std_value": np.std(means),
                    "variation_coefficient": np.std(means) / (np.mean(means) + 1e-10),
                    "layer_means": signal_stats["layer_means"],
                    "layer_stds": signal_stats["layer_stds"],
                    "num_layers": signal_stats["num_layers"],
                }

        return {
            "layer_statistics": layer_stats,
            "progression_metrics": progression_metrics,
        }

    def analyze_quality_evolution(
        self,
        quality_scores: Dict[str, List[float]],
        layer_indices: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze quality evolution over build.

        Args:
            quality_scores: Dictionary mapping quality metric names to lists of scores per layer
            layer_indices: Optional layer indices (if None, uses 0, 1, 2, ...)

        Returns:
            Dictionary with quality evolution metrics
        """
        if not quality_scores:
            return {}

        if layer_indices is None:
            max_layers = max(len(scores) for scores in quality_scores.values())
            layer_indices = list(range(max_layers))

        evolution_metrics = {}

        for metric_name, scores in quality_scores.items():
            if len(scores) < 2:
                continue

            scores_array = np.array(scores)
            x = np.array(layer_indices[: len(scores)])

            # Linear regression
            slope, intercept, r_value, p_value, _ = stats.linregress(x, scores_array)

            evolution_metrics[metric_name] = {
                "slope": slope,
                "intercept": intercept,
                "r_value": r_value,
                "p_value": p_value,
                "trend": ("improving" if slope > 0 else ("degrading" if slope < 0 else "stable")),
                "mean_quality": np.mean(scores_array),
                "min_quality": np.min(scores_array),
                "max_quality": np.max(scores_array),
                "quality_range": np.max(scores_array) - np.min(scores_array),
            }

        return evolution_metrics

    def analyze_trends(
        self,
        voxel_data: Any,
        signals: Optional[List[str]] = None,
        include_spatial: bool = True,
    ) -> TrendResults:
        """
        Perform comprehensive trend analysis.

        Args:
            voxel_data: Voxel domain data object
            signals: List of signal names to analyze (None = all signals)
            include_spatial: Whether to analyze spatial trends

        Returns:
            TrendResults object
        """
        if signals is None:
            signals = list(voxel_data.available_signals) if hasattr(voxel_data, "available_signals") else []

        temporal_trends = {}
        spatial_trends = {}

        for signal in signals:
            try:
                signal_array = voxel_data.get_signal_array(signal, default=0.0)

                # Temporal trend
                temporal_trends[signal] = self.analyze_temporal_trend(signal_array, signal)

                # Spatial trends
                if include_spatial:
                    spatial_trends[signal] = {}
                    for axis in [0, 1]:  # X and Y axes
                        axis_name = ["x", "y"][axis]
                        spatial_trends[signal][axis_name] = self.analyze_spatial_trend(
                            signal_array, axis=axis, signal_name=signal
                        )
            except Exception:
                continue

        # Build progression
        build_progression = self.analyze_build_progression(voxel_data, signals)

        return TrendResults(
            temporal_trends=temporal_trends,
            spatial_trends=spatial_trends,
            build_progression=build_progression,
            quality_evolution=None,  # Would need quality scores as input
        )
