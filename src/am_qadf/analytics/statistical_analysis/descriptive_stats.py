"""
Descriptive Statistics

Calculates descriptive statistics for voxel domain signals:
- Mean, median, std, min, max per voxel
- Distribution analysis (histograms, percentiles)
- Statistical summaries across signals
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from scipy import stats


@dataclass
class DescriptiveStatistics:
    """Descriptive statistics for a signal."""

    signal_name: str
    mean: float
    median: float
    std: float
    min: float
    max: float
    q25: float  # 25th percentile
    q75: float  # 75th percentile
    q95: float  # 95th percentile
    q99: float  # 99th percentile
    skewness: float
    kurtosis: float
    valid_count: int  # Number of valid (non-NaN, non-zero) values
    total_count: int  # Total number of voxels

    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary."""
        return {
            "signal_name": self.signal_name,
            "mean": self.mean,
            "median": self.median,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "q25": self.q25,
            "q75": self.q75,
            "q95": self.q95,
            "q99": self.q99,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "valid_count": self.valid_count,
            "total_count": self.total_count,
            "valid_ratio": (self.valid_count / self.total_count if self.total_count > 0 else 0.0),
        }


class DescriptiveStatsAnalyzer:
    """Analyzes descriptive statistics for voxel domain signals."""

    def __init__(self):
        """Initialize the descriptive statistics analyzer."""
        pass

    def calculate_statistics(self, signal_array: np.ndarray, signal_name: str = "unknown") -> DescriptiveStatistics:
        """
        Calculate descriptive statistics for a signal array.

        Args:
            signal_array: Signal array
            signal_name: Name of the signal

        Returns:
            DescriptiveStatistics object
        """
        # Flatten array and remove invalid values
        flat_array = signal_array.flatten()
        valid_mask = (~np.isnan(flat_array)) & (flat_array != 0.0)
        valid_values = flat_array[valid_mask]

        total_count = len(flat_array)
        valid_count = len(valid_values)

        if valid_count == 0:
            # Return zero statistics
            return DescriptiveStatistics(
                signal_name=signal_name,
                mean=0.0,
                median=0.0,
                std=0.0,
                min=0.0,
                max=0.0,
                q25=0.0,
                q75=0.0,
                q95=0.0,
                q99=0.0,
                skewness=0.0,
                kurtosis=0.0,
                valid_count=0,
                total_count=total_count,
            )

        # Calculate basic statistics
        mean = np.mean(valid_values)
        median = np.median(valid_values)
        std = np.std(valid_values)
        min_val = np.min(valid_values)
        max_val = np.max(valid_values)

        # Calculate percentiles
        q25 = np.percentile(valid_values, 25)
        q75 = np.percentile(valid_values, 75)
        q95 = np.percentile(valid_values, 95)
        q99 = np.percentile(valid_values, 99)

        # Calculate skewness and kurtosis
        if valid_count > 2 and std > 0:
            skewness = stats.skew(valid_values)
            kurtosis = stats.kurtosis(valid_values)
        else:
            skewness = 0.0
            kurtosis = 0.0

        return DescriptiveStatistics(
            signal_name=signal_name,
            mean=mean,
            median=median,
            std=std,
            min=min_val,
            max=max_val,
            q25=q25,
            q75=q75,
            q95=q95,
            q99=q99,
            skewness=skewness,
            kurtosis=kurtosis,
            valid_count=valid_count,
            total_count=total_count,
        )

    def calculate_per_voxel_statistics(self, signal_arrays: Dict[str, np.ndarray], statistic: str = "mean") -> np.ndarray:
        """
        Calculate per-voxel statistics across multiple signals.

        Args:
            signal_arrays: Dictionary mapping signal names to arrays
            statistic: Statistic to calculate ('mean', 'median', 'std', 'min', 'max')

        Returns:
            Array of per-voxel statistics
        """
        if not signal_arrays:
            return np.array([])

        # Stack arrays
        arrays = list(signal_arrays.values())
        stacked = np.stack(arrays, axis=-1)  # Shape: (X, Y, Z, N_signals)

        # Mask invalid values
        valid_mask = (~np.isnan(stacked)) & (stacked != 0.0)

        # Calculate statistic per voxel
        if statistic == "mean":
            result = np.nanmean(stacked, axis=-1)
        elif statistic == "median":
            result = np.nanmedian(stacked, axis=-1)
        elif statistic == "std":
            result = np.nanstd(stacked, axis=-1)
        elif statistic == "min":
            result = np.nanmin(stacked, axis=-1)
        elif statistic == "max":
            result = np.nanmax(stacked, axis=-1)
        else:
            raise ValueError(f"Unknown statistic: {statistic}")

        # Set invalid voxels to NaN
        result[~np.any(valid_mask, axis=-1)] = np.nan

        return result

    def analyze_distribution(self, signal_array: np.ndarray, bins: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Analyze signal distribution.

        Args:
            signal_array: Signal array
            bins: Number of bins for histogram

        Returns:
            (histogram_counts, bin_edges)
        """
        flat_array = signal_array.flatten()
        valid_values = flat_array[(~np.isnan(flat_array)) & (flat_array != 0.0)]

        if len(valid_values) == 0:
            return np.array([]), np.array([])

        counts, edges = np.histogram(valid_values, bins=bins)
        return counts, edges

    def compare_distributions(
        self, signal_arrays: Dict[str, np.ndarray], bins: int = 50
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Compare distributions of multiple signals.

        Args:
            signal_arrays: Dictionary mapping signal names to arrays
            bins: Number of bins for histogram

        Returns:
            Dictionary mapping signal names to (counts, edges)
        """
        distributions = {}
        for signal_name, signal_array in signal_arrays.items():
            counts, edges = self.analyze_distribution(signal_array, bins=bins)
            distributions[signal_name] = (counts, edges)
        return distributions

    def assess_all_signals(self, voxel_data: Any, signals: Optional[List[str]] = None) -> Dict[str, DescriptiveStatistics]:
        """
        Calculate descriptive statistics for all signals.

        Args:
            voxel_data: Voxel domain data object
            signals: List of signal names to analyze (None = all signals)

        Returns:
            Dictionary mapping signal names to DescriptiveStatistics
        """
        if signals is None:
            signals = list(voxel_data.available_signals) if hasattr(voxel_data, "available_signals") else []

        results = {}
        for signal in signals:
            try:
                signal_array = voxel_data.get_signal_array(signal, default=0.0)
                results[signal] = self.calculate_statistics(signal_array, signal)
            except Exception as e:
                print(f"⚠️ Error analyzing signal {signal}: {e}")
                continue

        return results
