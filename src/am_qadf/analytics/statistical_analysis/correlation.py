"""
Correlation Analysis

Analyzes correlations in voxel domain data:
- Signal Correlations: Correlations between signals
- Spatial Correlations: Spatial correlation patterns
- Temporal Correlations: Temporal correlation patterns
- Cross-Model Correlations: Compare across models
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class CorrelationResults:
    """Correlation analysis results."""

    signal_correlations: Dict[Tuple[str, str], float]  # Pairwise signal correlations
    correlation_matrix: Optional[np.ndarray] = None  # Full correlation matrix
    signal_names: Optional[List[str]] = None  # Signal names for matrix

    spatial_correlations: Optional[Dict[str, float]] = None  # Spatial autocorrelation per signal
    temporal_correlations: Optional[Dict[str, float]] = None  # Temporal autocorrelation per signal

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        result = {
            "signal_correlations": {f"{s1}_{s2}": corr for (s1, s2), corr in self.signal_correlations.items()},
            "num_correlations": len(self.signal_correlations),
        }
        if self.correlation_matrix is not None:
            result["correlation_matrix_shape"] = self.correlation_matrix.shape
        if self.signal_names:
            result["signal_names"] = self.signal_names
        if self.spatial_correlations:
            result["spatial_correlations"] = self.spatial_correlations
        if self.temporal_correlations:
            result["temporal_correlations"] = self.temporal_correlations
        return result


class CorrelationAnalyzer:
    """Analyzes correlations in voxel domain data."""

    def __init__(self):
        """Initialize the correlation analyzer."""
        pass

    def calculate_signal_correlations(
        self, signal_arrays: Dict[str, np.ndarray], min_samples: int = 2
    ) -> Dict[Tuple[str, str], float]:
        """
        Calculate pairwise correlations between signals.

        Args:
            signal_arrays: Dictionary mapping signal names to arrays
            min_samples: Minimum number of valid samples required

        Returns:
            Dictionary mapping (signal1, signal2) tuples to correlation coefficients
        """
        correlations = {}
        signal_names = list(signal_arrays.keys())

        for i, signal1 in enumerate(signal_names):
            for j, signal2 in enumerate(signal_names):
                if i >= j:
                    continue  # Only calculate upper triangle

                arr1 = signal_arrays[signal1].flatten()
                arr2 = signal_arrays[signal2].flatten()

                # Ensure arrays have same length
                min_len = min(len(arr1), len(arr2))
                arr1 = arr1[:min_len]
                arr2 = arr2[:min_len]

                # Find valid pairs (exclude NaN, but allow zeros)
                valid_mask = (~np.isnan(arr1)) & (~np.isnan(arr2))

                arr1_clean = arr1[valid_mask]
                arr2_clean = arr2[valid_mask]

                # Check if we have enough valid samples
                if len(arr1_clean) < min_samples:
                    continue

                # Calculate Pearson correlation
                # Allow correlation even if one signal has zero variance (will be 0 or NaN)
                if len(arr1_clean) >= 2:
                    if np.std(arr1_clean) > 1e-10 and np.std(arr2_clean) > 1e-10:
                        corr = np.corrcoef(arr1_clean, arr2_clean)[0, 1]
                        if not np.isnan(corr):
                            correlations[(signal1, signal2)] = corr
                    elif np.std(arr1_clean) <= 1e-10 and np.std(arr2_clean) <= 1e-10:
                        # Both constant - perfect correlation (though not meaningful)
                        correlations[(signal1, signal2)] = 1.0
                    else:
                        # One constant, one variable - no correlation
                        correlations[(signal1, signal2)] = 0.0

        return correlations

    def calculate_correlation_matrix(
        self, signal_arrays: Dict[str, np.ndarray], min_samples: int = 10
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Calculate full correlation matrix for all signals.

        Args:
            signal_arrays: Dictionary mapping signal names to arrays
            min_samples: Minimum number of valid samples required

        Returns:
            (correlation_matrix, signal_names)
        """
        signal_names = list(signal_arrays.keys())
        n_signals = len(signal_names)

        if n_signals == 0:
            return np.array([]), []

        correlation_matrix = np.eye(n_signals)  # Identity matrix (diagonal = 1.0)

        for i, signal1 in enumerate(signal_names):
            for j, signal2 in enumerate(signal_names):
                if i == j:
                    continue

                arr1 = signal_arrays[signal1].flatten()
                arr2 = signal_arrays[signal2].flatten()

                # Find valid pairs
                valid_mask = (~np.isnan(arr1)) & (~np.isnan(arr2)) & (arr1 != 0.0) & (arr2 != 0.0)

                if np.sum(valid_mask) < min_samples:
                    correlation_matrix[i, j] = 0.0
                    continue

                arr1_clean = arr1[valid_mask]
                arr2_clean = arr2[valid_mask]

                # Calculate Pearson correlation
                if np.std(arr1_clean) > 0 and np.std(arr2_clean) > 0:
                    corr = np.corrcoef(arr1_clean, arr2_clean)[0, 1]
                    correlation_matrix[i, j] = corr if not np.isnan(corr) else 0.0
                else:
                    correlation_matrix[i, j] = 0.0

        # Make symmetric
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2

        return correlation_matrix, signal_names

    def calculate_spatial_autocorrelation(self, signal_array: np.ndarray, lag: int = 1) -> float:
        """
        Calculate spatial autocorrelation (Moran's I approximation).

        Args:
            signal_array: Signal array
            lag: Spatial lag (distance in voxels)

        Returns:
            Spatial autocorrelation coefficient
        """
        # Flatten and get valid values
        flat_array = signal_array.flatten()
        valid_mask = (~np.isnan(flat_array)) & (flat_array != 0.0)
        valid_values = flat_array[valid_mask]

        if len(valid_values) < 10:
            return 0.0

        # Calculate autocorrelation using shifted arrays
        # For 3D, we'll use a simplified approach: correlate with shifted version
        mean_val = np.mean(valid_values)
        centered = valid_values - mean_val

        # Create shifted version (simplified: use adjacent voxels)
        # This is a simplified spatial autocorrelation
        if len(centered) < 2:
            return 0.0

        # Calculate autocorrelation
        autocorr = np.corrcoef(centered[:-1], centered[1:])[0, 1]

        return autocorr if not np.isnan(autocorr) else 0.0

    def calculate_temporal_autocorrelation(
        self, signal_array: np.ndarray, signal_name: str = "unknown", lag: int = 1
    ) -> float:
        """
        Calculate temporal autocorrelation (along z-axis/layers).

        Args:
            signal_array: Signal array (z-axis represents time/layers)
            signal_name: Name of the signal (for logging)
            lag: Temporal lag (layers)

        Returns:
            Temporal autocorrelation coefficient
        """
        if len(signal_array.shape) < 3 or signal_array.shape[2] < lag + 1:
            return 0.0

        # Extract values from different layers
        layer_values = []
        for z in range(signal_array.shape[2]):
            layer_slice = signal_array[:, :, z]
            valid_values = layer_slice[(~np.isnan(layer_slice)) & (layer_slice != 0.0)]
            if len(valid_values) > 0:
                layer_values.append(np.mean(valid_values))

        if len(layer_values) < lag + 1:
            return 0.0

        # Calculate autocorrelation
        values = np.array(layer_values)
        mean_val = np.mean(values)
        centered = values - mean_val

        if len(centered) < lag + 1:
            return 0.0

        # Calculate autocorrelation at lag
        autocorr = np.corrcoef(centered[:-lag], centered[lag:])[0, 1]

        return autocorr if not np.isnan(autocorr) else 0.0

    def analyze_correlations(
        self,
        voxel_data: Any,
        signals: Optional[List[str]] = None,
        include_spatial: bool = True,
        include_temporal: bool = True,
    ) -> CorrelationResults:
        """
        Perform comprehensive correlation analysis.

        Args:
            voxel_data: Voxel domain data object
            signals: List of signal names to analyze (None = all signals)
            include_spatial: Whether to calculate spatial autocorrelations
            include_temporal: Whether to calculate temporal autocorrelations

        Returns:
            CorrelationResults object
        """
        if signals is None:
            signals = list(voxel_data.available_signals) if hasattr(voxel_data, "available_signals") else []

        if len(signals) < 2:
            return CorrelationResults(signal_correlations={}, correlation_matrix=None, signal_names=signals)

        # Get signal arrays
        signal_arrays = {}
        for signal in signals:
            try:
                signal_arrays[signal] = voxel_data.get_signal_array(signal, default=0.0)
            except Exception:
                continue

        if len(signal_arrays) < 2:
            return CorrelationResults(
                signal_correlations={},
                correlation_matrix=None,
                signal_names=list(signal_arrays.keys()),
            )

        # Calculate signal correlations
        signal_correlations = self.calculate_signal_correlations(signal_arrays)

        # Calculate correlation matrix
        correlation_matrix, signal_names = self.calculate_correlation_matrix(signal_arrays)

        # Calculate spatial autocorrelations
        spatial_correlations = None
        if include_spatial:
            spatial_correlations = {}
            for signal in signals:
                try:
                    signal_array = signal_arrays[signal]
                    spatial_corr = self.calculate_spatial_autocorrelation(signal_array)
                    spatial_correlations[signal] = spatial_corr
                except Exception:
                    continue

        # Calculate temporal autocorrelations
        temporal_correlations = None
        if include_temporal:
            temporal_correlations = {}
            for signal in signals:
                try:
                    signal_array = signal_arrays[signal]
                    temporal_corr = self.calculate_temporal_autocorrelation(signal_array)
                    temporal_correlations[signal] = temporal_corr
                except Exception:
                    continue

        return CorrelationResults(
            signal_correlations=signal_correlations,
            correlation_matrix=correlation_matrix,
            signal_names=signal_names,
            spatial_correlations=spatial_correlations,
            temporal_correlations=temporal_correlations,
        )
