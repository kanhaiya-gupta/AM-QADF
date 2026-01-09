"""
Pattern Recognition

Identifies patterns in voxel domain data:
- Spatial Patterns: Identify spatial patterns
- Temporal Patterns: Identify temporal patterns
- Anomaly Patterns: Pattern-based anomaly detection
- Process Patterns: Identify process patterns
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from scipy import ndimage, signal
from scipy.cluster import hierarchy
from sklearn.cluster import DBSCAN


@dataclass
class PatternResults:
    """Pattern recognition results."""

    spatial_patterns: Dict[str, Any]  # Identified spatial patterns
    temporal_patterns: Dict[str, Any]  # Identified temporal patterns
    anomaly_patterns: Optional[Dict[str, Any]] = None  # Anomaly patterns
    process_patterns: Optional[Dict[str, Any]] = None  # Process patterns

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        result = {
            "spatial_patterns": self.spatial_patterns,
            "temporal_patterns": self.temporal_patterns,
        }
        if self.anomaly_patterns:
            result["anomaly_patterns"] = self.anomaly_patterns
        if self.process_patterns:
            result["process_patterns"] = self.process_patterns
        return result


class PatternAnalyzer:
    """Identifies patterns in voxel domain data."""

    def __init__(self):
        """Initialize the pattern analyzer."""
        pass

    def detect_spatial_clusters(
        self,
        signal_array: np.ndarray,
        threshold: Optional[float] = None,
        min_cluster_size: int = 10,
    ) -> Dict[str, Any]:
        """
        Detect spatial clusters in signal data.

        Args:
            signal_array: Signal array
            threshold: Threshold for cluster detection (None = use median)
            min_cluster_size: Minimum cluster size in voxels

        Returns:
            Dictionary with cluster information
        """
        # Flatten and get valid values
        flat_array = signal_array.flatten()
        valid_mask = (~np.isnan(flat_array)) & (flat_array != 0.0)
        valid_values = flat_array[valid_mask]

        if len(valid_values) < min_cluster_size:
            return {"num_clusters": 0, "clusters": [], "cluster_sizes": []}

        # Use threshold or median
        if threshold is None:
            threshold = np.median(valid_values)

        # Create binary mask
        binary_mask = (signal_array > threshold) & valid_mask.reshape(signal_array.shape)

        # Label connected components
        labeled, num_features = ndimage.label(binary_mask)

        clusters = []
        cluster_sizes = []

        for label_id in range(1, num_features + 1):
            cluster_mask = labeled == label_id
            cluster_size = np.sum(cluster_mask)

            if cluster_size >= min_cluster_size:
                # Get cluster properties
                cluster_coords = np.where(cluster_mask)
                cluster_values = signal_array[cluster_mask]

                clusters.append(
                    {
                        "label": label_id,
                        "size": cluster_size,
                        "mean_value": np.mean(cluster_values),
                        "max_value": np.max(cluster_values),
                        "min_value": np.min(cluster_values),
                        "centroid": tuple(np.mean(coords) for coords in cluster_coords),
                    }
                )
                cluster_sizes.append(cluster_size)

        return {
            "num_clusters": len(clusters),
            "clusters": clusters,
            "cluster_sizes": cluster_sizes,
            "threshold": threshold,
        }

    def detect_periodic_patterns(
        self, signal_array: np.ndarray, axis: int = 2  # Usually z-axis for temporal
    ) -> Dict[str, Any]:
        """
        Detect periodic patterns along an axis.

        Args:
            signal_array: Signal array
            axis: Axis to analyze (usually 2 for temporal/layers)

        Returns:
            Dictionary with periodic pattern information
        """
        if axis >= len(signal_array.shape):
            return {"has_periodicity": False, "period": None, "frequency": None}

        # Aggregate along axis
        axis_means = []
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
            else:
                axis_means.append(0.0)

        if len(axis_means) < 4:
            return {"has_periodicity": False, "period": None, "frequency": None}

        # FFT to detect periodicity
        fft_values = np.fft.fft(axis_means)
        fft_freq = np.fft.fftfreq(len(axis_means))

        # Find dominant frequency (excluding DC component)
        power = np.abs(fft_values[1 : len(fft_values) // 2]) ** 2
        dominant_freq_idx = np.argmax(power) + 1

        if power[dominant_freq_idx - 1] > np.mean(power) * 2:  # Significant peak
            period = len(axis_means) / dominant_freq_idx if dominant_freq_idx > 0 else None
            return {
                "has_periodicity": True,
                "period": period,
                "frequency": dominant_freq_idx / len(axis_means),
                "power": float(power[dominant_freq_idx - 1]),
            }
        else:
            return {"has_periodicity": False, "period": None, "frequency": None}

    def detect_anomaly_patterns(
        self,
        signal_array: np.ndarray,
        method: str = "statistical",
        threshold_std: float = 3.0,
    ) -> Dict[str, Any]:
        """
        Detect anomaly patterns in signal data.

        Args:
            signal_array: Signal array
            method: Detection method ('statistical', 'isolation', 'clustering')
            threshold_std: Standard deviation threshold for statistical method

        Returns:
            Dictionary with anomaly pattern information
        """
        # Flatten and get valid values
        flat_array = signal_array.flatten()
        valid_mask = (~np.isnan(flat_array)) & (flat_array != 0.0)
        valid_values = flat_array[valid_mask]

        if len(valid_values) < 10:
            return {"num_anomalies": 0, "anomaly_indices": [], "anomaly_fraction": 0.0}

        if method == "statistical":
            # Statistical outlier detection (Z-score)
            mean_val = np.mean(valid_values)
            std_val = np.std(valid_values)

            if std_val == 0:
                return {
                    "num_anomalies": 0,
                    "anomaly_indices": [],
                    "anomaly_fraction": 0.0,
                }

            z_scores = np.abs((valid_values - mean_val) / std_val)
            anomaly_mask = z_scores > threshold_std

            num_anomalies = np.sum(anomaly_mask)
            anomaly_fraction = num_anomalies / len(valid_values)

            # Get original indices
            valid_indices = np.where(valid_mask)[0]
            anomaly_indices = valid_indices[anomaly_mask]

            return {
                "num_anomalies": num_anomalies,
                "anomaly_indices": anomaly_indices.tolist(),
                "anomaly_fraction": anomaly_fraction,
                "threshold": mean_val + threshold_std * std_val,
                "method": method,
            }

        elif method == "isolation":
            # Isolation Forest (simplified: use statistical for now)
            return self.detect_anomaly_patterns(signal_array, method="statistical", threshold_std=threshold_std)

        else:  # clustering
            # DBSCAN clustering to find outliers
            try:
                # Reshape to 2D for clustering
                coords = np.where(valid_mask.reshape(signal_array.shape))
                points = np.column_stack(coords)
                values = valid_values.reshape(-1, 1)

                # Combine coordinates and values
                features = np.column_stack([points, values])

                # DBSCAN
                clustering = DBSCAN(eps=2.0, min_samples=5).fit(features)
                labels = clustering.labels_

                # Outliers are labeled as -1
                outlier_mask = labels == -1
                num_anomalies = np.sum(outlier_mask)

                return {
                    "num_anomalies": num_anomalies,
                    "anomaly_indices": np.where(outlier_mask)[0].tolist(),
                    "anomaly_fraction": num_anomalies / len(valid_values),
                    "method": method,
                }
            except Exception:
                # Fallback to statistical
                return self.detect_anomaly_patterns(signal_array, method="statistical", threshold_std=threshold_std)

    def identify_process_patterns(self, voxel_data: Any, signals: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Identify process patterns across signals.

        Args:
            voxel_data: Voxel domain data object
            signals: List of signal names to analyze (None = all signals)

        Returns:
            Dictionary with process pattern information
        """
        if signals is None:
            signals = list(voxel_data.available_signals) if hasattr(voxel_data, "available_signals") else []

        if len(signals) < 2:
            return {}

        process_patterns = {}

        # Analyze correlations between signals (process relationships)
        signal_arrays = {}
        for signal in signals:
            try:
                signal_arrays[signal] = voxel_data.get_signal_array(signal, default=0.0)
            except Exception:
                continue

        if len(signal_arrays) < 2:
            return {}

        # Calculate pairwise correlations
        correlations = {}
        signal_names = list(signal_arrays.keys())

        for i, signal1 in enumerate(signal_names):
            for j, signal2 in enumerate(signal_names):
                if i >= j:
                    continue

                arr1 = signal_arrays[signal1].flatten()
                arr2 = signal_arrays[signal2].flatten()

                valid_mask = (~np.isnan(arr1)) & (~np.isnan(arr2)) & (arr1 != 0.0) & (arr2 != 0.0)

                if np.sum(valid_mask) < 10:
                    continue

                arr1_clean = arr1[valid_mask]
                arr2_clean = arr2[valid_mask]

                if np.std(arr1_clean) > 0 and np.std(arr2_clean) > 0:
                    corr = np.corrcoef(arr1_clean, arr2_clean)[0, 1]
                    if not np.isnan(corr):
                        correlations[(signal1, signal2)] = corr

        # Identify strong correlations (process relationships)
        strong_correlations = {pair: corr for pair, corr in correlations.items() if abs(corr) > 0.7}

        process_patterns["signal_correlations"] = correlations
        process_patterns["strong_relationships"] = strong_correlations

        return process_patterns

    def detect_process_patterns(self, voxel_data: Any, signals: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Detect process patterns in voxel data.

        Args:
            voxel_data: Voxel domain data object
            signals: List of signal names to analyze (None = all signals)

        Returns:
            Dictionary with process pattern information
        """
        result = self.analyze_patterns(voxel_data, signals, include_anomalies=False, include_process=True)

        # Extract process patterns from result
        process_patterns = {}
        if result.process_patterns:
            process_patterns = result.process_patterns
        else:
            # If no process patterns found, return empty dict with structure
            process_patterns = {}

        return process_patterns

    def analyze_patterns(
        self,
        voxel_data: Any,
        signals: Optional[List[str]] = None,
        include_anomalies: bool = True,
        include_process: bool = True,
    ) -> PatternResults:
        """
        Perform comprehensive pattern analysis.

        Args:
            voxel_data: Voxel domain data object
            signals: List of signal names to analyze (None = all signals)
            include_anomalies: Whether to detect anomaly patterns
            include_process: Whether to identify process patterns

        Returns:
            PatternResults object
        """
        if signals is None:
            signals = list(voxel_data.available_signals) if hasattr(voxel_data, "available_signals") else []

        spatial_patterns = {}
        temporal_patterns = {}
        anomaly_patterns = {} if include_anomalies else None
        process_patterns = None

        for signal in signals:
            try:
                signal_array = voxel_data.get_signal_array(signal, default=0.0)

                # Spatial patterns (clusters)
                spatial_patterns[signal] = self.detect_spatial_clusters(signal_array)

                # Temporal patterns (periodicity)
                temporal_patterns[signal] = self.detect_periodic_patterns(signal_array, axis=2)

                # Anomaly patterns
                if include_anomalies:
                    anomaly_patterns[signal] = self.detect_anomaly_patterns(signal_array)
            except Exception:
                continue

        # Process patterns
        if include_process:
            process_patterns = self.identify_process_patterns(voxel_data, signals)

        return PatternResults(
            spatial_patterns=spatial_patterns,
            temporal_patterns=temporal_patterns,
            anomaly_patterns=anomaly_patterns,
            process_patterns=process_patterns,
        )
