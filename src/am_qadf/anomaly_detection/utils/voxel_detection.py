"""
Voxel-Based Anomaly Detection

Detects anomalies in voxel domain data:
- Spatial Anomaly Detection: Detect anomalies in voxel space
- Temporal Anomaly Detection: Detect anomalies over time/layers
- Multi-Signal Anomaly Detection: Detect anomalies across signals
- Anomaly Localization: Localize anomalies to specific voxels
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

# Try to import anomaly detection components from new structure
try:
    from ..core.base_detector import (
        BaseAnomalyDetector,
        AnomalyDetectionResult,
    )
    from ..core.types import AnomalyType
    from ..detectors.statistical import ZScoreDetector, IQRDetector, MahalanobisDetector
    from ..detectors.clustering import (
        IsolationForestDetector,
        DBSCANDetector,
        LOFDetector,
    )

    ANOMALY_DETECTION_AVAILABLE = True
except ImportError:
    # Fallback: define minimal interfaces
    ANOMALY_DETECTION_AVAILABLE = False
    from enum import Enum

    class AnomalyType(Enum):
        SPATIAL = "spatial"
        TEMPORAL = "temporal"
        MULTI_SIGNAL = "multi_signal"
        POINT = "point"

    @dataclass
    class AnomalyDetectionResult:
        is_anomaly: bool
        anomaly_score: float
        anomaly_type: AnomalyType


@dataclass
class VoxelAnomalyResult:
    """Anomaly detection result for voxel domain."""

    anomaly_map: np.ndarray  # Boolean array indicating anomalous voxels
    anomaly_scores: np.ndarray  # Anomaly scores per voxel
    anomaly_types: Optional[Dict[str, np.ndarray]] = None  # Anomaly types per signal
    num_anomalies: int = 0
    anomaly_fraction: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        result = {
            "num_anomalies": self.num_anomalies,
            "anomaly_fraction": self.anomaly_fraction,
            "anomaly_map_shape": self.anomaly_map.shape,
            "anomaly_scores_shape": self.anomaly_scores.shape,
        }
        if self.anomaly_types:
            result["anomaly_types"] = list(self.anomaly_types.keys())
        return result


class VoxelAnomalyDetector:
    """
    Detects anomalies in voxel domain data.

    Provides spatial, temporal, and multi-signal anomaly detection
    capabilities specifically designed for voxel domain representations.
    """

    def __init__(self, method: str = "isolation_forest", threshold: float = 0.5):
        """
        Initialize voxel anomaly detector.

        Args:
            method: Detection method ('z_score', 'iqr', 'isolation_forest', 'dbscan', 'lof')
            threshold: Anomaly score threshold (0-1)
        """
        self.method = method
        self.threshold = threshold
        self._detector = None

    def _get_detector(self):
        """Get or create detector instance."""
        if self._detector is not None:
            return self._detector

        if not ANOMALY_DETECTION_AVAILABLE:
            # Use simple statistical method
            return None

        if self.method == "z_score":
            self._detector = ZScoreDetector(threshold=3.0)
        elif self.method == "iqr":
            self._detector = IQRDetector()
        elif self.method == "isolation_forest":
            self._detector = IsolationForestDetector(contamination=0.1)
        elif self.method == "dbscan":
            self._detector = DBSCANDetector(eps=0.5, min_samples=5)
        elif self.method == "lof":
            self._detector = LOFDetector(n_neighbors=20, contamination=0.1)
        else:
            self._detector = IsolationForestDetector(contamination=0.1)

        return self._detector

    def detect_spatial_anomalies(self, signal_array: np.ndarray, use_neighbors: bool = True) -> VoxelAnomalyResult:
        """
        Detect spatial anomalies in voxel space.

        Args:
            signal_array: Signal array (3D)
            use_neighbors: Whether to consider neighboring voxels

        Returns:
            VoxelAnomalyResult object
        """
        # Flatten array for detection
        flat_array = signal_array.flatten()
        valid_mask = (~np.isnan(flat_array)) & (flat_array != 0.0)
        valid_values = flat_array[valid_mask]

        if len(valid_values) < 10:
            # Not enough data
            anomaly_map = np.zeros_like(signal_array, dtype=bool)
            anomaly_scores = np.zeros_like(signal_array, dtype=np.float32)
            return VoxelAnomalyResult(
                anomaly_map=anomaly_map,
                anomaly_scores=anomaly_scores,
                num_anomalies=0,
                anomaly_fraction=0.0,
            )

        # Detect anomalies
        if self.method == "z_score":
            # Z-score based detection
            mean_val = np.mean(valid_values)
            std_val = np.std(valid_values)
            if std_val > 0:
                z_scores = np.abs((flat_array - mean_val) / std_val)
                anomaly_scores_flat = np.clip(z_scores / 3.0, 0.0, 1.0)  # Normalize to 0-1
            else:
                anomaly_scores_flat = np.zeros_like(flat_array)
        elif self.method == "iqr":
            # IQR based detection
            q25 = np.percentile(valid_values, 25)
            q75 = np.percentile(valid_values, 75)
            iqr = q75 - q25
            if iqr > 0:
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr
                # Score based on distance from bounds
                distances = np.minimum(np.abs(flat_array - lower_bound), np.abs(flat_array - upper_bound))
                max_distance = np.max(distances[valid_mask]) if np.any(valid_mask) else 1.0
                anomaly_scores_flat = 1.0 - np.clip(distances / (max_distance + 1e-10), 0.0, 1.0)
            else:
                anomaly_scores_flat = np.zeros_like(flat_array)
        else:
            # Use detector if available
            detector = self._get_detector()
            if detector and ANOMALY_DETECTION_AVAILABLE:
                try:
                    # Reshape for detector (needs 2D)
                    values_2d = valid_values.reshape(-1, 1)
                    results = detector.predict(values_2d)

                    # Create scores array
                    anomaly_scores_flat = np.zeros_like(flat_array, dtype=np.float32)
                    valid_indices = np.where(valid_mask)[0]
                    for i, idx in enumerate(valid_indices):
                        if hasattr(results, "anomaly_scores"):
                            anomaly_scores_flat[idx] = results.anomaly_scores[i] if i < len(results.anomaly_scores) else 0.0
                        elif hasattr(results, "is_anomaly"):
                            anomaly_scores_flat[idx] = 1.0 if results.is_anomaly[i] else 0.0
                except Exception:
                    # Fallback to z-score
                    mean_val = np.mean(valid_values)
                    std_val = np.std(valid_values)
                    if std_val > 0:
                        z_scores = np.abs((flat_array - mean_val) / std_val)
                        anomaly_scores_flat = np.clip(z_scores / 3.0, 0.0, 1.0)
                    else:
                        anomaly_scores_flat = np.zeros_like(flat_array)
            else:
                # Fallback to z-score
                mean_val = np.mean(valid_values)
                std_val = np.std(valid_values)
                if std_val > 0:
                    z_scores = np.abs((flat_array - mean_val) / std_val)
                    anomaly_scores_flat = np.clip(z_scores / 3.0, 0.0, 1.0)
                else:
                    anomaly_scores_flat = np.zeros_like(flat_array)

        # Reshape to original shape
        anomaly_scores = anomaly_scores_flat.reshape(signal_array.shape)
        anomaly_map = anomaly_scores >= self.threshold

        # Calculate statistics
        num_anomalies = np.sum(anomaly_map)
        anomaly_fraction = num_anomalies / signal_array.size if signal_array.size > 0 else 0.0

        return VoxelAnomalyResult(
            anomaly_map=anomaly_map,
            anomaly_scores=anomaly_scores,
            num_anomalies=int(num_anomalies),
            anomaly_fraction=anomaly_fraction,
        )

    def detect_temporal_anomalies(
        self, signal_array: np.ndarray, axis: int = 2  # Usually z-axis for layers/time
    ) -> VoxelAnomalyResult:
        """
        Detect temporal anomalies along an axis.

        Args:
            signal_array: Signal array (3D)
            axis: Axis representing time/layers (usually 2)

        Returns:
            VoxelAnomalyResult object
        """
        if axis >= len(signal_array.shape):
            axis = 2

        # Aggregate values per layer/time step
        layer_values = []
        layer_indices = []

        for i in range(signal_array.shape[axis]):
            if axis == 0:
                slice_data = signal_array[i, :, :]
            elif axis == 1:
                slice_data = signal_array[:, i, :]
            else:  # axis == 2
                slice_data = signal_array[:, :, i]

            valid_values = slice_data[(~np.isnan(slice_data)) & (slice_data != 0.0)]
            if len(valid_values) > 0:
                layer_values.append(np.mean(valid_values))
                layer_indices.append(i)

        if len(layer_values) < 3:
            # Not enough temporal data
            anomaly_map = np.zeros_like(signal_array, dtype=bool)
            anomaly_scores = np.zeros_like(signal_array, dtype=np.float32)
            return VoxelAnomalyResult(
                anomaly_map=anomaly_map,
                anomaly_scores=anomaly_scores,
                num_anomalies=0,
                anomaly_fraction=0.0,
            )

        # Detect anomalies in temporal sequence
        layer_array = np.array(layer_values)
        mean_val = np.mean(layer_array)
        std_val = np.std(layer_array)

        if std_val > 0:
            z_scores = np.abs((layer_array - mean_val) / std_val)
            layer_anomaly_scores = np.clip(z_scores / 3.0, 0.0, 1.0)
        else:
            layer_anomaly_scores = np.zeros(len(layer_array))

        # Map back to 3D array
        anomaly_scores = np.zeros_like(signal_array, dtype=np.float32)
        for i, layer_idx in enumerate(layer_indices):
            score = layer_anomaly_scores[i]
            if axis == 0:
                anomaly_scores[layer_idx, :, :] = score
            elif axis == 1:
                anomaly_scores[:, layer_idx, :] = score
            else:  # axis == 2
                anomaly_scores[:, :, layer_idx] = score

        anomaly_map = anomaly_scores >= self.threshold

        num_anomalies = np.sum(anomaly_map)
        anomaly_fraction = num_anomalies / signal_array.size if signal_array.size > 0 else 0.0

        return VoxelAnomalyResult(
            anomaly_map=anomaly_map,
            anomaly_scores=anomaly_scores,
            num_anomalies=int(num_anomalies),
            anomaly_fraction=anomaly_fraction,
        )

    def detect_multi_signal_anomalies(
        self,
        voxel_data: Any,
        signals: List[str],
        combine_method: str = "max",  # 'max', 'mean', 'weighted'
    ) -> VoxelAnomalyResult:
        """
        Detect anomalies across multiple signals.

        Args:
            voxel_data: Voxel domain data object
            signals: List of signal names to analyze
            combine_method: Method to combine anomaly scores ('max', 'mean', 'weighted')

        Returns:
            VoxelAnomalyResult object
        """
        if not signals:
            raise ValueError("At least one signal must be provided")

        # Detect anomalies for each signal
        signal_results = {}
        for signal in signals:
            try:
                signal_array = voxel_data.get_signal_array(signal, default=0.0)
                result = self.detect_spatial_anomalies(signal_array)
                signal_results[signal] = result
            except Exception:
                continue

        if not signal_results:
            raise ValueError("No valid signals found")

        # Combine anomaly scores
        first_result = list(signal_results.values())[0]
        combined_scores = np.zeros_like(first_result.anomaly_scores, dtype=np.float32)
        combined_map = np.zeros_like(first_result.anomaly_map, dtype=bool)

        if combine_method == "max":
            # Maximum score across signals
            for result in signal_results.values():
                combined_scores = np.maximum(combined_scores, result.anomaly_scores)
        elif combine_method == "mean":
            # Mean score across signals
            count = 0
            for result in signal_results.values():
                combined_scores += result.anomaly_scores
                count += 1
            if count > 0:
                combined_scores /= count
        else:  # weighted (equal weights for now)
            # Weighted average
            count = 0
            for result in signal_results.values():
                combined_scores += result.anomaly_scores
                count += 1
            if count > 0:
                combined_scores /= count

        combined_map = combined_scores >= self.threshold

        num_anomalies = np.sum(combined_map)
        anomaly_fraction = num_anomalies / combined_map.size if combined_map.size > 0 else 0.0

        # Store per-signal anomaly types
        anomaly_types = {signal: result.anomaly_map for signal, result in signal_results.items()}

        return VoxelAnomalyResult(
            anomaly_map=combined_map,
            anomaly_scores=combined_scores,
            anomaly_types=anomaly_types,
            num_anomalies=int(num_anomalies),
            anomaly_fraction=anomaly_fraction,
        )

    def localize_anomalies(self, anomaly_result: VoxelAnomalyResult, voxel_data: Any) -> Dict[str, Any]:
        """
        Localize anomalies to specific voxel coordinates.

        Args:
            anomaly_result: Anomaly detection result
            voxel_data: Voxel domain data object

        Returns:
            Dictionary with localized anomaly information
        """
        # Get voxel coordinates
        anomaly_coords = np.where(anomaly_result.anomaly_map)

        if len(anomaly_coords[0]) == 0:
            return {"anomaly_voxels": [], "anomaly_coordinates": [], "num_anomalies": 0}

        # Convert to world coordinates
        bbox_min = voxel_data.bbox_min if hasattr(voxel_data, "bbox_min") else (0, 0, 0)
        resolution = voxel_data.resolution if hasattr(voxel_data, "resolution") else 1.0

        anomaly_voxels = []
        anomaly_coordinates = []

        for i in range(len(anomaly_coords[0])):
            voxel_idx = (
                anomaly_coords[0][i],
                anomaly_coords[1][i],
                anomaly_coords[2][i],
            )

            # Convert to world coordinates
            world_coord = tuple(bbox_min[j] + voxel_idx[j] * resolution for j in range(3))

            score = anomaly_result.anomaly_scores[voxel_idx]

            anomaly_voxels.append(
                {
                    "voxel_index": voxel_idx,
                    "world_coordinate": world_coord,
                    "anomaly_score": float(score),
                    "is_anomaly": bool(anomaly_result.anomaly_map[voxel_idx]),
                }
            )

            anomaly_coordinates.append(world_coord)

        return {
            "anomaly_voxels": anomaly_voxels,
            "anomaly_coordinates": anomaly_coordinates,
            "num_anomalies": len(anomaly_voxels),
        }
