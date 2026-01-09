"""
Synthetic Anomaly Generation for Anomaly Detection Evaluation

This module provides utilities for generating synthetic anomalies in fused
multimodal data for evaluating anomaly detection methods.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class AnomalyInjectionType(Enum):
    """Types of synthetic anomalies to inject."""

    POINT_OUTLIER = "point_outlier"  # Single point anomaly
    SPATIAL_CLUSTER = "spatial_cluster"  # Spatial cluster of anomalies
    TEMPORAL_DRIFT = "temporal_drift"  # Gradual parameter drift
    SUDDEN_CHANGE = "sudden_change"  # Sudden parameter change
    CORRELATION_BREAK = "correlation_break"  # Break in parameter correlations
    CONTEXTUAL = "contextual"  # Contextual anomaly


@dataclass
class AnomalyInjectionConfig:
    """Configuration for synthetic anomaly injection."""

    # Anomaly type
    anomaly_type: AnomalyInjectionType = AnomalyInjectionType.POINT_OUTLIER

    # Magnitude
    magnitude: float = 3.0  # Standard deviations or multiplier

    # Spatial parameters
    cluster_size: int = 10  # Number of voxels in cluster
    cluster_radius: float = 2.0  # Cluster radius in voxels

    # Temporal parameters
    drift_rate: float = 0.1  # Rate of drift per time step
    change_point: Optional[int] = None  # Time step for sudden change

    # Feature selection
    target_features: Optional[List[str]] = None  # Features to modify
    feature_weights: Optional[Dict[str, float]] = None  # Weights for features

    # Random seed
    random_seed: Optional[int] = None


class SyntheticAnomalyGenerator:
    """
    Generator for synthetic anomalies in fused multimodal data.

    Creates various types of anomalies for evaluating detection methods.
    """

    def __init__(self, config: Optional[AnomalyInjectionConfig] = None):
        """
        Initialize the anomaly generator.

        Args:
            config: Configuration for anomaly generation
        """
        self.config = config or AnomalyInjectionConfig()
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)

        logger.info(f"SyntheticAnomalyGenerator initialized with type: {self.config.anomaly_type.value}")

    def inject_anomalies(
        self,
        data: Dict[Tuple[int, int, int], Any],
        n_anomalies: int = 10,
        anomaly_labels: Optional[Dict[Tuple[int, int, int], bool]] = None,
    ) -> Tuple[Dict[Tuple[int, int, int], Any], Dict[Tuple[int, int, int], bool]]:
        """
        Inject synthetic anomalies into data.

        Args:
            data: Original fused voxel data
            n_anomalies: Number of anomalies to inject
            anomaly_labels: Optional existing labels (will be updated)

        Returns:
            Tuple of (modified_data, anomaly_labels)
        """
        if anomaly_labels is None:
            anomaly_labels = {idx: False for idx in data.keys()}

        modified_data = data.copy()

        if self.config.anomaly_type == AnomalyInjectionType.POINT_OUTLIER:
            modified_data, anomaly_labels = self._inject_point_outliers(modified_data, anomaly_labels, n_anomalies)
        elif self.config.anomaly_type == AnomalyInjectionType.SPATIAL_CLUSTER:
            modified_data, anomaly_labels = self._inject_spatial_clusters(modified_data, anomaly_labels, n_anomalies)
        elif self.config.anomaly_type == AnomalyInjectionType.TEMPORAL_DRIFT:
            modified_data, anomaly_labels = self._inject_temporal_drift(modified_data, anomaly_labels)
        elif self.config.anomaly_type == AnomalyInjectionType.SUDDEN_CHANGE:
            modified_data, anomaly_labels = self._inject_sudden_change(modified_data, anomaly_labels)
        elif self.config.anomaly_type == AnomalyInjectionType.CORRELATION_BREAK:
            modified_data, anomaly_labels = self._inject_correlation_break(modified_data, anomaly_labels, n_anomalies)
        else:
            logger.warning(f"Unknown anomaly type: {self.config.anomaly_type}")

        logger.info(f"Injected {sum(anomaly_labels.values())} anomalies")
        return modified_data, anomaly_labels

    def _inject_point_outliers(
        self,
        data: Dict[Tuple[int, int, int], Any],
        labels: Dict[Tuple[int, int, int], bool],
        n_anomalies: int,
    ) -> Tuple[Dict[Tuple[int, int, int], Any], Dict[Tuple[int, int, int], bool]]:
        """Inject point outlier anomalies."""
        # Select random voxels
        voxel_indices = list(data.keys())
        selected = np.random.choice(len(voxel_indices), size=min(n_anomalies, len(voxel_indices)), replace=False)

        # Calculate feature statistics
        feature_stats = self._calculate_feature_statistics(data)

        for idx_pos in selected:
            voxel_idx = voxel_indices[idx_pos]
            item = data[voxel_idx]

            # Modify features
            for feature_name, (mean, std) in feature_stats.items():
                if self.config.target_features is None or feature_name in self.config.target_features:
                    # Add outlier value
                    outlier_value = mean + self.config.magnitude * std * np.random.choice([-1, 1])
                    if hasattr(item, feature_name):
                        setattr(item, feature_name, outlier_value)

            labels[voxel_idx] = True

        return data, labels

    def _inject_spatial_clusters(
        self,
        data: Dict[Tuple[int, int, int], Any],
        labels: Dict[Tuple[int, int, int], bool],
        n_clusters: int,
    ) -> Tuple[Dict[Tuple[int, int, int], Any], Dict[Tuple[int, int, int], bool]]:
        """Inject spatial cluster anomalies."""
        voxel_indices = np.array(list(data.keys()))

        # Select cluster centers
        selected_centers = np.random.choice(len(voxel_indices), size=min(n_clusters, len(voxel_indices)), replace=False)

        feature_stats = self._calculate_feature_statistics(data)

        for center_idx in selected_centers:
            center = voxel_indices[center_idx]

            # Find nearby voxels
            distances = np.array([np.sqrt(sum((a - b) ** 2 for a, b in zip(idx, center))) for idx in voxel_indices])
            nearby_mask = distances <= self.config.cluster_radius
            nearby_indices_positions = np.where(nearby_mask)[0]

            # Limit cluster size
            if len(nearby_indices_positions) > self.config.cluster_size:
                selected_positions = np.random.choice(
                    nearby_indices_positions,
                    size=self.config.cluster_size,
                    replace=False,
                )
            else:
                selected_positions = nearby_indices_positions

            # Modify nearby voxels
            for pos in selected_positions:
                voxel_idx = voxel_indices[pos]
                item = data[tuple(voxel_idx)]

                for feature_name, (mean, std) in feature_stats.items():
                    if self.config.target_features is None or feature_name in self.config.target_features:
                        outlier_value = mean + self.config.magnitude * std * np.random.choice([-1, 1])
                        if hasattr(item, feature_name):
                            setattr(item, feature_name, outlier_value)

                labels[tuple(voxel_idx)] = True

        return data, labels

    def _inject_temporal_drift(
        self,
        data: Dict[Tuple[int, int, int], Any],
        labels: Dict[Tuple[int, int, int], bool],
    ) -> Tuple[Dict[Tuple[int, int, int], Any], Dict[Tuple[int, int, int], bool]]:
        """Inject temporal drift anomalies."""
        # Sort by layer number or build time if available
        sorted_items = sorted(data.items(), key=lambda x: self._get_temporal_key(x[1]))

        feature_stats = self._calculate_feature_statistics(data)
        n_items = len(sorted_items)

        # Apply drift starting from middle
        drift_start = n_items // 2

        for i, (voxel_idx, item) in enumerate(sorted_items[drift_start:], start=drift_start):
            drift_factor = 1.0 + self.config.drift_rate * (i - drift_start) / n_items

            for feature_name, (mean, std) in feature_stats.items():
                if self.config.target_features is None or feature_name in self.config.target_features:
                    current_value = getattr(item, feature_name, mean)
                    if isinstance(current_value, (int, float)):
                        new_value = current_value * drift_factor
                        setattr(item, feature_name, new_value)

            labels[voxel_idx] = True

        return data, labels

    def _inject_sudden_change(
        self,
        data: Dict[Tuple[int, int, int], Any],
        labels: Dict[Tuple[int, int, int], bool],
    ) -> Tuple[Dict[Tuple[int, int, int], Any], Dict[Tuple[int, int, int], bool]]:
        """Inject sudden change anomalies."""
        sorted_items = sorted(data.items(), key=lambda x: self._get_temporal_key(x[1]))

        if self.config.change_point is None:
            change_point = len(sorted_items) // 2
        else:
            change_point = self.config.change_point

        feature_stats = self._calculate_feature_statistics(data)

        # Apply sudden change
        for voxel_idx, item in sorted_items[change_point:]:
            for feature_name, (mean, std) in feature_stats.items():
                if self.config.target_features is None or feature_name in self.config.target_features:
                    change_value = mean + self.config.magnitude * std * np.random.choice([-1, 1])
                    if hasattr(item, feature_name):
                        setattr(item, feature_name, change_value)

            labels[voxel_idx] = True

        return data, labels

    def _inject_correlation_break(
        self,
        data: Dict[Tuple[int, int, int], Any],
        labels: Dict[Tuple[int, int, int], bool],
        n_anomalies: int,
    ) -> Tuple[Dict[Tuple[int, int, int], Any], Dict[Tuple[int, int, int], bool]]:
        """Inject correlation break anomalies."""
        # Select random voxels
        voxel_indices = list(data.keys())
        selected = np.random.choice(len(voxel_indices), size=min(n_anomalies, len(voxel_indices)), replace=False)

        # Calculate correlations
        feature_correlations = self._calculate_feature_correlations(data)

        for idx_pos in selected:
            voxel_idx = voxel_indices[idx_pos]
            item = data[voxel_idx]

            # Break correlation by inverting relationship
            if len(feature_correlations) > 1:
                feat_names = list(feature_correlations.keys())
                feat1, feat2 = np.random.choice(feat_names, size=2, replace=False)

                val1 = getattr(item, feat1, 0.0)
                val2 = getattr(item, feat2, 0.0)

                # Invert the relationship
                if hasattr(item, feat2):
                    setattr(item, feat2, -val2 * self.config.magnitude)

            labels[voxel_idx] = True

        return data, labels

    def _calculate_feature_statistics(self, data: Dict[Tuple[int, int, int], Any]) -> Dict[str, Tuple[float, float]]:
        """Calculate mean and std for each feature."""
        if not data:
            return {}

        first_item = next(iter(data.values()))
        feature_names = []

        if hasattr(first_item, "__dict__"):
            for key, value in first_item.__dict__.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    feature_names.append(key)

        stats = {}
        for feat_name in feature_names:
            values = [getattr(item, feat_name, 0.0) for item in data.values()]
            values = [v for v in values if isinstance(v, (int, float)) and not np.isnan(v)]
            if values:
                stats[feat_name] = (
                    np.mean(values),
                    np.std(values) if len(values) > 1 else 1.0,
                )

        return stats

    def _calculate_feature_correlations(self, data: Dict[Tuple[int, int, int], Any]) -> Dict[str, float]:
        """Calculate feature correlations."""
        # Simplified - return feature names
        if not data:
            return {}

        first_item = next(iter(data.values()))
        feature_names = []

        if hasattr(first_item, "__dict__"):
            for key, value in first_item.__dict__.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    feature_names.append(key)

        return {name: 1.0 for name in feature_names}

    def _get_temporal_key(self, item: Any) -> float:
        """Get temporal key for sorting (layer number or timestamp)."""
        if hasattr(item, "layer_number") and item.layer_number is not None:
            return float(item.layer_number)
        elif hasattr(item, "build_time") and item.build_time is not None:
            if isinstance(item.build_time, datetime):
                return item.build_time.timestamp()
            return 0.0
        return 0.0
