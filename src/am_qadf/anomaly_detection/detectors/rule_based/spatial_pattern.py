"""
Spatial Pattern Anomaly Detection

Detects anomalies in 3D spatial distribution, geometry-based patterns,
and spatial clustering of anomalies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from scipy.spatial.distance import cdist
from scipy import stats

from ...core.base_detector import (
    BaseAnomalyDetector,
    AnomalyDetectionResult,
    AnomalyDetectionConfig,
)

logger = logging.getLogger(__name__)


class SpatialPatternDetector(BaseAnomalyDetector):
    """
    Spatial pattern anomaly detector.

    Detects anomalies in 3D spatial distribution, geometry-based patterns,
    and spatial clustering of anomalies.
    """

    def __init__(
        self,
        neighborhood_radius: float = 0.5,
        use_density_analysis: bool = True,
        use_gradient_analysis: bool = True,
        use_clustering_analysis: bool = True,
        min_cluster_size: int = 3,
        threshold_percentile: float = 95.0,
        config: Optional[AnomalyDetectionConfig] = None,
    ):
        """
        Initialize spatial pattern detector.

        Args:
            neighborhood_radius: Radius for neighborhood analysis (default: 0.5)
            use_density_analysis: Analyze local density patterns
            use_gradient_analysis: Detect spatial gradients
            use_clustering_analysis: Detect spatial clusters
            min_cluster_size: Minimum size for spatial clusters (default: 3)
            config: Optional detector configuration
        """
        if config is None:
            config = AnomalyDetectionConfig()
        super().__init__(config)

        self.neighborhood_radius = neighborhood_radius
        self.use_density_analysis = use_density_analysis
        self.use_gradient_analysis = use_gradient_analysis
        self.use_clustering_analysis = use_clustering_analysis
        self.min_cluster_size = min_cluster_size
        self.threshold_percentile = threshold_percentile

        self.baseline_density_ = None
        self.baseline_gradient_ = None
        self.coordinates_ = None

    def fit(
        self,
        data: Union[np.ndarray, pd.DataFrame, Dict[Tuple[int, int, int], Any]],
        labels: Optional[np.ndarray] = None,
    ) -> "SpatialPatternDetector":
        """
        Fit the detector by learning baseline spatial patterns.

        Args:
            data: Training data (normal data only)
            labels: Not used for spatial pattern detection

        Returns:
            Self for method chaining
        """
        array_data = self._preprocess_data(data)

        # Extract coordinates if available
        coordinates = None
        if hasattr(self, "_voxel_coordinates"):
            coordinates = self._voxel_coordinates
        elif isinstance(data, dict):
            # Try to extract from fused voxel data
            coordinates = []
            for key, value in data.items():
                if hasattr(value, "voxel_coordinates") and value.voxel_coordinates:
                    coords = value.voxel_coordinates
                    coordinates.append([coords.x, coords.y, coords.z])
                else:
                    # Use key as coordinates if it's a tuple
                    if isinstance(key, tuple) and len(key) == 3:
                        coordinates.append([float(key[0]), float(key[1]), float(key[2])])
                    else:
                        coordinates.append([0.0, 0.0, 0.0])

        if coordinates is None:
            # Generate default coordinates from indices
            n_samples = array_data.shape[0]
            coordinates = np.column_stack(
                [
                    np.arange(n_samples) % 10,
                    (np.arange(n_samples) // 10) % 10,
                    np.arange(n_samples) // 100,
                ]
            )

        self.coordinates_ = np.array(coordinates)

        # Learn baseline spatial patterns
        # Calculate distances if needed for density or gradient analysis
        if self.use_density_analysis or self.use_gradient_analysis:
            distances = cdist(self.coordinates_, self.coordinates_)

        if self.use_density_analysis:
            # Calculate baseline density
            neighborhood_mask = distances <= self.neighborhood_radius
            local_densities = np.sum(neighborhood_mask, axis=1) - 1  # Exclude self
            self.baseline_density_ = {
                "mean": np.mean(local_densities),
                "std": np.std(local_densities),
            }

        if self.use_gradient_analysis:
            # Calculate baseline gradients
            gradients = []
            for i in range(len(self.coordinates_)):
                neighbors = distances[i] <= self.neighborhood_radius
                neighbor_indices = np.where(neighbors)[0]
                neighbor_indices = neighbor_indices[neighbor_indices != i]

                if len(neighbor_indices) > 0:
                    # Calculate gradient for each feature
                    for j in range(array_data.shape[1]):
                        if np.isfinite(array_data[i, j]):
                            neighbor_values = array_data[neighbor_indices, j]
                            valid_neighbors = neighbor_values[np.isfinite(neighbor_values)]

                            if len(valid_neighbors) > 0:
                                gradient = abs(array_data[i, j] - np.mean(valid_neighbors))
                                gradients.append(gradient)

            if gradients:
                self.baseline_gradient_ = {
                    "mean": np.mean(gradients),
                    "std": np.std(gradients),
                }
            else:
                self.baseline_gradient_ = {"mean": 0.0, "std": 1.0}

        self.is_fitted = True
        logger.info(f"SpatialPatternDetector fitted on {array_data.shape[0]} samples")
        return self

    def predict(self, data: Union[np.ndarray, pd.DataFrame, Dict[Tuple[int, int, int], Any]]) -> List[AnomalyDetectionResult]:
        """
        Detect anomalies based on spatial pattern deviations.

        Args:
            data: Data to detect anomalies in

        Returns:
            List of AnomalyDetectionResult objects
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")

        array_data = self._preprocess_data(data)

        # Extract or generate coordinates
        coordinates = None
        if hasattr(self, "_voxel_coordinates"):
            coordinates = self._voxel_coordinates
        elif isinstance(data, dict):
            coordinates = []
            for key, value in data.items():
                if hasattr(value, "voxel_coordinates") and value.voxel_coordinates:
                    coords = value.voxel_coordinates
                    coordinates.append([coords.x, coords.y, coords.z])
                elif isinstance(key, tuple) and len(key) == 3:
                    coordinates.append([float(key[0]), float(key[1]), float(key[2])])
                else:
                    coordinates.append([0.0, 0.0, 0.0])

        if coordinates is None:
            n_samples = array_data.shape[0]
            coordinates = np.column_stack(
                [
                    np.arange(n_samples) % 10,
                    (np.arange(n_samples) // 10) % 10,
                    np.arange(n_samples) // 100,
                ]
            )

        coordinates = np.array(coordinates)

        # Calculate spatial anomaly scores
        spatial_scores = np.zeros(array_data.shape[0])

        # Calculate distances
        distances = cdist(coordinates, coordinates)

        # Density analysis
        if self.use_density_analysis and self.baseline_density_:
            neighborhood_mask = distances <= self.neighborhood_radius
            local_densities = np.sum(neighborhood_mask, axis=1) - 1

            # Anomaly score based on density deviation
            density_deviations = abs(local_densities - self.baseline_density_["mean"]) / (
                self.baseline_density_["std"] + 1e-10
            )
            spatial_scores += density_deviations

        # Gradient analysis
        if self.use_gradient_analysis and self.baseline_gradient_:
            gradient_scores = np.zeros(array_data.shape[0])

            for i in range(len(coordinates)):
                neighbors = distances[i] <= self.neighborhood_radius
                neighbor_indices = np.where(neighbors)[0]
                neighbor_indices = neighbor_indices[neighbor_indices != i]

                if len(neighbor_indices) > 0:
                    feature_gradients = []
                    for j in range(array_data.shape[1]):
                        if np.isfinite(array_data[i, j]):
                            neighbor_values = array_data[neighbor_indices, j]
                            valid_neighbors = neighbor_values[np.isfinite(neighbor_values)]

                            if len(valid_neighbors) > 0:
                                gradient = abs(array_data[i, j] - np.mean(valid_neighbors))
                                feature_gradients.append(gradient)

                    if feature_gradients:
                        avg_gradient = np.mean(feature_gradients)
                        gradient_deviation = abs(avg_gradient - self.baseline_gradient_["mean"]) / (
                            self.baseline_gradient_["std"] + 1e-10
                        )
                        gradient_scores[i] = gradient_deviation

            spatial_scores += gradient_scores

        # Clustering analysis
        if self.use_clustering_analysis:
            # Detect spatial clusters of similar values
            cluster_scores = np.zeros(array_data.shape[0])

            for j in range(array_data.shape[1]):
                feature_values = array_data[:, j]
                valid_mask = np.isfinite(feature_values)

                if np.sum(valid_mask) >= self.min_cluster_size:
                    # Find clusters of similar values
                    valid_values = feature_values[valid_mask]
                    valid_coords = coordinates[valid_mask]

                    # Use distance and value similarity
                    value_distances = np.abs(valid_values[:, None] - valid_values[None, :])
                    spatial_distances = cdist(valid_coords, valid_coords)

                    # Combine distance metrics
                    combined_distances = value_distances / (np.std(valid_values) + 1e-10) + spatial_distances / (
                        np.max(spatial_distances) + 1e-10
                    )

                    # Find dense clusters
                    cluster_mask = combined_distances < 0.5
                    cluster_sizes = np.sum(cluster_mask, axis=1)

                    # Anomalies are points in small clusters or isolated points
                    for idx, size in enumerate(cluster_sizes):
                        if size < self.min_cluster_size:
                            original_idx = np.where(valid_mask)[0][idx]
                            cluster_scores[original_idx] += 1.0

            spatial_scores += 0.5 * cluster_scores

        # Normalize scores
        if np.max(spatial_scores) > 0:
            spatial_scores = spatial_scores / (np.max(spatial_scores) + 1e-10)

        # Determine anomalies
        threshold = (
            np.percentile(spatial_scores[np.isfinite(spatial_scores)], self.threshold_percentile)
            if np.any(np.isfinite(spatial_scores))
            else 0.1
        )
        is_anomaly = spatial_scores > threshold

        # Get indices and coordinates
        indices = getattr(self, "_voxel_indices", None)
        if indices is None:
            indices = [(int(c[0]), int(c[1]), int(c[2])) for c in coordinates]

        coord_list = [(float(c[0]), float(c[1]), float(c[2])) for c in coordinates]

        # Create results
        results = self._create_results(scores=spatial_scores, indices=indices, coordinates=coord_list)

        return results
