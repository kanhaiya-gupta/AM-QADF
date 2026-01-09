"""
K-Means with Distance Threshold Anomaly Detection

Cluster-based anomaly detection using K-Means clustering.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

try:
    from sklearn.cluster import KMeans

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ...core.base_detector import (
    BaseAnomalyDetector,
    AnomalyDetectionResult,
    AnomalyDetectionConfig,
)

logger = logging.getLogger(__name__)


class KMeansDetector(BaseAnomalyDetector):
    """
    K-Means based anomaly detector.

    Uses K-Means clustering and identifies anomalies as points far from
    their cluster centers.
    """

    def __init__(
        self,
        n_clusters: int = 5,
        distance_threshold: float = 2.0,
        random_state: Optional[int] = None,
        config: Optional[AnomalyDetectionConfig] = None,
    ):
        """
        Initialize K-Means detector.

        Args:
            n_clusters: Number of clusters
            distance_threshold: Distance threshold (in std devs) for anomaly detection
            random_state: Random seed
            config: Optional detector configuration
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for KMeansDetector. Install with: pip install scikit-learn")

        if config is None:
            config = AnomalyDetectionConfig()
        super().__init__(config)
        self.n_clusters = n_clusters
        self.distance_threshold = distance_threshold
        self.random_state = random_state
        self.kmeans_ = None
        self.cluster_centers_ = None
        self.cluster_stds_ = None

    def fit(
        self,
        data: Union[np.ndarray, pd.DataFrame, Dict[Tuple[int, int, int], Any]],
        labels: Optional[np.ndarray] = None,
    ) -> "KMeansDetector":
        """
        Fit the detector on normal data.

        Args:
            data: Training data (normal data only)
            labels: Not used for K-Means (unsupervised)

        Returns:
            Self for method chaining
        """
        array_data = self._preprocess_data(data)

        # Adjust n_clusters if needed
        n_clusters = min(self.n_clusters, len(array_data))

        # Fit K-Means
        self.kmeans_ = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        self.kmeans_.fit(array_data)
        self.cluster_centers_ = self.kmeans_.cluster_centers_

        # Calculate cluster standard deviations
        labels = self.kmeans_.labels_
        self.cluster_stds_ = []
        for i in range(n_clusters):
            cluster_points = array_data[labels == i]
            if len(cluster_points) > 0:
                cluster_std = np.std(cluster_points, axis=0)
                self.cluster_stds_.append(np.mean(cluster_std))
            else:
                self.cluster_stds_.append(1.0)

        self.is_fitted = True
        logger.info(f"KMeansDetector fitted on {array_data.shape[0]} samples with {n_clusters} clusters")
        return self

    def predict(self, data: Union[np.ndarray, pd.DataFrame, Dict[Tuple[int, int, int], Any]]) -> List[AnomalyDetectionResult]:
        """
        Detect anomalies using K-Means.

        Args:
            data: Data to detect anomalies in

        Returns:
            List of AnomalyDetectionResult objects
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")

        array_data = self._preprocess_data(data)

        # Predict cluster assignments
        labels = self.kmeans_.predict(array_data)

        # Calculate distances to cluster centers
        scores = np.zeros(len(array_data))
        for i, point in enumerate(array_data):
            cluster_id = labels[i]
            cluster_center = self.cluster_centers_[cluster_id]
            cluster_std = self.cluster_stds_[cluster_id]

            # Distance to cluster center
            distance = np.linalg.norm(point - cluster_center)

            # Score based on distance relative to cluster std
            if cluster_std > 0:
                score = distance / (cluster_std * self.distance_threshold)
            else:
                score = distance

            scores[i] = score

        # Normalize scores
        max_score = np.max(scores) if np.max(scores) > 0 else 1.0
        scores = scores / max_score if max_score > 0 else scores

        # Get indices
        indices = getattr(self, "_voxel_indices", None)
        if indices is None:
            indices = [(0, 0, 0)] * len(scores)

        coordinates = [(0.0, 0.0, 0.0)] * len(scores)

        # Create results
        results = self._create_results(scores=scores, indices=indices, coordinates=coordinates)

        return results
