"""
DBSCAN (Density-Based Spatial Clustering) Anomaly Detection

Detects anomalies as low-density regions using DBSCAN clustering.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

try:
    from sklearn.cluster import DBSCAN

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ...core.base_detector import (
    BaseAnomalyDetector,
    AnomalyDetectionResult,
    AnomalyDetectionConfig,
)

logger = logging.getLogger(__name__)


class DBSCANDetector(BaseAnomalyDetector):
    """
    DBSCAN-based anomaly detector.

    Uses DBSCAN clustering to identify low-density regions as anomalies.
    Points that are not assigned to any cluster (noise) are considered anomalies.
    """

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        config: Optional[AnomalyDetectionConfig] = None,
    ):
        """
        Initialize DBSCAN detector.

        Args:
            eps: Maximum distance between samples in the same neighborhood
            min_samples: Minimum number of samples in a neighborhood
            config: Optional detector configuration
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for DBSCANDetector. Install with: pip install scikit-learn")

        if config is None:
            config = AnomalyDetectionConfig()
        super().__init__(config)
        self.eps = eps
        self.min_samples = min_samples
        self.clusterer_ = None
        self.labels_ = None

    def fit(
        self,
        data: Union[np.ndarray, pd.DataFrame, Dict[Tuple[int, int, int], Any]],
        labels: Optional[np.ndarray] = None,
    ) -> "DBSCANDetector":
        """
        Fit the detector on normal data.

        Args:
            data: Training data (normal data only)
            labels: Not used for DBSCAN (unsupervised)

        Returns:
            Self for method chaining
        """
        array_data = self._preprocess_data(data)

        # Fit DBSCAN
        self.clusterer_ = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.labels_ = self.clusterer_.fit_predict(array_data)

        self.is_fitted = True
        logger.info(f"DBSCANDetector fitted on {array_data.shape[0]} samples")
        return self

    def predict(self, data: Union[np.ndarray, pd.DataFrame, Dict[Tuple[int, int, int], Any]]) -> List[AnomalyDetectionResult]:
        """
        Detect anomalies using DBSCAN.

        Args:
            data: Data to detect anomalies in

        Returns:
            List of AnomalyDetectionResult objects
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")

        array_data = self._preprocess_data(data)

        # Predict cluster labels
        labels = self.clusterer_.fit_predict(array_data)

        # Calculate distances to nearest cluster core
        scores = np.zeros(len(array_data))
        for i, point in enumerate(array_data):
            if labels[i] == -1:  # Noise point (anomaly)
                # Calculate distance to nearest core point
                core_samples = self.clusterer_.components_
                if len(core_samples) > 0:
                    distances = np.linalg.norm(core_samples - point, axis=1)
                    scores[i] = np.min(distances) / self.eps  # Normalize by eps
                else:
                    scores[i] = 1.0  # Maximum score for noise
            else:
                # In-cluster point, score based on distance to cluster center
                # Use prediction labels to find points in same cluster
                cluster_mask = labels == labels[i]
                if np.any(cluster_mask):
                    cluster_points = array_data[cluster_mask]
                    cluster_center = np.mean(cluster_points, axis=0)
                    distance = np.linalg.norm(point - cluster_center)
                    scores[i] = distance / (self.eps * 10)  # Lower score for in-cluster points

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
