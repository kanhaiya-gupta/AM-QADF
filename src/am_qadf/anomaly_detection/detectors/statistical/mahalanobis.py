"""
Mahalanobis Distance Based Anomaly Detection

Detects anomalies using Mahalanobis distance, which accounts for
correlations between features.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from scipy.linalg import inv, LinAlgError

from ...core.base_detector import (
    BaseAnomalyDetector,
    AnomalyDetectionResult,
    AnomalyDetectionConfig,
)

logger = logging.getLogger(__name__)


class MahalanobisDetector(BaseAnomalyDetector):
    """
    Mahalanobis distance based anomaly detector.

    Detects anomalies using Mahalanobis distance, which accounts for
    feature correlations. Points with high Mahalanobis distance are anomalies.
    """

    def __init__(self, threshold: float = 3.0, config: Optional[AnomalyDetectionConfig] = None):
        """
        Initialize Mahalanobis detector.

        Args:
            threshold: Mahalanobis distance threshold (default: 3.0)
            config: Optional detector configuration
        """
        if config is None:
            config = AnomalyDetectionConfig(threshold=threshold, normalize_features=False)
        else:
            config.threshold = threshold
            config.normalize_features = False

        super().__init__(config)
        self.threshold = threshold
        self.mean_ = None
        self.cov_ = None
        self.inv_cov_ = None

    def fit(
        self,
        data: Union[np.ndarray, pd.DataFrame, Dict[Tuple[int, int, int], Any]],
        labels: Optional[np.ndarray] = None,
    ) -> "MahalanobisDetector":
        """
        Fit the detector on normal data.

        Args:
            data: Training data (normal data only)
            labels: Not used for Mahalanobis (unsupervised)

        Returns:
            Self for method chaining
        """
        array_data = self._preprocess_data(data)

        # Calculate mean and covariance
        self.mean_ = np.mean(array_data, axis=0)
        cov = np.cov(array_data.T)

        # Ensure covariance is 2D (handle single feature case)
        if cov.ndim == 0:
            cov = np.array([[cov]])
        elif cov.ndim == 1:
            cov = np.diag(cov)

        self.cov_ = cov

        # Handle singular covariance matrix
        try:
            self.inv_cov_ = inv(self.cov_)
        except (LinAlgError, ValueError) as e:
            # Add small regularization if singular
            self.cov_ += np.eye(self.cov_.shape[0]) * 1e-6
            self.inv_cov_ = inv(self.cov_)

        self.is_fitted = True
        logger.info(f"MahalanobisDetector fitted on {array_data.shape[0]} samples")
        return self

    def predict(self, data: Union[np.ndarray, pd.DataFrame, Dict[Tuple[int, int, int], Any]]) -> List[AnomalyDetectionResult]:
        """
        Detect anomalies using Mahalanobis distance.

        Args:
            data: Data to detect anomalies in

        Returns:
            List of AnomalyDetectionResult objects
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")

        array_data = self._preprocess_data(data)

        # Calculate Mahalanobis distances
        distances = []
        for point in array_data:
            diff = point - self.mean_
            distance = np.sqrt(diff @ self.inv_cov_ @ diff)
            distances.append(distance)

        distances = np.array(distances)

        # Normalize scores (divide by threshold for relative scoring)
        scores = distances / self.threshold if self.threshold > 0 else distances

        # Get indices
        indices = getattr(self, "_voxel_indices", None)
        if indices is None:
            indices = [(0, 0, 0)] * len(scores)

        coordinates = [(0.0, 0.0, 0.0)] * len(scores)

        # Create results
        results = self._create_results(scores=scores, indices=indices, coordinates=coordinates)

        return results
