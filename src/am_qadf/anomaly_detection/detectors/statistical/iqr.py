"""
Interquartile Range (IQR) Based Anomaly Detection

Detects anomalies using quartile-based thresholds.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

from ...core.base_detector import (
    BaseAnomalyDetector,
    AnomalyDetectionResult,
    AnomalyDetectionConfig,
)

logger = logging.getLogger(__name__)


class IQRDetector(BaseAnomalyDetector):
    """
    Interquartile Range (IQR) based anomaly detector.

    Detects anomalies using the IQR method: points beyond Q1 - 1.5*IQR
    or Q3 + 1.5*IQR are considered anomalies.
    """

    def __init__(self, multiplier: float = 1.5, config: Optional[AnomalyDetectionConfig] = None):
        """
        Initialize IQR detector.

        Args:
            multiplier: IQR multiplier (default: 1.5)
            config: Optional detector configuration
        """
        if config is None:
            config = AnomalyDetectionConfig(normalize_features=False)
        super().__init__(config)
        self.multiplier = multiplier
        self.Q1_ = None
        self.Q3_ = None
        self.IQR_ = None

    def fit(
        self,
        data: Union[np.ndarray, pd.DataFrame, Dict[Tuple[int, int, int], Any]],
        labels: Optional[np.ndarray] = None,
    ) -> "IQRDetector":
        """
        Fit the detector on normal data.

        Args:
            data: Training data (normal data only)
            labels: Not used for IQR (unsupervised)

        Returns:
            Self for method chaining
        """
        array_data = self._preprocess_data(data)

        # Calculate quartiles
        self.Q1_ = np.percentile(array_data, 25, axis=0)
        self.Q3_ = np.percentile(array_data, 75, axis=0)
        self.IQR_ = self.Q3_ - self.Q1_

        self.is_fitted = True
        logger.info(f"IQRDetector fitted on {array_data.shape[0]} samples")
        return self

    def predict(self, data: Union[np.ndarray, pd.DataFrame, Dict[Tuple[int, int, int], Any]]) -> List[AnomalyDetectionResult]:
        """
        Detect anomalies using IQR method.

        Args:
            data: Data to detect anomalies in

        Returns:
            List of AnomalyDetectionResult objects
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")

        array_data = self._preprocess_data(data)

        # Calculate lower and upper bounds
        lower_bound = self.Q1_ - self.multiplier * self.IQR_
        upper_bound = self.Q3_ + self.multiplier * self.IQR_

        # Check if points are outside bounds
        outside_mask = (array_data < lower_bound) | (array_data > upper_bound)

        # Calculate anomaly scores (distance from bounds)
        distances = np.zeros(len(array_data))
        for i in range(len(array_data)):
            for j in range(array_data.shape[1]):
                if array_data[i, j] < lower_bound[j]:
                    distances[i] = max(distances[i], (lower_bound[j] - array_data[i, j]) / self.IQR_[j])
                elif array_data[i, j] > upper_bound[j]:
                    distances[i] = max(distances[i], (array_data[i, j] - upper_bound[j]) / self.IQR_[j])

        # Normalize scores
        max_distance = np.max(distances) if len(distances) > 0 and np.max(distances) > 0 else 1.0
        scores = distances / max_distance if max_distance > 0 else distances

        # Get indices
        indices = getattr(self, "_voxel_indices", None)
        if indices is None:
            indices = [(0, 0, 0)] * len(scores)

        coordinates = [(0.0, 0.0, 0.0)] * len(scores)

        # Create results
        results = self._create_results(scores=scores, indices=indices, coordinates=coordinates)

        return results
