"""
Z-Score Based Anomaly Detection

Detects anomalies using standard deviations from the mean.
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


class ZScoreDetector(BaseAnomalyDetector):
    """
    Z-Score based anomaly detector.

    Detects anomalies by calculating Z-scores (standard deviations from mean)
    and flagging points beyond a threshold.
    """

    def __init__(self, threshold: float = 3.0, config: Optional[AnomalyDetectionConfig] = None):
        """
        Initialize Z-Score detector.

        Args:
            threshold: Z-score threshold (default: 3.0, i.e., 3 standard deviations)
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
        self.std_ = None

    def fit(
        self,
        data: Union[np.ndarray, pd.DataFrame, Dict[Tuple[int, int, int], Any]],
        labels: Optional[np.ndarray] = None,
    ) -> "ZScoreDetector":
        """
        Fit the detector on normal data.

        Args:
            data: Training data (normal data only)
            labels: Not used for Z-score (unsupervised)

        Returns:
            Self for method chaining
        """
        array_data = self._preprocess_data(data)

        # Calculate mean and std
        self.mean_ = np.mean(array_data, axis=0)
        self.std_ = np.std(array_data, axis=0)
        self.std_[self.std_ == 0] = 1.0  # Avoid division by zero

        self.is_fitted = True
        logger.info(f"ZScoreDetector fitted on {array_data.shape[0]} samples")
        return self

    def predict(self, data: Union[np.ndarray, pd.DataFrame, Dict[Tuple[int, int, int], Any]]) -> List[AnomalyDetectionResult]:
        """
        Detect anomalies using Z-score.

        Args:
            data: Data to detect anomalies in

        Returns:
            List of AnomalyDetectionResult objects
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")

        array_data = self._preprocess_data(data)

        # Calculate Z-scores
        z_scores = np.abs((array_data - self.mean_) / self.std_)

        # Maximum Z-score across features (most anomalous feature)
        max_z_scores = np.max(z_scores, axis=1)

        # Get indices and coordinates if available
        indices = getattr(self, "_voxel_indices", None)
        if indices is None:
            indices = [(0, 0, 0)] * len(max_z_scores)

        coordinates = [(0.0, 0.0, 0.0)] * len(max_z_scores)  # Placeholder

        # Create results
        results = self._create_results(scores=max_z_scores, indices=indices, coordinates=coordinates)

        return results
