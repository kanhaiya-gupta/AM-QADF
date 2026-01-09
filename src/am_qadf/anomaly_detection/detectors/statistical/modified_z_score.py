"""
Modified Z-Score Based Anomaly Detection

Detects anomalies using median absolute deviation (MAD), which is
more robust to outliers than standard deviation.
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


class ModifiedZScoreDetector(BaseAnomalyDetector):
    """
    Modified Z-Score based anomaly detector.

    Uses median and median absolute deviation (MAD) instead of mean and std,
    making it more robust to outliers.
    """

    def __init__(self, threshold: float = 3.5, config: Optional[AnomalyDetectionConfig] = None):
        """
        Initialize Modified Z-Score detector.

        Args:
            threshold: Modified Z-score threshold (default: 3.5)
            config: Optional detector configuration
        """
        if config is None:
            config = AnomalyDetectionConfig(threshold=threshold, normalize_features=False)
        else:
            config.threshold = threshold
            config.normalize_features = False

        super().__init__(config)
        self.threshold = threshold
        self.median_ = None
        self.mad_ = None

    def fit(
        self,
        data: Union[np.ndarray, pd.DataFrame, Dict[Tuple[int, int, int], Any]],
        labels: Optional[np.ndarray] = None,
    ) -> "ModifiedZScoreDetector":
        """
        Fit the detector on normal data.

        Args:
            data: Training data (normal data only)
            labels: Not used for Modified Z-score (unsupervised)

        Returns:
            Self for method chaining
        """
        array_data = self._preprocess_data(data)

        # Calculate median and MAD
        self.median_ = np.median(array_data, axis=0)

        # MAD = median(|x_i - median(x)|)
        deviations = np.abs(array_data - self.median_)
        self.mad_ = np.median(deviations, axis=0)
        self.mad_[self.mad_ == 0] = 1.0  # Avoid division by zero

        self.is_fitted = True
        logger.info(f"ModifiedZScoreDetector fitted on {array_data.shape[0]} samples")
        return self

    def predict(self, data: Union[np.ndarray, pd.DataFrame, Dict[Tuple[int, int, int], Any]]) -> List[AnomalyDetectionResult]:
        """
        Detect anomalies using Modified Z-score.

        Args:
            data: Data to detect anomalies in

        Returns:
            List of AnomalyDetectionResult objects
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")

        array_data = self._preprocess_data(data)

        # Calculate modified Z-scores
        modified_z_scores = np.abs(0.6745 * (array_data - self.median_) / self.mad_)

        # Maximum modified Z-score across features
        max_modified_z_scores = np.max(modified_z_scores, axis=1)

        # Get indices
        indices = getattr(self, "_voxel_indices", None)
        if indices is None:
            indices = [(0, 0, 0)] * len(max_modified_z_scores)

        coordinates = [(0.0, 0.0, 0.0)] * len(max_modified_z_scores)

        # Create results
        results = self._create_results(scores=max_modified_z_scores, indices=indices, coordinates=coordinates)

        return results
