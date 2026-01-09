"""
Grubbs' Test for Anomaly Detection

Statistical test for detecting a single outlier in a univariate dataset.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from scipy import stats

from ...core.base_detector import (
    BaseAnomalyDetector,
    AnomalyDetectionResult,
    AnomalyDetectionConfig,
)

logger = logging.getLogger(__name__)


class GrubbsDetector(BaseAnomalyDetector):
    """
    Grubbs' test based anomaly detector.

    Detects a single outlier using Grubbs' test statistic. For multivariate
    data, applies the test to each feature and combines results.
    """

    def __init__(self, alpha: float = 0.05, config: Optional[AnomalyDetectionConfig] = None):
        """
        Initialize Grubbs' detector.

        Args:
            alpha: Significance level (default: 0.05)
            config: Optional detector configuration
        """
        if config is None:
            config = AnomalyDetectionConfig()
        super().__init__(config)
        self.alpha = alpha
        self.mean_ = None
        self.std_ = None

    def fit(
        self,
        data: Union[np.ndarray, pd.DataFrame, Dict[Tuple[int, int, int], Any]],
        labels: Optional[np.ndarray] = None,
    ) -> "GrubbsDetector":
        """
        Fit the detector on normal data.

        Args:
            data: Training data (normal data only)
            labels: Not used for Grubbs' test (unsupervised)

        Returns:
            Self for method chaining
        """
        array_data = self._preprocess_data(data)

        # Calculate mean and std
        self.mean_ = np.mean(array_data, axis=0)
        self.std_ = np.std(array_data, axis=0)
        self.std_[self.std_ == 0] = 1.0

        self.is_fitted = True
        logger.info(f"GrubbsDetector fitted on {array_data.shape[0]} samples")
        return self

    def _grubbs_statistic(self, data: np.ndarray) -> Tuple[float, int]:
        """
        Calculate Grubbs' test statistic.

        Args:
            data: Univariate data array

        Returns:
            Tuple of (G statistic, index of potential outlier)
        """
        n = len(data)
        if n < 3:
            return 0.0, -1

        mean = np.mean(data)
        std = np.std(data)

        if std == 0:
            return 0.0, -1

        # Find the point furthest from mean
        deviations = np.abs(data - mean)
        max_idx = np.argmax(deviations)
        max_deviation = deviations[max_idx]

        # Grubbs' statistic
        G = max_deviation / std

        return G, max_idx

    def _grubbs_critical_value(self, n: int, alpha: float) -> float:
        """
        Calculate critical value for Grubbs' test.

        Args:
            n: Sample size
            alpha: Significance level

        Returns:
            Critical value
        """
        # Approximate critical value using t-distribution
        t_critical = stats.t.ppf(1 - alpha / (2 * n), n - 2)
        G_critical = (n - 1) / np.sqrt(n) * np.sqrt(t_critical**2 / (n - 2 + t_critical**2))
        return G_critical

    def predict(self, data: Union[np.ndarray, pd.DataFrame, Dict[Tuple[int, int, int], Any]]) -> List[AnomalyDetectionResult]:
        """
        Detect anomalies using Grubbs' test.

        Args:
            data: Data to detect anomalies in

        Returns:
            List of AnomalyDetectionResult objects
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")

        array_data = self._preprocess_data(data)
        n_samples = len(array_data)

        # Calculate Grubbs' statistics for each feature
        grubbs_scores = np.zeros(n_samples)

        for feature_idx in range(array_data.shape[1]):
            feature_data = array_data[:, feature_idx]

            # Calculate Grubbs' statistic
            G, max_idx = self._grubbs_statistic(feature_data)

            # Calculate critical value
            G_critical = self._grubbs_critical_value(n_samples, self.alpha)

            # If G > G_critical, mark as anomaly
            if G > G_critical and max_idx >= 0:
                # Score based on how much G exceeds critical value
                grubbs_scores[max_idx] = max(grubbs_scores[max_idx], (G - G_critical) / G_critical)

        # Normalize scores
        max_score = np.max(grubbs_scores) if np.max(grubbs_scores) > 0 else 1.0
        scores = grubbs_scores / max_score if max_score > 0 else grubbs_scores

        # Get indices
        indices = getattr(self, "_voxel_indices", None)
        if indices is None:
            indices = [(0, 0, 0)] * len(scores)

        coordinates = [(0.0, 0.0, 0.0)] * len(scores)

        # Create results
        results = self._create_results(scores=scores, indices=indices, coordinates=coordinates)

        return results
