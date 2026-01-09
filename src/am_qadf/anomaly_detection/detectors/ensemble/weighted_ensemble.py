"""
Weighted Ensemble Anomaly Detection

Combines multiple detectors using weighted averaging of scores.
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


class WeightedEnsembleDetector(BaseAnomalyDetector):
    """
    Weighted ensemble detector.

    Combines multiple detectors using weighted averaging of anomaly scores.
    Weights can be uniform, performance-based, or manually specified.
    """

    def __init__(
        self,
        detectors: List[BaseAnomalyDetector],
        weights: Optional[List[float]] = None,
        normalize_weights: bool = True,
        config: Optional[AnomalyDetectionConfig] = None,
    ):
        """
        Initialize weighted ensemble detector.

        Args:
            detectors: List of detectors to combine
            weights: Optional weights for each detector (if None, uniform weights)
            normalize_weights: Whether to normalize weights to sum to 1
            config: Optional detector configuration
        """
        if config is None:
            config = AnomalyDetectionConfig()
        super().__init__(config)

        self.detectors = detectors

        if weights is None:
            # Uniform weights
            self.weights = np.ones(len(detectors)) / len(detectors)
        else:
            self.weights = np.array(weights)
            if normalize_weights:
                self.weights = self.weights / np.sum(self.weights)

        logger.info(f"WeightedEnsembleDetector initialized with {len(detectors)} detectors")

    def fit(
        self,
        data: Union[np.ndarray, pd.DataFrame, Dict[Tuple[int, int, int], Any]],
        labels: Optional[np.ndarray] = None,
    ) -> "WeightedEnsembleDetector":
        """
        Fit all detectors in the ensemble.

        Args:
            data: Training data
            labels: Optional labels (for supervised detectors)

        Returns:
            Self for method chaining
        """
        for i, detector in enumerate(self.detectors):
            logger.info(f"Fitting detector {i+1}/{len(self.detectors)}: {detector.name}")
            detector.fit(data, labels)

        self.is_fitted = True
        return self

    def set_weights_from_performance(self, performance_scores: List[float]):
        """
        Set weights based on performance scores (e.g., F1 scores from validation).

        Args:
            performance_scores: Performance scores for each detector
        """
        if len(performance_scores) != len(self.detectors):
            raise ValueError("Number of performance scores must match number of detectors")

        # Normalize performance scores to weights
        scores = np.array(performance_scores)
        scores = np.maximum(scores, 0)  # Ensure non-negative
        self.weights = scores / np.sum(scores) if np.sum(scores) > 0 else np.ones(len(scores)) / len(scores)

        logger.info(f"Updated weights based on performance: {self.weights}")

    def predict(self, data: Union[np.ndarray, pd.DataFrame, Dict[Tuple[int, int, int], Any]]) -> List[AnomalyDetectionResult]:
        """
        Detect anomalies using weighted ensemble.

        Args:
            data: Data to detect anomalies in

        Returns:
            List of AnomalyDetectionResult objects
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")

        # Get predictions from all detectors
        all_results = []
        for detector in self.detectors:
            results = detector.predict(data)
            all_results.append(results)

        # Combine using weighted average
        n_samples = len(all_results[0])
        ensemble_results = []

        for i in range(n_samples):
            # Collect scores from all detectors
            scores = np.array([results[i].anomaly_score for results in all_results])

            # Weighted average
            weighted_score = np.sum(self.weights * scores)

            # Determine if anomaly based on threshold
            is_anomaly = bool(weighted_score >= self.config.threshold)

            # Calculate confidence (could be based on score agreement)
            score_std = np.std(scores)
            confidence = 1.0 / (1.0 + score_std)  # Higher agreement = higher confidence

            # Use first result as template
            template = all_results[0][i]

            result = AnomalyDetectionResult(
                voxel_index=template.voxel_index,
                voxel_coordinates=template.voxel_coordinates,
                is_anomaly=is_anomaly,
                anomaly_score=weighted_score,
                confidence=confidence,
                detector_name="WeightedEnsemble",
                features=template.features,
                metadata={
                    "individual_scores": scores.tolist(),
                    "weights": self.weights.tolist(),
                    "n_detectors": len(self.detectors),
                },
            )
            ensemble_results.append(result)

        return ensemble_results
