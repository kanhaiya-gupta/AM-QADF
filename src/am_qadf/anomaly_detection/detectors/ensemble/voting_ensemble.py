"""
Voting-Based Ensemble Anomaly Detection

Combines multiple detectors using voting mechanisms (majority, unanimous, etc.).
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


class VotingEnsembleDetector(BaseAnomalyDetector):
    """
    Voting-based ensemble detector.

    Combines multiple detectors using voting mechanisms:
    - Majority voting: Anomaly if majority of detectors agree
    - Unanimous voting: Anomaly only if all detectors agree
    - At least N: Anomaly if at least N detectors agree
    """

    def __init__(
        self,
        detectors: List[BaseAnomalyDetector],
        voting_method: str = "majority",
        threshold: int = None,
        config: Optional[AnomalyDetectionConfig] = None,
    ):
        """
        Initialize voting ensemble detector.

        Args:
            detectors: List of detectors to combine
            voting_method: 'majority', 'unanimous', or 'at_least_n'
            threshold: For 'at_least_n', minimum number of detectors that must agree
            config: Optional detector configuration
        """
        if config is None:
            config = AnomalyDetectionConfig()
        super().__init__(config)

        if not detectors or len(detectors) == 0:
            raise ValueError("VotingEnsembleDetector requires at least one detector")

        self.detectors = detectors
        self.voting_method = voting_method

        if voting_method == "at_least_n":
            if threshold is None:
                threshold = len(detectors) // 2 + 1
            self.threshold = threshold
        else:
            self.threshold = None

        logger.info(f"VotingEnsembleDetector initialized with {len(detectors)} detectors, method: {voting_method}")

    def fit(
        self,
        data: Union[np.ndarray, pd.DataFrame, Dict[Tuple[int, int, int], Any]],
        labels: Optional[np.ndarray] = None,
    ) -> "VotingEnsembleDetector":
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

    def predict(self, data: Union[np.ndarray, pd.DataFrame, Dict[Tuple[int, int, int], Any]]) -> List[AnomalyDetectionResult]:
        """
        Detect anomalies using voting ensemble.

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

        # Combine using voting
        n_samples = len(all_results[0])
        ensemble_results = []

        for i in range(n_samples):
            # Collect votes from all detectors
            votes = [results[i].is_anomaly for results in all_results]
            scores = [results[i].anomaly_score for results in all_results]

            # Determine ensemble decision
            if self.voting_method == "majority":
                is_anomaly = sum(votes) > len(votes) / 2
            elif self.voting_method == "unanimous":
                is_anomaly = all(votes)
            elif self.voting_method == "at_least_n":
                is_anomaly = sum(votes) >= self.threshold
            else:
                is_anomaly = sum(votes) > len(votes) / 2  # Default to majority

            # Average score
            avg_score = np.mean(scores)

            # Use first result as template
            template = all_results[0][i]

            result = AnomalyDetectionResult(
                voxel_index=template.voxel_index,
                voxel_coordinates=template.voxel_coordinates,
                is_anomaly=is_anomaly,
                anomaly_score=avg_score,
                confidence=template.confidence,  # Could calculate ensemble confidence
                detector_name=f"VotingEnsemble({self.voting_method})",
                features=template.features,
                metadata={
                    "votes": votes,
                    "voting_method": self.voting_method,
                    "n_detectors": len(self.detectors),
                },
            )
            ensemble_results.append(result)

        return ensemble_results
