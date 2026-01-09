"""
Local Outlier Factor (LOF) Anomaly Detection

Density-based local anomaly detection using k-nearest neighbors.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

try:
    from sklearn.neighbors import LocalOutlierFactor

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ...core.base_detector import (
    BaseAnomalyDetector,
    AnomalyDetectionResult,
    AnomalyDetectionConfig,
)

logger = logging.getLogger(__name__)


class LOFDetector(BaseAnomalyDetector):
    """
    Local Outlier Factor (LOF) based anomaly detector.

    Measures the local deviation of density of a given sample with respect
    to its neighbors. Points with significantly lower density than neighbors
    are considered anomalies.
    """

    def __init__(
        self,
        n_neighbors: int = 20,
        contamination: float = 0.1,
        config: Optional[AnomalyDetectionConfig] = None,
    ):
        """
        Initialize LOF detector.

        Args:
            n_neighbors: Number of neighbors to consider
            contamination: Expected proportion of anomalies
            config: Optional detector configuration
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for LOFDetector. Install with: pip install scikit-learn")

        if config is None:
            config = AnomalyDetectionConfig()
        super().__init__(config)
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.lof_ = None

    def fit(
        self,
        data: Union[np.ndarray, pd.DataFrame, Dict[Tuple[int, int, int], Any]],
        labels: Optional[np.ndarray] = None,
    ) -> "LOFDetector":
        """
        Fit the detector on normal data.

        Args:
            data: Training data (normal data only)
            labels: Not used for LOF (unsupervised)

        Returns:
            Self for method chaining
        """
        array_data = self._preprocess_data(data)

        # Adjust n_neighbors if needed
        n_neighbors = min(self.n_neighbors, len(array_data) - 1)

        # Fit LOF
        self.lof_ = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=self.contamination,
            novelty=True,  # Enable prediction on new data
        )
        self.lof_.fit(array_data)

        self.is_fitted = True
        logger.info(f"LOFDetector fitted on {array_data.shape[0]} samples")
        return self

    def predict(self, data: Union[np.ndarray, pd.DataFrame, Dict[Tuple[int, int, int], Any]]) -> List[AnomalyDetectionResult]:
        """
        Detect anomalies using LOF.

        Args:
            data: Data to detect anomalies in

        Returns:
            List of AnomalyDetectionResult objects
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")

        array_data = self._preprocess_data(data)

        # Get LOF scores (negative scores indicate anomalies)
        scores_raw = self.lof_.score_samples(array_data)

        # Convert to positive scores (higher = more anomalous)
        scores = -scores_raw

        # Normalize to [0, 1] range
        min_score = np.min(scores)
        max_score = np.max(scores)
        if max_score > min_score:
            scores = (scores - min_score) / (max_score - min_score)
        else:
            scores = np.zeros_like(scores)

        # Get indices
        indices = getattr(self, "_voxel_indices", None)
        if indices is None:
            indices = [(0, 0, 0)] * len(scores)

        coordinates = [(0.0, 0.0, 0.0)] * len(scores)

        # Create results
        results = self._create_results(scores=scores, indices=indices, coordinates=coordinates)

        return results
