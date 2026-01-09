"""
Isolation Forest Anomaly Detection

Tree-based method for isolating anomalies using random forests.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

try:
    from sklearn.ensemble import IsolationForest

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ...core.base_detector import (
    BaseAnomalyDetector,
    AnomalyDetectionResult,
    AnomalyDetectionConfig,
)

logger = logging.getLogger(__name__)


class IsolationForestDetector(BaseAnomalyDetector):
    """
    Isolation Forest based anomaly detector.

    Uses random forests to isolate anomalies. Anomalies are easier to isolate
    and require fewer splits.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.1,
        random_state: Optional[int] = None,
        config: Optional[AnomalyDetectionConfig] = None,
    ):
        """
        Initialize Isolation Forest detector.

        Args:
            n_estimators: Number of trees in the forest
            contamination: Expected proportion of anomalies
            random_state: Random seed
            config: Optional detector configuration
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for IsolationForestDetector. Install with: pip install scikit-learn")

        if config is None:
            config = AnomalyDetectionConfig()
        super().__init__(config)
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.forest_ = None

    def fit(
        self,
        data: Union[np.ndarray, pd.DataFrame, Dict[Tuple[int, int, int], Any]],
        labels: Optional[np.ndarray] = None,
    ) -> "IsolationForestDetector":
        """
        Fit the detector on normal data.

        Args:
            data: Training data (normal data only)
            labels: Not used for Isolation Forest (unsupervised)

        Returns:
            Self for method chaining
        """
        array_data = self._preprocess_data(data)

        # Fit Isolation Forest
        self.forest_ = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        self.forest_.fit(array_data)

        self.is_fitted = True
        logger.info(f"IsolationForestDetector fitted on {array_data.shape[0]} samples")
        return self

    def predict(self, data: Union[np.ndarray, pd.DataFrame, Dict[Tuple[int, int, int], Any]]) -> List[AnomalyDetectionResult]:
        """
        Detect anomalies using Isolation Forest.

        Args:
            data: Data to detect anomalies in

        Returns:
            List of AnomalyDetectionResult objects
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")

        array_data = self._preprocess_data(data)

        # Get anomaly scores (negative scores indicate anomalies)
        scores_raw = self.forest_.score_samples(array_data)

        # Convert to positive scores (higher = more anomalous)
        # Isolation Forest returns negative scores for anomalies
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
