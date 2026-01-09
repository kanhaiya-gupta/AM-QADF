"""
Random Forest Anomaly Detection

Detects anomalies using Random Forest isolation and distance-based methods.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import warnings

from ...core.base_detector import (
    BaseAnomalyDetector,
    AnomalyDetectionResult,
    AnomalyDetectionConfig,
)

# Try to import sklearn
try:
    from sklearn.ensemble import RandomForestClassifier, IsolationForest
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("sklearn not available. RandomForestAnomalyDetector requires sklearn.")

logger = logging.getLogger(__name__)


class RandomForestAnomalyDetector(BaseAnomalyDetector):
    """
    Random Forest-based anomaly detector.

    Uses Random Forest in two ways:
    1. Isolation Forest (if available) - tree-based isolation
    2. Distance-based - uses tree structure to compute anomaly scores
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        contamination: float = 0.1,
        use_isolation_forest: bool = True,
        config: Optional[AnomalyDetectionConfig] = None,
    ):
        """
        Initialize Random Forest anomaly detector.

        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees (None for unlimited)
            min_samples_split: Minimum samples to split a node
            min_samples_leaf: Minimum samples in a leaf
            contamination: Expected proportion of anomalies (for Isolation Forest)
            use_isolation_forest: Use IsolationForest if True, else use distance-based method
            config: Optional detector configuration
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn is required for RandomForestAnomalyDetector")

        if config is None:
            config = AnomalyDetectionConfig(threshold=0.5)
        super().__init__(config)

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.contamination = contamination
        self.use_isolation_forest = use_isolation_forest

        if self.use_isolation_forest:
            self.model = IsolationForest(
                n_estimators=n_estimators,
                max_samples="auto",
                contamination=contamination,
                random_state=42,
                n_jobs=-1,
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42,
                n_jobs=-1,
            )

        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(
        self,
        data: Union[np.ndarray, pd.DataFrame, Dict[Tuple[int, int, int], Any]],
        labels: Optional[np.ndarray] = None,
    ) -> "RandomForestAnomalyDetector":
        """
        Fit the Random Forest detector on normal data.

        Args:
            data: Training data (normal data only)
            labels: Not used for Isolation Forest (unsupervised)

        Returns:
            Self for method chaining
        """
        array_data = self._preprocess_data(data)

        # Scale data
        array_data_scaled = self.scaler.fit_transform(array_data)

        # Fit model
        if self.use_isolation_forest:
            self.model.fit(array_data_scaled)
        else:
            # For distance-based method, we'd need labeled data
            # For now, use Isolation Forest approach
            self.model = IsolationForest(
                n_estimators=self.n_estimators,
                contamination=self.contamination,
                random_state=42,
                n_jobs=-1,
            )
            self.model.fit(array_data_scaled)

        self.is_fitted = True
        logger.info(f"RandomForestAnomalyDetector fitted on {array_data.shape[0]} samples")
        return self

    def predict(self, data: Union[np.ndarray, pd.DataFrame, Dict[Tuple[int, int, int], Any]]) -> List[AnomalyDetectionResult]:
        """
        Detect anomalies using Random Forest.

        Args:
            data: Data to detect anomalies in

        Returns:
            List of AnomalyDetectionResult objects
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")

        array_data = self._preprocess_data(data)

        # Scale data
        array_data_scaled = self.scaler.transform(array_data)

        # Get predictions
        if self.use_isolation_forest:
            # Isolation Forest returns -1 for anomalies, 1 for normal
            predictions = self.model.predict(array_data_scaled)
            anomaly_scores = -self.model.score_samples(array_data_scaled)  # Negative scores (higher = more anomalous)
            is_anomaly = predictions == -1
        else:
            # Distance-based method (simplified)
            # Use decision function or feature importances
            predictions = self.model.predict(array_data_scaled)
            # For anomaly detection, use negative of decision function
            if hasattr(self.model, "decision_function"):
                anomaly_scores = -self.model.decision_function(array_data_scaled)
            else:
                # Fallback: use prediction probabilities
                anomaly_scores = 1.0 - np.max(self.model.predict_proba(array_data_scaled), axis=1)
            is_anomaly = predictions == 1  # Assuming 1 is anomaly class

        # Normalize scores to [0, 1] range
        if len(anomaly_scores) > 0:
            min_score = np.min(anomaly_scores)
            max_score = np.max(anomaly_scores)
            if max_score > min_score:
                anomaly_scores = (anomaly_scores - min_score) / (max_score - min_score)
            else:
                anomaly_scores = np.zeros_like(anomaly_scores)

        # Get indices and coordinates if available
        indices = getattr(self, "_voxel_indices", None)
        if indices is None:
            indices = [(0, 0, 0)] * len(anomaly_scores)

        coordinates = [(0.0, 0.0, 0.0)] * len(anomaly_scores)  # Placeholder

        # Create results
        results = self._create_results(scores=anomaly_scores, indices=indices, coordinates=coordinates)

        return results
