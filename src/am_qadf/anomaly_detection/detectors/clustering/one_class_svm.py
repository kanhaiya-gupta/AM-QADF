"""
One-Class SVM Anomaly Detection

Support vector machine for novelty detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

try:
    from sklearn.svm import OneClassSVM

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ...core.base_detector import (
    BaseAnomalyDetector,
    AnomalyDetectionResult,
    AnomalyDetectionConfig,
)

logger = logging.getLogger(__name__)


class OneClassSVMDetector(BaseAnomalyDetector):
    """
    One-Class SVM based anomaly detector.

    Learns a decision function for novelty detection: classifying new data
    as similar or different to the training set.
    """

    def __init__(
        self,
        nu: float = 0.1,
        kernel: str = "rbf",
        gamma: str = "scale",
        config: Optional[AnomalyDetectionConfig] = None,
    ):
        """
        Initialize One-Class SVM detector.

        Args:
            nu: Upper bound on fraction of outliers
            kernel: Kernel type ('rbf', 'linear', 'poly', 'sigmoid')
            gamma: Kernel coefficient ('scale', 'auto', or float)
            config: Optional detector configuration
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for OneClassSVMDetector. Install with: pip install scikit-learn")

        if config is None:
            config = AnomalyDetectionConfig()
        super().__init__(config)
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.svm_ = None

    def fit(
        self,
        data: Union[np.ndarray, pd.DataFrame, Dict[Tuple[int, int, int], Any]],
        labels: Optional[np.ndarray] = None,
    ) -> "OneClassSVMDetector":
        """
        Fit the detector on normal data.

        Args:
            data: Training data (normal data only)
            labels: Not used for One-Class SVM (unsupervised)

        Returns:
            Self for method chaining
        """
        array_data = self._preprocess_data(data)

        # Fit One-Class SVM
        self.svm_ = OneClassSVM(nu=self.nu, kernel=self.kernel, gamma=self.gamma)
        self.svm_.fit(array_data)

        self.is_fitted = True
        logger.info(f"OneClassSVMDetector fitted on {array_data.shape[0]} samples")
        return self

    def predict(self, data: Union[np.ndarray, pd.DataFrame, Dict[Tuple[int, int, int], Any]]) -> List[AnomalyDetectionResult]:
        """
        Detect anomalies using One-Class SVM.

        Args:
            data: Data to detect anomalies in

        Returns:
            List of AnomalyDetectionResult objects
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")

        array_data = self._preprocess_data(data)

        # Get decision function scores (negative = anomaly)
        scores_raw = self.svm_.decision_function(array_data)

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
