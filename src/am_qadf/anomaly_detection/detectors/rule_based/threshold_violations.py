"""
Threshold Violation Based Anomaly Detection

Detects anomalies by checking if process parameters exceed predefined thresholds.
Uses domain knowledge and sensitivity analysis to define normal parameter ranges.
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


class ThresholdViolationDetector(BaseAnomalyDetector):
    """
    Threshold-based anomaly detector.

    Detects anomalies by checking if process parameters exceed predefined thresholds.
    Can use absolute thresholds or percentile-based thresholds.
    """

    def __init__(
        self,
        thresholds: Optional[Dict[str, Tuple[float, float]]] = None,
        threshold_percentile: float = 95.0,
        use_absolute: bool = False,
        strict_mode: bool = False,
        config: Optional[AnomalyDetectionConfig] = None,
    ):
        """
        Initialize threshold violation detector.

        Args:
            thresholds: Dictionary mapping parameter names to (min, max) thresholds.
                       If None, thresholds are learned from training data.
            threshold_percentile: Percentile to use for threshold calculation (default: 95th)
            use_absolute: If True, use absolute thresholds. If False, use percentile-based.
            strict_mode: If True, any violation is an anomaly. If False, use severity scoring.
            config: Optional detector configuration
        """
        if config is None:
            config = AnomalyDetectionConfig(normalize_features=False)
        super().__init__(config)

        self.thresholds = thresholds or {}
        self.threshold_percentile = threshold_percentile
        self.use_absolute = use_absolute
        self.strict_mode = strict_mode

        self.learned_thresholds_ = None
        self.feature_names_ = None
        self.learned_violation_threshold_ = None

    def fit(
        self,
        data: Union[np.ndarray, pd.DataFrame, Dict[Tuple[int, int, int], Any]],
        labels: Optional[np.ndarray] = None,
    ) -> "ThresholdViolationDetector":
        """
        Fit the detector by learning thresholds from normal data.

        Args:
            data: Training data (normal data only)
            labels: Not used for threshold-based detection

        Returns:
            Self for method chaining
        """
        array_data = self._preprocess_data(data)
        self.feature_names_ = getattr(self, "_feature_names", None)

        if not self.use_absolute and not self.thresholds:
            # Learn thresholds from data
            self.learned_thresholds_ = {}

            for i in range(array_data.shape[1]):
                feature_name = self.feature_names_[i] if self.feature_names_ else f"feature_{i}"
                values = array_data[:, i]

                # Remove NaN/inf values
                valid_values = values[np.isfinite(values)]

                if len(valid_values) > 0:
                    if self.strict_mode:
                        # Use tighter bounds (e.g., 3-sigma rule)
                        mean_val = np.mean(valid_values)
                        std_val = np.std(valid_values)
                        min_thresh = mean_val - 3 * std_val
                        max_thresh = mean_val + 3 * std_val
                    else:
                        # Use percentile-based thresholds
                        # Min threshold is low percentile, max threshold is high percentile
                        # With threshold_percentile=95, min should be 5th percentile, max should be 95th percentile
                        min_thresh = np.percentile(valid_values, 100 - self.threshold_percentile)
                        max_thresh = np.percentile(valid_values, self.threshold_percentile)

                    self.learned_thresholds_[feature_name] = (min_thresh, max_thresh)
                else:
                    logger.warning(f"All values for {feature_name} are NaN/inf, skipping threshold learning")
        else:
            # If using predefined thresholds, initialize learned_thresholds_ as empty dict
            # The thresholds will be used from self.thresholds in predict
            self.learned_thresholds_ = {}

        # Calculate baseline violation threshold from training data (for non-strict mode)
        thresholds_to_use = self.thresholds if self.thresholds else self.learned_thresholds_
        if not self.strict_mode and len(thresholds_to_use) > 0:
            training_violation_scores = np.zeros(array_data.shape[0])
            feature_names = (
                self.feature_names_ if self.feature_names_ else [f"feature_{i}" for i in range(array_data.shape[1])]
            )

            for i, feature_name in enumerate(feature_names):
                if feature_name in thresholds_to_use:
                    min_thresh, max_thresh = thresholds_to_use[feature_name]
                    values = array_data[:, i]

                    below_min = np.where(
                        values < min_thresh,
                        (min_thresh - values) / (abs(min_thresh) + 1e-10),
                        0,
                    )
                    above_max = np.where(
                        values > max_thresh,
                        (values - max_thresh) / (abs(max_thresh) + 1e-10),
                        0,
                    )
                    violations = below_min + above_max
                    training_violation_scores += violations

            training_violation_scores = training_violation_scores / len(thresholds_to_use)
            if np.any(np.isfinite(training_violation_scores)):
                # Use a small threshold since training data should have no violations
                # Use max of 95th percentile or a small absolute value (0.01) to catch real violations
                percentile_threshold = np.percentile(
                    training_violation_scores[np.isfinite(training_violation_scores)],
                    self.threshold_percentile,
                )
                self.learned_violation_threshold_ = max(percentile_threshold, 0.01)
            else:
                self.learned_violation_threshold_ = 0.01

        self.is_fitted = True
        logger.info(f"ThresholdViolationDetector fitted on {array_data.shape[0]} samples")
        return self

    def predict(self, data: Union[np.ndarray, pd.DataFrame, Dict[Tuple[int, int, int], Any]]) -> List[AnomalyDetectionResult]:
        """
        Detect anomalies based on threshold violations.

        Args:
            data: Data to detect anomalies in

        Returns:
            List of AnomalyDetectionResult objects
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")

        array_data = self._preprocess_data(data)

        # Use learned thresholds or provided thresholds
        thresholds_to_use = self.thresholds if self.thresholds else self.learned_thresholds_

        if not thresholds_to_use:
            raise ValueError("No thresholds available. Fit the detector or provide thresholds.")

        # Calculate violation scores
        violation_scores = np.zeros(array_data.shape[0])
        feature_names = self.feature_names_ if self.feature_names_ else [f"feature_{i}" for i in range(array_data.shape[1])]

        # Track if any feature violates threshold (for strict mode)
        has_violation = np.zeros(array_data.shape[0], dtype=bool)

        for i, feature_name in enumerate(feature_names):
            if feature_name in thresholds_to_use:
                min_thresh, max_thresh = thresholds_to_use[feature_name]
                values = array_data[:, i]

                # Check for violations (binary check for strict mode)
                below_min_mask = values < min_thresh
                above_max_mask = values > max_thresh
                has_violation = has_violation | below_min_mask | above_max_mask

                # Calculate violation severity (for scoring)
                # Use normalized deviation from threshold
                std_val = np.std(values[np.isfinite(values)])
                if std_val > 0:
                    below_min = np.where(below_min_mask, (min_thresh - values) / std_val, 0)
                    above_max = np.where(above_max_mask, (values - max_thresh) / std_val, 0)
                else:
                    # Fallback if std is 0
                    below_min = np.where(
                        below_min_mask,
                        (min_thresh - values) / (abs(min_thresh) + 1e-10),
                        0,
                    )
                    above_max = np.where(
                        above_max_mask,
                        (values - max_thresh) / (abs(max_thresh) + 1e-10),
                        0,
                    )

                # Combine violations (use max for strict mode, sum for severity mode)
                if self.strict_mode:
                    violations = np.maximum(below_min, above_max)
                else:
                    violations = below_min + above_max

                violation_scores += violations

        # Normalize scores
        if len(thresholds_to_use) > 0:
            violation_scores = violation_scores / len(thresholds_to_use)

        # Determine anomalies
        if self.strict_mode:
            # Any violation is an anomaly (check if any feature violates its threshold)
            is_anomaly = has_violation
        else:
            # Use learned threshold from training data
            if self.learned_violation_threshold_ is not None:
                threshold = self.learned_violation_threshold_
            elif self.use_absolute:
                threshold = 0.1  # Default severity threshold
            else:
                # Fallback: use a small threshold to catch violations
                threshold = 0.01
            is_anomaly = violation_scores > threshold

        # Get indices and coordinates if available
        indices = getattr(self, "_voxel_indices", None)
        if indices is None:
            indices = [(0, 0, 0)] * len(violation_scores)

        coordinates = [(0.0, 0.0, 0.0)] * len(violation_scores)  # Placeholder

        # Create results
        results = self._create_results(scores=violation_scores, indices=indices, coordinates=coordinates)

        return results
