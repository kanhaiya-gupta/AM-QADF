"""
Multi-Signal Correlation Anomaly Detection

Detects anomalies by checking cross-signal consistency and correlation patterns.
Identifies when signals that should be correlated deviate from expected relationships.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from scipy.stats import pearsonr

from ...core.base_detector import (
    BaseAnomalyDetector,
    AnomalyDetectionResult,
    AnomalyDetectionConfig,
)

logger = logging.getLogger(__name__)


class MultiSignalCorrelationDetector(BaseAnomalyDetector):
    """
    Multi-signal correlation anomaly detector.

    Detects anomalies by checking cross-signal consistency and correlation patterns.
    Identifies when signals that should be correlated deviate from expected relationships.
    """

    def __init__(
        self,
        correlation_threshold: float = 0.7,
        use_expected_correlations: bool = True,
        use_residual_analysis: bool = True,
        use_cross_validation: bool = True,
        expected_correlations: Optional[Dict[Tuple[str, str], float]] = None,
        threshold_percentile: float = 95.0,
        config: Optional[AnomalyDetectionConfig] = None,
    ):
        """
        Initialize multi-signal correlation detector.

        Args:
            correlation_threshold: Minimum correlation to consider signals related (default: 0.7)
            use_expected_correlations: Use provided expected correlations
            use_residual_analysis: Analyze residuals from expected relationships
            use_cross_validation: Cross-validate signal relationships
            expected_correlations: Dictionary mapping (signal1, signal2) to expected correlation
            config: Optional detector configuration
        """
        if config is None:
            # Disable normalization for correlation-based detection
            # Normalization can distort correlation relationships
            config = AnomalyDetectionConfig(normalize_features=False)
        super().__init__(config)

        self.correlation_threshold = correlation_threshold
        self.use_expected_correlations = use_expected_correlations
        self.use_residual_analysis = use_residual_analysis
        self.use_cross_validation = use_cross_validation
        self.expected_correlations = expected_correlations or {}
        self.threshold_percentile = threshold_percentile

        self.learned_correlations_ = None
        self.relationship_models_ = None
        self.feature_names_ = None
        self.baseline_threshold_ = None

    def fit(
        self,
        data: Union[np.ndarray, pd.DataFrame, Dict[Tuple[int, int, int], Any]],
        labels: Optional[np.ndarray] = None,
    ) -> "MultiSignalCorrelationDetector":
        """
        Fit the detector by learning signal correlations from normal data.

        Args:
            data: Training data (normal data only)
            labels: Not used for correlation-based detection

        Returns:
            Self for method chaining
        """
        array_data = self._preprocess_data(data)
        self.feature_names_ = getattr(self, "_feature_names", None)

        if self.feature_names_ is None:
            self.feature_names_ = [f"feature_{i}" for i in range(array_data.shape[1])]

        # Learn correlations between signals
        self.learned_correlations_ = {}
        self.relationship_models_ = {}

        n_features = array_data.shape[1]

        for i in range(n_features):
            for j in range(i + 1, n_features):
                signal1_name = self.feature_names_[i]
                signal2_name = self.feature_names_[j]

                values1 = array_data[:, i]
                values2 = array_data[:, j]

                # Remove NaN/inf values
                valid_mask = np.isfinite(values1) & np.isfinite(values2)
                valid_values1 = values1[valid_mask]
                valid_values2 = values2[valid_mask]

                if len(valid_values1) >= 3:  # Need at least 3 points for correlation
                    # Calculate correlation
                    try:
                        correlation, p_value = pearsonr(valid_values1, valid_values2)

                        if not np.isnan(correlation) and abs(correlation) >= self.correlation_threshold:
                            self.learned_correlations_[(signal1_name, signal2_name)] = {
                                "correlation": correlation,
                                "p_value": p_value,
                                "valid_count": len(valid_values1),
                            }

                            # Learn relationship model (linear regression)
                            if self.use_residual_analysis:
                                # Simple linear model: signal2 = a * signal1 + b
                                coeffs = np.polyfit(valid_values1, valid_values2, 1)
                                # Store training std for consistent normalization during prediction
                                training_std = np.std(valid_values2)
                                self.relationship_models_[(signal1_name, signal2_name)] = {
                                    "slope": coeffs[0],
                                    "intercept": coeffs[1],
                                    "r_squared": correlation**2,
                                    "training_std": training_std,
                                }
                    except Exception as e:
                        logger.warning(f"Error calculating correlation between {signal1_name} and {signal2_name}: {e}")

        # Calculate baseline threshold from training data
        if len(self.learned_correlations_) > 0:
            # Calculate scores on training data to establish baseline threshold
            training_scores = np.zeros(array_data.shape[0])
            feature_names = (
                self.feature_names_ if self.feature_names_ else [f"feature_{i}" for i in range(array_data.shape[1])]
            )

            for (
                signal1_name,
                signal2_name,
            ), correlation_info in self.learned_correlations_.items():
                try:
                    idx1 = feature_names.index(signal1_name)
                    idx2 = feature_names.index(signal2_name)
                except ValueError:
                    continue

                values1 = array_data[:, idx1]
                values2 = array_data[:, idx2]

                for i in range(len(values1)):
                    if np.isfinite(values1[i]) and np.isfinite(values2[i]):
                        if self.use_residual_analysis and (signal1_name, signal2_name) in self.relationship_models_:
                            model = self.relationship_models_[(signal1_name, signal2_name)]
                            residual = self._calculate_residual(values1[i], values2[i], model)
                            training_scores[i] += residual

                        if self.use_cross_validation:
                            if (
                                signal1_name,
                                signal2_name,
                            ) in self.relationship_models_:
                                model = self.relationship_models_[(signal1_name, signal2_name)]
                                expected_value2 = model["slope"] * values1[i] + model["intercept"]
                                # Use training std for consistency
                                training_std = model.get(
                                    "training_std",
                                    np.std(values2[np.isfinite(values2)]),
                                )
                                deviation = abs(values2[i] - expected_value2) / (training_std + 1e-10)
                                training_scores[i] += 0.5 * deviation

            training_scores = training_scores / len(self.learned_correlations_)
            if np.any(np.isfinite(training_scores)):
                # Use percentile-based threshold to ensure <20% false positives
                finite_scores = training_scores[np.isfinite(training_scores)]
                # Use 85th percentile to ensure <15% false positives (well below 20% requirement)
                # This means only the top 15% of training scores will be above the threshold
                percentile_85 = np.percentile(finite_scores, 85.0)
                # Add a small safety margin (5%) to handle any numerical precision differences
                # between training and prediction score calculations
                self.baseline_threshold_ = percentile_85 * 1.05
            else:
                self.baseline_threshold_ = 0.1
        else:
            self.baseline_threshold_ = 0.1

        self.is_fitted = True
        logger.info(
            f"MultiSignalCorrelationDetector fitted on {array_data.shape[0]} samples. "
            f"Found {len(self.learned_correlations_)} significant correlations."
        )
        return self

    def _calculate_residual(self, value1: float, value2: float, model: Dict) -> float:
        """Calculate residual from expected relationship."""
        if not (np.isfinite(value1) and np.isfinite(value2)):
            return 0.0

        expected_value2 = model["slope"] * value1 + model["intercept"]
        residual = abs(value2 - expected_value2)

        # Normalize by expected value range
        normalized_residual = residual / (abs(expected_value2) + 1e-10)
        return normalized_residual

    def predict(self, data: Union[np.ndarray, pd.DataFrame, Dict[Tuple[int, int, int], Any]]) -> List[AnomalyDetectionResult]:
        """
        Detect anomalies based on multi-signal correlation violations.

        Args:
            data: Data to detect anomalies in

        Returns:
            List of AnomalyDetectionResult objects
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")

        array_data = self._preprocess_data(data)
        feature_names = self.feature_names_ if self.feature_names_ else [f"feature_{i}" for i in range(array_data.shape[1])]

        # Calculate correlation violation scores
        correlation_scores = np.zeros(array_data.shape[0])

        # Check each learned correlation
        for (
            signal1_name,
            signal2_name,
        ), correlation_info in self.learned_correlations_.items():
            try:
                idx1 = feature_names.index(signal1_name)
                idx2 = feature_names.index(signal2_name)
            except ValueError:
                continue

            values1 = array_data[:, idx1]
            values2 = array_data[:, idx2]

            # Calculate correlation violation for each sample
            for i in range(len(values1)):
                if np.isfinite(values1[i]) and np.isfinite(values2[i]):
                    # Residual analysis
                    if self.use_residual_analysis and (signal1_name, signal2_name) in self.relationship_models_:
                        model = self.relationship_models_[(signal1_name, signal2_name)]
                        residual = self._calculate_residual(values1[i], values2[i], model)
                        correlation_scores[i] += residual

                    # Cross-validation: check if this point maintains correlation
                    if self.use_cross_validation:
                        # Calculate deviation from expected relationship
                        if (signal1_name, signal2_name) in self.relationship_models_:
                            model = self.relationship_models_[(signal1_name, signal2_name)]
                            expected_value2 = model["slope"] * values1[i] + model["intercept"]
                            # Use training std for consistency with training score calculation
                            training_std = model.get("training_std", np.std(values2[np.isfinite(values2)]))
                            deviation = abs(values2[i] - expected_value2) / (training_std + 1e-10)
                            correlation_scores[i] += 0.5 * deviation

        # Normalize scores
        if len(self.learned_correlations_) > 0:
            correlation_scores = correlation_scores / len(self.learned_correlations_)

        # Determine anomalies using baseline threshold from training
        threshold = (
            self.baseline_threshold_
            if self.baseline_threshold_ is not None
            else (
                np.percentile(
                    correlation_scores[np.isfinite(correlation_scores)],
                    self.threshold_percentile,
                )
                if np.any(np.isfinite(correlation_scores))
                else 0.1
            )
        )
        is_anomaly = correlation_scores > threshold

        # Get indices and coordinates if available
        indices = getattr(self, "_voxel_indices", None)
        if indices is None:
            indices = [(0, 0, 0)] * len(correlation_scores)

        coordinates = [(0.0, 0.0, 0.0)] * len(correlation_scores)  # Placeholder

        # Temporarily update config threshold so _create_results uses the correct threshold
        original_threshold = self.config.threshold
        self.config.threshold = threshold

        try:
            # Create results using our custom threshold
            results = self._create_results(scores=correlation_scores, indices=indices, coordinates=coordinates)
        finally:
            # Restore original threshold
            self.config.threshold = original_threshold

        return results
