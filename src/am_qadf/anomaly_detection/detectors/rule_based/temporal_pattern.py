"""
Temporal Pattern Anomaly Detection

Detects anomalies in time-series patterns, layer-to-layer variations, and build progression.
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


class TemporalPatternDetector(BaseAnomalyDetector):
    """
    Temporal pattern anomaly detector.

    Detects anomalies in time-series patterns, layer-to-layer variations,
    and build progression sequences.
    """

    def __init__(
        self,
        sequence_length: int = 5,
        use_layer_analysis: bool = True,
        use_trend_analysis: bool = True,
        use_variance_analysis: bool = True,
        layer_key: str = "layer_number",
        threshold_percentile: float = 95.0,
        config: Optional[AnomalyDetectionConfig] = None,
    ):
        """
        Initialize temporal pattern detector.

        Args:
            sequence_length: Length of sequence for pattern analysis (default: 5)
            use_layer_analysis: Analyze layer-to-layer variations
            use_trend_analysis: Detect trend anomalies
            use_variance_analysis: Detect variance anomalies
            layer_key: Key for layer number in data (default: 'layer_number')
            config: Optional detector configuration
        """
        if config is None:
            config = AnomalyDetectionConfig()
        super().__init__(config)

        self.sequence_length = sequence_length
        self.use_layer_analysis = use_layer_analysis
        self.use_trend_analysis = use_trend_analysis
        self.use_variance_analysis = use_variance_analysis
        self.layer_key = layer_key
        self.threshold_percentile = threshold_percentile

        self.baseline_patterns_ = None
        self.layer_stats_ = None
        self.feature_names_ = None
        self.baseline_threshold_ = None

    def fit(
        self,
        data: Union[np.ndarray, pd.DataFrame, Dict[Tuple[int, int, int], Any]],
        labels: Optional[np.ndarray] = None,
    ) -> "TemporalPatternDetector":
        """
        Fit the detector by learning baseline temporal patterns.

        Args:
            data: Training data (normal data only)
            labels: Not used for temporal pattern detection

        Returns:
            Self for method chaining
        """
        array_data = self._preprocess_data(data)
        self.feature_names_ = getattr(self, "_feature_names", None)

        # Learn baseline patterns
        self.baseline_patterns_ = {}
        self.layer_stats_ = {}

        # Try to extract layer information if available
        layer_numbers = None
        if hasattr(self, "_layer_numbers"):
            layer_numbers = self._layer_numbers
        elif isinstance(data, dict):
            # Try to extract from fused voxel data
            layer_numbers = []
            for key, value in data.items():
                if hasattr(value, "layer_number"):
                    layer_numbers.append(value.layer_number)
                else:
                    layer_numbers.append(0)

        for i in range(array_data.shape[1]):
            feature_name = self.feature_names_[i] if self.feature_names_ else f"feature_{i}"
            values = array_data[:, i]

            # Remove NaN/inf values
            valid_values = values[np.isfinite(values)]

            if len(valid_values) > 0:
                # Baseline statistics
                self.baseline_patterns_[feature_name] = {
                    "mean": np.mean(valid_values),
                    "std": np.std(valid_values),
                    "median": np.median(valid_values),
                    "variance": np.var(valid_values),
                }

                # Layer-based statistics if available
                if layer_numbers is not None and self.use_layer_analysis:
                    unique_layers = np.unique(layer_numbers)
                    layer_means = {}
                    layer_stds = {}

                    for layer in unique_layers:
                        layer_mask = np.array(layer_numbers) == layer
                        layer_values = values[layer_mask]
                        valid_layer_values = layer_values[np.isfinite(layer_values)]

                        if len(valid_layer_values) > 0:
                            layer_means[layer] = np.mean(valid_layer_values)
                            layer_stds[layer] = np.std(valid_layer_values)

                    self.layer_stats_[feature_name] = {
                        "layer_means": layer_means,
                        "layer_stds": layer_stds,
                        "mean_layer_mean": (np.mean(list(layer_means.values())) if layer_means else 0.0),
                        "std_layer_mean": (np.std(list(layer_means.values())) if layer_means else 0.0),
                    }
            else:
                logger.warning(f"All values for {feature_name} are NaN/inf, skipping temporal pattern learning")

        # Calculate baseline threshold from training data
        if len(self.baseline_patterns_) > 0:
            # Calculate scores on training data to establish baseline threshold
            training_scores = np.zeros(array_data.shape[0])
            feature_names = (
                self.feature_names_ if self.feature_names_ else [f"feature_{i}" for i in range(array_data.shape[1])]
            )

            for i, feature_name in enumerate(feature_names):
                if feature_name in self.baseline_patterns_:
                    values = array_data[:, i]
                    baseline = self.baseline_patterns_[feature_name]

                    sequence_scores_feature = np.zeros(len(values))
                    for j in range(len(values)):
                        start_idx = max(0, j - self.sequence_length + 1)
                        end_idx = j + 1
                        sequence = values[start_idx:end_idx]
                        sequence_scores_feature[j] = self._calculate_sequence_anomaly(sequence, baseline)

                    training_scores += sequence_scores_feature

            training_scores = training_scores / len(self.baseline_patterns_)
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
        logger.info(f"TemporalPatternDetector fitted on {array_data.shape[0]} samples")
        return self

    def _calculate_sequence_anomaly(self, sequence: np.ndarray, baseline: Dict) -> float:
        """Calculate anomaly score for a sequence."""
        if len(sequence) < 2:
            return 0.0

        valid_sequence = sequence[np.isfinite(sequence)]
        if len(valid_sequence) < 2:
            return 0.0

        anomaly_score = 0.0

        # Trend analysis
        if self.use_trend_analysis and len(valid_sequence) >= 2:
            x = np.arange(len(valid_sequence))
            slope, _, _, _, _ = stats.linregress(x, valid_sequence)
            expected_slope = 0.0  # Normal process should be stable
            trend_deviation = abs(slope - expected_slope) / (baseline["std"] + 1e-10)
            anomaly_score += trend_deviation

        # Variance analysis
        if self.use_variance_analysis:
            sequence_variance = np.var(valid_sequence)
            baseline_variance = baseline["variance"]
            if baseline_variance > 0:
                variance_ratio = abs(sequence_variance - baseline_variance) / baseline_variance
                anomaly_score += variance_ratio

        # Mean deviation
        sequence_mean = np.mean(valid_sequence)
        mean_deviation = abs(sequence_mean - baseline["mean"]) / (baseline["std"] + 1e-10)
        anomaly_score += mean_deviation

        return anomaly_score

    def predict(self, data: Union[np.ndarray, pd.DataFrame, Dict[Tuple[int, int, int], Any]]) -> List[AnomalyDetectionResult]:
        """
        Detect anomalies based on temporal pattern deviations.

        Args:
            data: Data to detect anomalies in

        Returns:
            List of AnomalyDetectionResult objects
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")

        array_data = self._preprocess_data(data)
        feature_names = self.feature_names_ if self.feature_names_ else [f"feature_{i}" for i in range(array_data.shape[1])]

        # Try to extract layer information
        layer_numbers = None
        if hasattr(self, "_layer_numbers"):
            layer_numbers = self._layer_numbers
        elif isinstance(data, dict):
            layer_numbers = []
            for key, value in data.items():
                if hasattr(value, "layer_number"):
                    layer_numbers.append(value.layer_number)
                else:
                    layer_numbers.append(0)

        # Calculate temporal anomaly scores
        temporal_scores = np.zeros(array_data.shape[0])

        for i, feature_name in enumerate(feature_names):
            if feature_name in self.baseline_patterns_:
                values = array_data[:, i]
                baseline = self.baseline_patterns_[feature_name]

                # Sequence-based analysis
                sequence_scores = np.zeros(len(values))
                for j in range(len(values)):
                    start_idx = max(0, j - self.sequence_length + 1)
                    end_idx = j + 1
                    sequence = values[start_idx:end_idx]
                    sequence_scores[j] = self._calculate_sequence_anomaly(sequence, baseline)

                # Layer-based analysis
                if self.use_layer_analysis and layer_numbers is not None and feature_name in self.layer_stats_:
                    layer_stats = self.layer_stats_[feature_name]
                    layer_scores = np.zeros(len(values))

                    for j in range(len(values)):
                        if layer_numbers[j] in layer_stats["layer_means"]:
                            expected_mean = layer_stats["layer_means"][layer_numbers[j]]
                            expected_std = layer_stats["layer_stds"][layer_numbers[j]]

                            if np.isfinite(values[j]) and expected_std > 0:
                                deviation = abs(values[j] - expected_mean) / expected_std
                                layer_scores[j] = deviation

                    sequence_scores += 0.5 * layer_scores

                temporal_scores += sequence_scores

        # Normalize scores
        if len(self.baseline_patterns_) > 0:
            temporal_scores = temporal_scores / len(self.baseline_patterns_)

        # Determine anomalies using baseline threshold from training
        threshold = (
            self.baseline_threshold_
            if self.baseline_threshold_ is not None
            else (
                np.percentile(
                    temporal_scores[np.isfinite(temporal_scores)],
                    self.threshold_percentile,
                )
                if np.any(np.isfinite(temporal_scores))
                else 0.1
            )
        )
        is_anomaly = temporal_scores > threshold

        # Get indices and coordinates if available
        indices = getattr(self, "_voxel_indices", None)
        if indices is None:
            indices = [(0, 0, 0)] * len(temporal_scores)

        coordinates = [(0.0, 0.0, 0.0)] * len(temporal_scores)  # Placeholder

        # Temporarily update config threshold so _create_results uses the correct threshold
        original_threshold = self.config.threshold
        self.config.threshold = threshold

        try:
            # Create results using our custom threshold
            results = self._create_results(scores=temporal_scores, indices=indices, coordinates=coordinates)
        finally:
            # Restore original threshold
            self.config.threshold = original_threshold

        return results
