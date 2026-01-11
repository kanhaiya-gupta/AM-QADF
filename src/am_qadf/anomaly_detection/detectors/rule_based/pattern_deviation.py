"""
Pattern Deviation Detection

Detects anomalies using Statistical Process Control (SPC) charts and pattern analysis.
Identifies deviations from expected patterns in process parameters.
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


class PatternDeviationDetector(BaseAnomalyDetector):
    """
    Pattern deviation detector using SPC charts.

    Detects anomalies by identifying deviations from expected statistical patterns.
    Uses control charts (X-bar, R-chart, etc.) and pattern recognition.
    """

    def __init__(
        self,
        control_limit_sigma: float = 3.0,
        pattern_window: int = 10,
        use_trend_detection: bool = True,
        use_cyclical_detection: bool = True,
        threshold_percentile: float = 95.0,
        use_spc_module: bool = True,  # Use new SPC module
        config: Optional[AnomalyDetectionConfig] = None,
    ):
        """
        Initialize pattern deviation detector.

        Args:
            control_limit_sigma: Number of standard deviations for control limits (default: 3.0)
            pattern_window: Window size for pattern analysis (default: 10)
            use_trend_detection: Detect trend patterns (increasing/decreasing)
            use_cyclical_detection: Detect cyclical patterns
            threshold_percentile: Percentile threshold for anomaly detection (default: 95.0)
            use_spc_module: Whether to use the new SPC module (default: True)
            config: Optional detector configuration
        """
        if config is None:
            config = AnomalyDetectionConfig()
        super().__init__(config)

        self.control_limit_sigma = control_limit_sigma
        self.pattern_window = pattern_window
        self.use_trend_detection = use_trend_detection
        self.use_cyclical_detection = use_cyclical_detection
        self.threshold_percentile = threshold_percentile
        self.use_spc_module = use_spc_module

        self.control_limits_ = None
        self.baseline_stats_ = None
        self.feature_names_ = None

        # Initialize SPC client if enabled
        self.spc_client = None
        self.spc_config = None
        if use_spc_module:
            try:
                from am_qadf.analytics.spc import SPCClient, SPCConfig

                self.spc_config = SPCConfig(
                    control_limit_sigma=control_limit_sigma,
                    subgroup_size=1,  # Use individual chart for anomaly detection
                    enable_warnings=True,
                    warning_sigma=2.0,
                )
                self.spc_client = SPCClient(config=self.spc_config, mongo_client=None)
                logger.info("PatternDeviationDetector using SPC module")
            except ImportError as e:
                logger.warning(f"SPC module not available, using legacy implementation: {e}")
                self.use_spc_module = False

    def fit(
        self,
        data: Union[np.ndarray, pd.DataFrame, Dict[Tuple[int, int, int], Any]],
        labels: Optional[np.ndarray] = None,
    ) -> "PatternDeviationDetector":
        """
        Fit the detector by learning baseline patterns from normal data.

        Args:
            data: Training data (normal data only)
            labels: Not used for pattern-based detection

        Returns:
            Self for method chaining
        """
        array_data = self._preprocess_data(data)
        self.feature_names_ = getattr(self, "_feature_names", None)

        # Use SPC module if available, otherwise use legacy implementation
        if self.use_spc_module and self.spc_client is not None:
            # Establish baselines using SPC module
            self.baseline_stats_ = {}
            self.control_limits_ = {}

            for i in range(array_data.shape[1]):
                feature_name = self.feature_names_[i] if self.feature_names_ else f"feature_{i}"
                values = array_data[:, i]

                # Remove NaN/inf values
                valid_values = values[np.isfinite(values)]

                if len(valid_values) > 0:
                    try:
                        # Establish baseline using SPC module
                        baseline = self.spc_client.establish_baseline(valid_values, subgroup_size=1, config=self.spc_config)

                        # Create individual chart to get control limits
                        chart_result = self.spc_client.create_control_chart(
                            valid_values, chart_type="individual", config=self.spc_config
                        )

                        # Store baseline stats
                        self.baseline_stats_[feature_name] = {
                            "mean": baseline.mean,
                            "std": baseline.std,
                            "median": baseline.median,
                            "min": baseline.min,
                            "max": baseline.max,
                        }

                        # Store control limits from chart
                        self.control_limits_[feature_name] = {
                            "ucl": chart_result.upper_control_limit,
                            "lcl": chart_result.lower_control_limit,
                            "center": chart_result.center_line,
                        }
                    except Exception as e:
                        logger.warning(f"Error using SPC module for {feature_name}, falling back to legacy: {e}")
                        # Fall back to legacy calculation
                        mean_val = np.mean(valid_values)
                        std_val = np.std(valid_values)
                        self.baseline_stats_[feature_name] = {
                            "mean": mean_val,
                            "std": std_val,
                            "median": np.median(valid_values),
                            "min": np.min(valid_values),
                            "max": np.max(valid_values),
                        }
                        self.control_limits_[feature_name] = {
                            "ucl": mean_val + self.control_limit_sigma * std_val,
                            "lcl": mean_val - self.control_limit_sigma * std_val,
                            "center": mean_val,
                        }
                else:
                    logger.warning(f"All values for {feature_name} are NaN/inf, skipping pattern learning")
        else:
            # Legacy implementation (maintain backward compatibility)
            self.baseline_stats_ = {}
            self.control_limits_ = {}

            for i in range(array_data.shape[1]):
                feature_name = self.feature_names_[i] if self.feature_names_ else f"feature_{i}"
                values = array_data[:, i]

                # Remove NaN/inf values
                valid_values = values[np.isfinite(values)]

                if len(valid_values) > 0:
                    mean_val = np.mean(valid_values)
                    std_val = np.std(valid_values)
                    median_val = np.median(valid_values)

                    self.baseline_stats_[feature_name] = {
                        "mean": mean_val,
                        "std": std_val,
                        "median": median_val,
                        "min": np.min(valid_values),
                        "max": np.max(valid_values),
                    }

                    # Calculate control limits (UCL and LCL)
                    ucl = mean_val + self.control_limit_sigma * std_val
                    lcl = mean_val - self.control_limit_sigma * std_val

                    self.control_limits_[feature_name] = {
                        "ucl": ucl,
                        "lcl": lcl,
                        "center": mean_val,
                    }
                else:
                    logger.warning(f"All values for {feature_name} are NaN/inf, skipping pattern learning")

        self.is_fitted = True
        logger.info(f"PatternDeviationDetector fitted on {array_data.shape[0]} samples")
        return self

    def _detect_trend(self, values: np.ndarray) -> float:
        """Detect trend in values using linear regression."""
        if len(values) < 2:
            return 0.0

        x = np.arange(len(values))
        valid_mask = np.isfinite(values)

        if np.sum(valid_mask) < 2:
            return 0.0

        x_valid = x[valid_mask]
        y_valid = values[valid_mask]

        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_valid, y_valid)

        # Return normalized trend score
        trend_score = abs(slope) / (np.std(y_valid) + 1e-10)
        return trend_score

    def _detect_cyclical(self, values: np.ndarray) -> float:
        """Detect cyclical patterns using autocorrelation."""
        if len(values) < self.pattern_window:
            return 0.0

        valid_values = values[np.isfinite(values)]
        if len(valid_values) < self.pattern_window:
            return 0.0

        # Simple autocorrelation at lag = pattern_window
        if len(valid_values) > self.pattern_window:
            lagged = valid_values[: -self.pattern_window]
            current = valid_values[self.pattern_window :]
            correlation = np.corrcoef(lagged, current)[0, 1]
            return abs(correlation) if not np.isnan(correlation) else 0.0

        return 0.0

    def predict(self, data: Union[np.ndarray, pd.DataFrame, Dict[Tuple[int, int, int], Any]]) -> List[AnomalyDetectionResult]:
        """
        Detect anomalies based on pattern deviations.

        Uses SPC module if available to detect out-of-control points and rule violations.

        Args:
            data: Data to detect anomalies in

        Returns:
            List of AnomalyDetectionResult objects
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")

        array_data = self._preprocess_data(data)
        feature_names = self.feature_names_ if self.feature_names_ else [f"feature_{i}" for i in range(array_data.shape[1])]

        # Calculate deviation scores
        deviation_scores = np.zeros(array_data.shape[0])
        all_rule_violations = set()  # Track all points with rule violations

        # Use SPC module if available
        if self.use_spc_module and self.spc_client is not None:
            for i, feature_name in enumerate(feature_names):
                if feature_name in self.control_limits_:
                    values = array_data[:, i]
                    valid_mask = np.isfinite(values)
                    valid_values = values[valid_mask]

                    if len(valid_values) == 0:
                        continue

                    try:
                        # Create control chart using SPC module
                        chart_result = self.spc_client.create_control_chart(
                            valid_values, chart_type="individual", config=self.spc_config
                        )

                        # Get out-of-control points from chart
                        ooc_indices = chart_result.out_of_control_points

                        # Detect rule violations
                        violations = []
                        try:
                            violations = self.spc_client.detect_rule_violations(chart_result, rule_set="western_electric")
                        except Exception as e:
                            logger.warning(f"Error detecting rule violations for {feature_name}: {e}")

                        # Collect all points with rule violations
                        for violation in violations:
                            all_rule_violations.update(violation.affected_points)

                        # Calculate deviation scores based on control limits
                        limits = self.control_limits_[feature_name]

                        # Map valid indices back to original array
                        valid_indices = np.where(valid_mask)[0]
                        for j, val in enumerate(valid_values):
                            orig_idx = valid_indices[j]

                            # Out-of-control violation
                            if j in ooc_indices:
                                if val > limits["ucl"]:
                                    violation_score = (val - limits["ucl"]) / (abs(limits["ucl"]) + 1e-10)
                                else:
                                    violation_score = (limits["lcl"] - val) / (abs(limits["lcl"]) + 1e-10)
                                deviation_scores[orig_idx] += violation_score * 2.0  # Higher weight for OOC

                            # Warning zone (between 2-sigma and 3-sigma)
                            if chart_result.upper_warning_limit and chart_result.lower_warning_limit:
                                if val > chart_result.upper_warning_limit or val < chart_result.lower_warning_limit:
                                    if j not in ooc_indices:  # Not already OOC
                                        warning_score = abs(val - chart_result.center_line) / (
                                            limits["ucl"] - chart_result.center_line + 1e-10
                                        )
                                        deviation_scores[orig_idx] += warning_score * 0.5

                            # Rule violation score
                            if j in all_rule_violations:
                                deviation_scores[orig_idx] += 1.0

                    except Exception as e:
                        logger.warning(f"Error using SPC module for {feature_name}, using legacy method: {e}")
                        # Fall back to legacy implementation for this feature
                        limits = self.control_limits_[feature_name]
                        stats = self.baseline_stats_[feature_name]

                        ucl_violations = np.where(
                            values > limits["ucl"],
                            (values - limits["ucl"]) / (abs(limits["ucl"]) + 1e-10),
                            0,
                        )
                        lcl_violations = np.where(
                            values < limits["lcl"],
                            (limits["lcl"] - values) / (abs(limits["lcl"]) + 1e-10),
                            0,
                        )
                        control_violations = ucl_violations + lcl_violations

                        # Pattern deviations
                        pattern_scores = np.zeros(len(values))
                        if self.use_trend_detection or self.use_cyclical_detection:
                            for j in range(len(values)):
                                start_idx = max(0, j - self.pattern_window + 1)
                                end_idx = j + 1
                                window_values = values[start_idx:end_idx]
                                if len(window_values) >= 2:
                                    if self.use_trend_detection:
                                        pattern_scores[j] += self._detect_trend(window_values)
                                    if self.use_cyclical_detection and len(window_values) >= self.pattern_window:
                                        pattern_scores[j] += self._detect_cyclical(window_values)

                        feature_score = control_violations + 0.5 * pattern_scores
                        deviation_scores += feature_score
        else:
            # Legacy implementation (maintain backward compatibility)
            for i, feature_name in enumerate(feature_names):
                if feature_name in self.control_limits_:
                    values = array_data[:, i]
                    limits = self.control_limits_[feature_name]
                    stats = self.baseline_stats_[feature_name]

                    # Control chart violations
                    ucl_violations = np.where(
                        values > limits["ucl"],
                        (values - limits["ucl"]) / (abs(limits["ucl"]) + 1e-10),
                        0,
                    )
                    lcl_violations = np.where(
                        values < limits["lcl"],
                        (limits["lcl"] - values) / (abs(limits["lcl"]) + 1e-10),
                        0,
                    )
                    control_violations = ucl_violations + lcl_violations

                    # Pattern deviations
                    pattern_scores = np.zeros(len(values))

                    if self.use_trend_detection or self.use_cyclical_detection:
                        # Use sliding window for pattern detection
                        for j in range(len(values)):
                            start_idx = max(0, j - self.pattern_window + 1)
                            end_idx = j + 1
                            window_values = values[start_idx:end_idx]

                            if len(window_values) >= 2:
                                if self.use_trend_detection:
                                    trend_score = self._detect_trend(window_values)
                                    pattern_scores[j] += trend_score

                                if self.use_cyclical_detection and len(window_values) >= self.pattern_window:
                                    cyclical_score = self._detect_cyclical(window_values)
                                    pattern_scores[j] += cyclical_score

                    # Combine control violations and pattern deviations
                    feature_score = control_violations + 0.5 * pattern_scores
                    deviation_scores += feature_score

        # Normalize scores
        if len(self.control_limits_) > 0:
            deviation_scores = deviation_scores / len(self.control_limits_)

        # Determine anomalies (use percentile threshold)
        threshold = (
            np.percentile(
                deviation_scores[np.isfinite(deviation_scores)],
                self.threshold_percentile,
            )
            if np.any(np.isfinite(deviation_scores))
            else 0.1
        )
        is_anomaly = deviation_scores > threshold

        # Get indices and coordinates if available
        indices = getattr(self, "_voxel_indices", None)
        if indices is None:
            indices = [(0, 0, 0)] * len(deviation_scores)

        coordinates = [(0.0, 0.0, 0.0)] * len(deviation_scores)  # Placeholder

        # Create results
        results = self._create_results(scores=deviation_scores, indices=indices, coordinates=coordinates)

        return results
