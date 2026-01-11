"""
Threshold Manager

Alert threshold management and dynamic threshold adjustment.
Supports absolute, relative, rate-of-change, and SPC-based thresholds.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import logging
from datetime import datetime
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class ThresholdConfig:
    """Configuration for alert thresholds."""

    metric_name: str
    threshold_type: str = "absolute"  # 'absolute', 'relative', 'rate_of_change', 'spc_limit'
    lower_threshold: Optional[float] = None
    upper_threshold: Optional[float] = None
    window_size: int = 100  # Samples for moving window
    enable_spc_integration: bool = False
    spc_baseline_id: Optional[str] = None
    severity_mapping: Dict[str, str] = field(default_factory=dict)  # threshold -> severity

    def __post_init__(self):
        """Validate threshold configuration."""
        if self.threshold_type not in ["absolute", "relative", "rate_of_change", "spc_limit"]:
            raise ValueError(f"Invalid threshold_type: {self.threshold_type}")

        if self.threshold_type == "absolute":
            if self.lower_threshold is None and self.upper_threshold is None:
                raise ValueError("At least one threshold (lower or upper) must be specified for absolute thresholds")


class ThresholdManager:
    """
    Alert threshold management.

    Provides:
    - Threshold configuration for metrics
    - Threshold checking with different types
    - SPC integration for dynamic thresholds
    - Rate-of-change detection
    - Adaptive thresholds
    """

    def __init__(self):
        """Initialize threshold manager."""
        # Threshold configurations by metric name
        self._thresholds: Dict[str, ThresholdConfig] = {}

        # Historical values for rate-of-change and relative thresholds
        self._history: Dict[str, deque] = {}

        # SPC baselines (for SPC-based thresholds)
        self._spc_baselines: Dict[str, Any] = {}  # Will store BaselineStatistics

        logger.info("ThresholdManager initialized")

    def add_threshold(self, metric_name: str, threshold_config: ThresholdConfig) -> None:
        """
        Add threshold configuration for metric.

        Args:
            metric_name: Metric name
            threshold_config: ThresholdConfig for this metric
        """
        if threshold_config.metric_name != metric_name:
            threshold_config.metric_name = metric_name

        self._thresholds[metric_name] = threshold_config

        # Initialize history buffer if needed
        if threshold_config.threshold_type in ["relative", "rate_of_change"]:
            self._history[metric_name] = deque(maxlen=threshold_config.window_size)

        logger.info(f"Added threshold for metric: {metric_name} (type: {threshold_config.threshold_type})")

    def remove_threshold(self, metric_name: str) -> None:
        """
        Remove threshold configuration.

        Args:
            metric_name: Metric name
        """
        if metric_name in self._thresholds:
            del self._thresholds[metric_name]

        if metric_name in self._history:
            del self._history[metric_name]

        if metric_name in self._spc_baselines:
            del self._spc_baselines[metric_name]

        logger.info(f"Removed threshold for metric: {metric_name}")

    def update_threshold(self, metric_name: str, threshold_config: ThresholdConfig) -> None:
        """
        Update threshold configuration.

        Args:
            metric_name: Metric name
            threshold_config: Updated ThresholdConfig
        """
        self.add_threshold(metric_name, threshold_config)

    def check_value(self, metric_name: str, value: float, timestamp: datetime) -> Optional["Alert"]:
        """
        Check value against thresholds and return alert if violated.

        Args:
            metric_name: Metric name
            value: Metric value
            timestamp: Timestamp of the value

        Returns:
            Alert if threshold violated, None otherwise
        """
        if metric_name not in self._thresholds:
            return None

        threshold_config = self._thresholds[metric_name]

        # Check based on threshold type
        alert = None

        if threshold_config.threshold_type == "absolute":
            alert = self._check_absolute_threshold(metric_name, value, threshold_config)

        elif threshold_config.threshold_type == "relative":
            alert = self._check_relative_threshold(metric_name, value, threshold_config)

        elif threshold_config.threshold_type == "rate_of_change":
            alert = self._check_rate_of_change_threshold(metric_name, value, timestamp, threshold_config)

        elif threshold_config.threshold_type == "spc_limit":
            alert = self._check_spc_threshold(metric_name, value, threshold_config)

        # Update history if needed
        if threshold_config.threshold_type in ["relative", "rate_of_change"]:
            if metric_name not in self._history:
                self._history[metric_name] = deque(maxlen=threshold_config.window_size)
            self._history[metric_name].append((timestamp, value))

        return alert

    def _check_absolute_threshold(self, metric_name: str, value: float, config: ThresholdConfig) -> Optional["Alert"]:
        """Check absolute threshold."""
        from .alert_system import Alert

        violated = False
        message = ""
        severity = "medium"

        if config.lower_threshold is not None and value < config.lower_threshold:
            violated = True
            message = f"{metric_name} value {value:.4f} is below lower threshold {config.lower_threshold:.4f}"
            severity = self._get_severity(config, "lower")

        if config.upper_threshold is not None and value > config.upper_threshold:
            violated = True
            message = f"{metric_name} value {value:.4f} is above upper threshold {config.upper_threshold:.4f}"
            severity = self._get_severity(config, "upper")

        if violated:
            return Alert(
                alert_id="",  # Will be generated by AlertSystem
                alert_type="quality_threshold",
                severity=severity,
                message=message,
                timestamp=datetime.now(),
                source="ThresholdManager",
                metadata={
                    "metric_name": metric_name,
                    "value": value,
                    "lower_threshold": config.lower_threshold,
                    "upper_threshold": config.upper_threshold,
                    "threshold_type": "absolute",
                },
            )

        return None

    def _check_relative_threshold(self, metric_name: str, value: float, config: ThresholdConfig) -> Optional["Alert"]:
        """Check relative threshold (percentage change from baseline)."""
        from .alert_system import Alert

        if metric_name not in self._history or len(self._history[metric_name]) == 0:
            return None

        # Calculate baseline (mean of historical values)
        history_values = [v for _, v in self._history[metric_name]]
        baseline = sum(history_values) / len(history_values)

        if baseline == 0:
            return None

        # Calculate relative change
        relative_change = abs((value - baseline) / baseline) * 100.0

        # Check against thresholds (interpreted as percentage)
        violated = False
        message = ""
        severity = "medium"

        if config.lower_threshold is not None and (value - baseline) / baseline * 100 < -config.lower_threshold:
            violated = True
            message = f"{metric_name} value {value:.4f} is {relative_change:.2f}% below baseline {baseline:.4f}"
            severity = self._get_severity(config, "lower")

        if config.upper_threshold is not None and (value - baseline) / baseline * 100 > config.upper_threshold:
            violated = True
            message = f"{metric_name} value {value:.4f} is {relative_change:.2f}% above baseline {baseline:.4f}"
            severity = self._get_severity(config, "upper")

        if violated:
            return Alert(
                alert_id="",
                alert_type="quality_threshold",
                severity=severity,
                message=message,
                timestamp=datetime.now(),
                source="ThresholdManager",
                metadata={
                    "metric_name": metric_name,
                    "value": value,
                    "baseline": baseline,
                    "relative_change_percent": relative_change,
                    "threshold_type": "relative",
                },
            )

        return None

    def _check_rate_of_change_threshold(
        self, metric_name: str, value: float, timestamp: datetime, config: ThresholdConfig
    ) -> Optional["Alert"]:
        """Check rate-of-change threshold."""
        from .alert_system import Alert

        if metric_name not in self._history or len(self._history[metric_name]) < 2:
            return None

        # Get last value
        last_timestamp, last_value = self._history[metric_name][-1]

        # Calculate rate of change (value per second)
        time_delta = (timestamp - last_timestamp).total_seconds()
        if time_delta == 0:
            return None

        rate_of_change = abs((value - last_value) / time_delta)

        # Check against thresholds (interpreted as rate)
        violated = False
        message = ""
        severity = "medium"

        if config.upper_threshold is not None and rate_of_change > config.upper_threshold:
            violated = True
            message = f"{metric_name} rate of change {rate_of_change:.4f} exceeds threshold {config.upper_threshold:.4f}"
            severity = self._get_severity(config, "upper")

        if violated:
            return Alert(
                alert_id="",
                alert_type="quality_threshold",
                severity=severity,
                message=message,
                timestamp=timestamp,
                source="ThresholdManager",
                metadata={
                    "metric_name": metric_name,
                    "value": value,
                    "previous_value": last_value,
                    "rate_of_change": rate_of_change,
                    "time_delta_seconds": time_delta,
                    "threshold_type": "rate_of_change",
                },
            )

        return None

    def _check_spc_threshold(self, metric_name: str, value: float, config: ThresholdConfig) -> Optional["Alert"]:
        """Check SPC-based threshold (using control limits)."""
        from .alert_system import Alert

        if not config.enable_spc_integration or metric_name not in self._spc_baselines:
            return None

        baseline = self._spc_baselines[metric_name]

        # Check against control limits (UCL and LCL)
        # Assuming baseline has mean and std attributes
        mean = baseline.mean if hasattr(baseline, "mean") else None
        std = baseline.std if hasattr(baseline, "std") else None

        if mean is None or std is None or std == 0:
            return None

        # Calculate control limits (using 3-sigma)
        ucl = mean + 3 * std
        lcl = mean - 3 * std

        violated = False
        message = ""
        severity = "high"

        if value > ucl:
            violated = True
            message = f"{metric_name} value {value:.4f} exceeds UCL {ucl:.4f} (mean: {mean:.4f}, std: {std:.4f})"
        elif value < lcl:
            violated = True
            message = f"{metric_name} value {value:.4f} below LCL {lcl:.4f} (mean: {mean:.4f}, std: {std:.4f})"

        if violated:
            return Alert(
                alert_id="",
                alert_type="spc_out_of_control",
                severity=severity,
                message=message,
                timestamp=datetime.now(),
                source="ThresholdManager",
                metadata={
                    "metric_name": metric_name,
                    "value": value,
                    "mean": mean,
                    "std": std,
                    "ucl": ucl,
                    "lcl": lcl,
                    "threshold_type": "spc_limit",
                    "spc_baseline_id": config.spc_baseline_id,
                },
            )

        return None

    def _get_severity(self, config: ThresholdConfig, threshold_direction: str) -> str:
        """Get severity from severity mapping."""
        key = f"{threshold_direction}_threshold"
        return config.severity_mapping.get(key, "medium")

    def integrate_spc_baseline(self, metric_name: str, baseline: Any) -> None:
        """
        Integrate SPC baseline for dynamic thresholds.

        Args:
            metric_name: Metric name
            baseline: BaselineStatistics object from SPC module
        """
        self._spc_baselines[metric_name] = baseline

        # Update threshold config if it exists
        if metric_name in self._thresholds:
            self._thresholds[metric_name].enable_spc_integration = True

        logger.info(f"Integrated SPC baseline for metric: {metric_name}")

    def get_current_thresholds(self) -> Dict[str, ThresholdConfig]:
        """
        Get current threshold configurations.

        Returns:
            Dictionary mapping metric names to ThresholdConfig objects
        """
        return self._thresholds.copy()
