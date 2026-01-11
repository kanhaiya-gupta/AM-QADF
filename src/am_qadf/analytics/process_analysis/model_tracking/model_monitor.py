"""
Model Monitor for PBF-LB/M Systems

This module provides model monitoring and alerting capabilities for
tracking model performance and drift in production environments.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import threading
import time

logger = logging.getLogger(__name__)


@dataclass
class MonitoringConfig:
    """Configuration for model monitoring."""

    drift_threshold: float = 0.2  # Drift threshold (0-1)
    performance_degradation_threshold: float = 0.1  # 10% degradation threshold
    evaluation_interval_seconds: float = 3600.0  # Evaluate every hour by default
    enable_drift_detection: bool = True
    enable_performance_monitoring: bool = True
    alert_on_drift: bool = True
    alert_on_degradation: bool = True


class ModelMonitor:
    """
    Monitor model performance and drift in production.

    This class provides capabilities to continuously monitor model performance,
    detect drift, and trigger alerts or retraining when issues are detected.
    """

    def __init__(
        self,
        model_registry: "ModelRegistry",
        performance_tracker: "ModelPerformanceTracker",
        monitoring_config: Optional[MonitoringConfig] = None,
    ):
        """
        Initialize model monitor.

        Args:
            model_registry: ModelRegistry instance
            performance_tracker: ModelPerformanceTracker instance
            monitoring_config: Monitoring configuration (uses defaults if None)
        """
        self.model_registry = model_registry
        self.performance_tracker = performance_tracker
        self.config = monitoring_config or MonitoringConfig()

        self._monitoring_active = False
        self._monitoring_thread: Optional[threading.Thread] = None
        self._last_evaluation_time: Optional[datetime] = None

        logger.info(f"Model Monitor initialized for model {performance_tracker.model_id}")

    def monitor_model_performance(
        self, model_id: str, streaming_data_source: Callable, evaluation_interval: Optional[float] = None
    ) -> None:
        """
        Monitor model performance with streaming data.

        Args:
            model_id: Model ID to monitor
            streaming_data_source: Function that returns latest data for evaluation
            evaluation_interval: Interval between evaluations in seconds (uses config if None)
        """
        if evaluation_interval is None:
            evaluation_interval = self.config.evaluation_interval_seconds

        if self._monitoring_active:
            logger.warning("Monitoring already active, stopping existing monitoring")
            self.stop_monitoring()

        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop, args=(model_id, streaming_data_source, evaluation_interval), daemon=True
        )
        self._monitoring_thread.start()

        logger.info(f"Started monitoring model {model_id} (interval: {evaluation_interval}s)")

    def _monitoring_loop(self, model_id: str, streaming_data_source: Callable, evaluation_interval: float) -> None:
        """Internal monitoring loop."""
        try:
            while self._monitoring_active:
                try:
                    # Get latest data
                    current_data = streaming_data_source()

                    if current_data is None or len(current_data) == 0:
                        logger.debug("No data available for monitoring, waiting...")
                        time.sleep(evaluation_interval)
                        continue

                    # Load model
                    model, model_version = self.model_registry.load_model(model_id)

                    # Check drift
                    if self.config.enable_drift_detection:
                        drift_detected, drift_score = self.check_model_drift(model_id, current_data)

                        if drift_detected and self.config.alert_on_drift:
                            logger.warning(f"Model drift detected for {model_id}: drift_score={drift_score:.3f}")
                            # Could trigger alert here

                    # Evaluate performance (simplified - would need quality targets)
                    if self.config.enable_performance_monitoring:
                        # Note: Performance evaluation requires quality targets
                        # This is a placeholder - actual implementation would need
                        # streaming data with quality labels
                        pass

                    self._last_evaluation_time = datetime.now()

                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")

                # Wait for next evaluation interval
                time.sleep(evaluation_interval)

        except Exception as e:
            logger.error(f"Monitoring loop error: {e}")
            self._monitoring_active = False

    def stop_monitoring(self) -> None:
        """Stop monitoring."""
        self._monitoring_active = False
        if self._monitoring_thread is not None:
            self._monitoring_thread.join(timeout=5.0)
        logger.info("Stopped monitoring")

    def check_model_drift(
        self, model_id: str, current_data: pd.DataFrame, training_data: Optional[pd.DataFrame] = None
    ) -> Tuple[bool, float]:
        """
        Check for model drift.

        Args:
            model_id: Model ID to check
            current_data: Current data distribution
            training_data: Original training data (uses from registry if None)

        Returns:
            Tuple of (drift_detected, drift_score)
        """
        try:
            # Get training data if not provided
            if training_data is None:
                try:
                    _, model_version = self.model_registry.load_model(model_id)
                    # Try to get training data from metadata
                    if "training_data_path" in model_version.metadata:
                        import pickle

                        with open(model_version.metadata["training_data_path"], "rb") as f:
                            training_data = pickle.load(f)
                    else:
                        logger.warning("Training data not available in model metadata")
                        return False, 0.0
                except Exception as e:
                    logger.warning(f"Could not load training data: {e}")
                    return False, 0.0

            # Calculate drift score using performance tracker
            drift_score = self.performance_tracker.calculate_drift_score(current_data, training_data)

            drift_detected = drift_score > self.config.drift_threshold

            if drift_detected:
                logger.warning(
                    f"Model drift detected: drift_score={drift_score:.3f} " f"(threshold: {self.config.drift_threshold})"
                )

            return drift_detected, drift_score

        except Exception as e:
            logger.error(f"Error checking model drift: {e}")
            return False, 0.0

    def trigger_model_retraining(self, model_id: str, trigger_reason: str) -> None:
        """
        Trigger model retraining when drift detected or performance degraded.

        Args:
            model_id: Model ID to retrain
            trigger_reason: Reason for retraining (e.g., 'drift_detected', 'performance_degraded')
        """
        try:
            logger.info(f"Triggering retraining for model {model_id}: {trigger_reason}")

            # Get current model metadata
            _, model_version = self.model_registry.load_model(model_id)

            # Log retraining trigger
            retraining_log = {
                "model_id": model_id,
                "trigger_reason": trigger_reason,
                "trigger_time": datetime.now().isoformat(),
                "current_version": model_version.version,
                "performance_metrics": model_version.performance_metrics,
                "drift_score": getattr(model_version, "drift_score", 0.0),
            }

            logger.info(f"Retraining triggered: {retraining_log}")

            # In a real implementation, this would:
            # 1. Collect new training data
            # 2. Retrain the model
            # 3. Validate the new model
            # 4. Register the new model version
            # 5. Compare with previous version
            # 6. Deploy if better

            # For now, just log the trigger

        except Exception as e:
            logger.error(f"Error triggering model retraining: {e}")

    def check_performance_degradation(self, metric_name: Optional[str] = None) -> Tuple[bool, float]:
        """
        Check if model performance has degraded.

        Args:
            metric_name: Specific metric to check (uses first metric if None)

        Returns:
            Tuple of (degradation_detected, degradation_percentage)
        """
        return self.performance_tracker.detect_performance_degradation(
            threshold=self.config.performance_degradation_threshold, metric_name=metric_name
        )

    def get_monitoring_status(self) -> Dict[str, Any]:
        """
        Get current monitoring status.

        Returns:
            Dictionary with monitoring status information
        """
        status = {
            "monitoring_active": self._monitoring_active,
            "last_evaluation_time": self._last_evaluation_time.isoformat() if self._last_evaluation_time else None,
            "model_id": self.performance_tracker.model_id,
            "drift_threshold": self.config.drift_threshold,
            "performance_degradation_threshold": self.config.performance_degradation_threshold,
            "evaluation_interval_seconds": self.config.evaluation_interval_seconds,
        }

        # Check current drift and performance
        if self.config.enable_drift_detection:
            # Would need training data to check drift
            pass

        if self.config.enable_performance_monitoring:
            degradation_detected, degradation_pct = self.check_performance_degradation()
            status["performance_degradation_detected"] = degradation_detected
            status["performance_degradation_percentage"] = degradation_pct

        return status
