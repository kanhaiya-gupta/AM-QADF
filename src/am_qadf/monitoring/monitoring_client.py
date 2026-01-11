"""
Monitoring Client

Unified monitoring interface for real-time process monitoring.
Integrates alert system, health monitoring, and threshold management.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import logging
from datetime import datetime
import threading

logger = logging.getLogger(__name__)


@dataclass
class MonitoringConfig:
    """Configuration for monitoring operations."""

    enable_alerts: bool = True
    alert_check_interval_seconds: float = 1.0
    enable_email_notifications: bool = False
    email_smtp_server: Optional[str] = None
    email_smtp_port: int = 587
    email_from_address: Optional[str] = None
    email_recipients: List[str] = field(default_factory=list)
    enable_sms_notifications: bool = False
    sms_provider: Optional[str] = None  # 'twilio', 'aws_sns', etc.
    sms_recipients: List[str] = field(default_factory=list)
    enable_dashboard_notifications: bool = True
    websocket_port: int = 8765
    enable_health_monitoring: bool = True
    health_check_interval_seconds: float = 5.0
    alert_cooldown_seconds: float = 300.0  # Prevent alert spam


class MonitoringClient:
    """
    Unified monitoring interface.

    Provides:
    - Metric registration and monitoring
    - Alert generation and management
    - Health monitoring for components
    - Threshold management
    - Notification dispatch
    """

    def __init__(self, config: Optional[MonitoringConfig] = None):
        """
        Initialize monitoring client.

        Args:
            config: Optional MonitoringConfig for default settings
        """
        self.config = config if config is not None else MonitoringConfig()

        # Initialize components
        from .alert_system import AlertSystem
        from .threshold_manager import ThresholdManager
        from .health_monitor import HealthMonitor
        from .notification_channels import NotificationChannels

        self.alert_system = AlertSystem(self.config)
        self.threshold_manager = ThresholdManager()
        self.health_monitor = HealthMonitor(check_interval_seconds=self.config.health_check_interval_seconds)
        self.notification_channels = NotificationChannels(self.config)

        # Current metrics
        self._metrics: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

        # Monitoring state
        self._is_monitoring = False
        self._monitoring_thread: Optional[threading.Thread] = None

        logger.info("MonitoringClient initialized")

    def start_monitoring(self) -> None:
        """Start monitoring services."""
        if self._is_monitoring:
            logger.warning("Monitoring is already running")
            return

        self._is_monitoring = True

        # Start health monitoring
        if self.config.enable_health_monitoring:
            self.health_monitor.start_monitoring()

        # Start monitoring loop
        def monitoring_loop():
            """Monitoring loop running in separate thread."""
            while self._is_monitoring:
                try:
                    self._check_metrics()
                    threading.Event().wait(self.config.alert_check_interval_seconds)
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    threading.Event().wait(1.0)

        self._monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self._monitoring_thread.start()

        logger.info("Monitoring started")

    def stop_monitoring(self) -> None:
        """Stop monitoring services."""
        if not self._is_monitoring:
            logger.warning("Monitoring is not running")
            return

        self._is_monitoring = False

        # Stop health monitoring
        if self.config.enable_health_monitoring:
            self.health_monitor.stop_monitoring()

        # Wait for monitoring thread
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5.0)

        logger.info("Monitoring stopped")

    def register_metric(self, metric_name: str, threshold_config: "ThresholdConfig") -> None:
        """
        Register metric for monitoring.

        Args:
            metric_name: Name of the metric
            threshold_config: ThresholdConfig for this metric
        """
        from .threshold_manager import ThresholdConfig

        self.threshold_manager.add_threshold(metric_name, threshold_config)

        with self._lock:
            self._metrics[metric_name] = {
                "name": metric_name,
                "value": None,
                "timestamp": None,
                "threshold_config": threshold_config,
            }

        logger.info(f"Registered metric: {metric_name}")

    def update_metric(self, metric_name: str, value: float, timestamp: Optional[datetime] = None) -> None:
        """
        Update metric value and check thresholds.

        Args:
            metric_name: Name of the metric
            value: Metric value
            timestamp: Optional timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()

        with self._lock:
            if metric_name not in self._metrics:
                logger.warning(f"Metric {metric_name} not registered, registering with default threshold")
                from .threshold_manager import ThresholdConfig

                # Create default config with wide thresholds to avoid violations
                default_config = ThresholdConfig(
                    metric_name=metric_name,
                    threshold_type="absolute",
                    lower_threshold=-float("inf"),
                    upper_threshold=float("inf"),
                )
                self.threshold_manager.add_threshold(metric_name, default_config)
                self._metrics[metric_name] = {
                    "name": metric_name,
                    "value": None,
                    "timestamp": None,
                    "threshold_config": default_config,
                }

            self._metrics[metric_name]["value"] = value
            self._metrics[metric_name]["timestamp"] = timestamp

        # Check thresholds
        if self.config.enable_alerts:
            alert = self.threshold_manager.check_value(metric_name, value, timestamp)
            if alert:
                generated_alert = self.alert_system.generate_alert(
                    alert_type=alert.alert_type,
                    severity=alert.severity,
                    message=alert.message,
                    source=alert.source,
                    metadata=alert.metadata,
                )

                # Dispatch alert through notification channels
                if generated_alert:
                    self.notification_channels.broadcast_alert(generated_alert)

    def get_current_metrics(self) -> Dict[str, float]:
        """
        Get current values of all monitored metrics.

        Returns:
            Dictionary mapping metric names to current values
        """
        with self._lock:
            return {name: data["value"] for name, data in self._metrics.items() if data["value"] is not None}

    def get_health_status(self, component_name: Optional[str] = None) -> Dict[str, "HealthStatus"]:
        """
        Get health status of components.

        Args:
            component_name: Optional component name (None = all components)

        Returns:
            Dictionary mapping component names to HealthStatus objects
        """
        if component_name:
            return {component_name: self.health_monitor.check_process_health(component_name)}
        else:
            return {"system": self.health_monitor.check_system_health(), **self.health_monitor.get_all_component_health()}

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> None:
        """
        Acknowledge an alert.

        Args:
            alert_id: Alert ID
            acknowledged_by: Name of person/system acknowledging
        """
        self.alert_system.acknowledge_alert(alert_id, acknowledged_by)

    def _check_metrics(self) -> None:
        """Internal method to check all registered metrics."""
        with self._lock:
            for metric_name, metric_data in self._metrics.items():
                if metric_data["value"] is not None and metric_data["timestamp"] is not None:
                    # Threshold check is done in update_metric, but we can add periodic checks here
                    pass
