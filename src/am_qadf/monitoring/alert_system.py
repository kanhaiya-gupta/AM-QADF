"""
Alert System

Alert generation, escalation, and management.
Provides alert cooldown, acknowledgment, and history tracking.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
import uuid
import threading

logger = logging.getLogger(__name__)


@dataclass
class Alert:
    """Alert information."""

    alert_id: str
    alert_type: str  # 'quality_threshold', 'spc_out_of_control', 'system_error', etc.
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    timestamp: datetime
    source: str  # Component that generated alert
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None

    def __post_init__(self):
        """Generate alert ID if not provided."""
        if not self.alert_id:
            self.alert_id = str(uuid.uuid4())


class AlertSystem:
    """
    Alert generation and escalation system.

    Provides:
    - Alert generation with unique IDs
    - Alert cooldown to prevent spam
    - Alert acknowledgment
    - Alert history tracking
    - Alert escalation
    """

    def __init__(self, config: "MonitoringConfig"):
        """
        Initialize alert system.

        Args:
            config: MonitoringConfig with alert settings
        """
        from .monitoring_client import MonitoringConfig

        self.config = config

        # Active alerts (not acknowledged)
        self._active_alerts: Dict[str, Alert] = {}

        # Alert history
        self._alert_history: List[Alert] = []

        # Cooldown tracking (alert_type -> last_alert_time)
        self._cooldown: Dict[str, datetime] = {}

        # Thread safety
        self._lock = threading.Lock()

        logger.info("AlertSystem initialized")

    def generate_alert(
        self, alert_type: str, severity: str, message: str, source: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Alert:
        """
        Generate and dispatch alert.

        Args:
            alert_type: Type of alert
            severity: Alert severity ('low', 'medium', 'high', 'critical')
            message: Alert message
            source: Source component
            metadata: Optional additional metadata

        Returns:
            Alert object
        """
        # Check cooldown
        if self._is_in_cooldown(alert_type):
            logger.debug(f"Alert {alert_type} is in cooldown, skipping")
            return None

        # Create alert
        alert = Alert(
            alert_id=str(uuid.uuid4()),
            alert_type=alert_type,
            severity=severity,
            message=message,
            timestamp=datetime.now(),
            source=source,
            metadata=metadata or {},
        )

        with self._lock:
            # Add to active alerts
            self._active_alerts[alert.alert_id] = alert

            # Add to history
            self._alert_history.append(alert)

            # Update cooldown
            self._cooldown[alert_type] = datetime.now()

        # Dispatch notifications
        self._dispatch_alert(alert)

        logger.info(f"Generated alert: {alert.alert_id} ({alert_type}, {severity}) - {message}")

        return alert

    def check_thresholds(self, metric_name: str, value: float) -> List[Alert]:
        """
        Check thresholds and generate alerts if needed.

        Note: This method is typically called by ThresholdManager.
        This is kept for backward compatibility and direct use.

        Args:
            metric_name: Metric name
            value: Metric value

        Returns:
            List of generated alerts
        """
        # This method will be used by ThresholdManager
        # For now, return empty list (threshold checking is done in ThresholdManager)
        return []

    def escalate_alert(self, alert_id: str) -> None:
        """
        Escalate alert if not acknowledged.

        Args:
            alert_id: Alert ID to escalate
        """
        with self._lock:
            if alert_id not in self._active_alerts:
                logger.warning(f"Alert {alert_id} not found or already acknowledged")
                return

            alert = self._active_alerts[alert_id]

            # Increase severity if possible
            severity_order = ["low", "medium", "high", "critical"]
            current_index = severity_order.index(alert.severity) if alert.severity in severity_order else 0

            if current_index < len(severity_order) - 1:
                alert.severity = severity_order[current_index + 1]
                alert.metadata["escalated"] = True
                alert.metadata["escalated_at"] = datetime.now().isoformat()

                logger.info(f"Escalated alert {alert_id} to {alert.severity}")

                # Re-dispatch escalated alert
                self._dispatch_alert(alert)

    def get_active_alerts(self, severity: Optional[str] = None) -> List[Alert]:
        """
        Get active (unacknowledged) alerts.

        Args:
            severity: Optional severity filter

        Returns:
            List of active alerts
        """
        with self._lock:
            alerts = list(self._active_alerts.values())

            if severity:
                alerts = [a for a in alerts if a.severity == severity]

            return alerts

    def get_alert_history(
        self, start_time: datetime, end_time: datetime, filters: Optional[Dict[str, Any]] = None
    ) -> List[Alert]:
        """
        Get alert history.

        Args:
            start_time: Start time for history
            end_time: End time for history
            filters: Optional filters (alert_type, severity, source, etc.)

        Returns:
            List of alerts in time range
        """
        with self._lock:
            history = [alert for alert in self._alert_history if start_time <= alert.timestamp <= end_time]

            # Apply filters
            if filters:
                if "alert_type" in filters:
                    history = [a for a in history if a.alert_type == filters["alert_type"]]
                if "severity" in filters:
                    history = [a for a in history if a.severity == filters["severity"]]
                if "source" in filters:
                    history = [a for a in history if a.source == filters["source"]]
                if "acknowledged" in filters:
                    history = [a for a in history if a.acknowledged == filters["acknowledged"]]

            return history

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> None:
        """
        Acknowledge an alert.

        Args:
            alert_id: Alert ID
            acknowledged_by: Name of person/system acknowledging
        """
        with self._lock:
            if alert_id not in self._active_alerts:
                logger.warning(f"Alert {alert_id} not found or already acknowledged")
                return

            alert = self._active_alerts[alert_id]
            alert.acknowledged = True
            alert.acknowledged_by = acknowledged_by
            alert.acknowledged_at = datetime.now()

            # Remove from active alerts
            del self._active_alerts[alert_id]

            logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")

    def enable_alert_cooldown(self, alert_type: str, cooldown_seconds: float) -> None:
        """
        Enable cooldown period for alert type.

        Args:
            alert_type: Alert type
            cooldown_seconds: Cooldown period in seconds
        """
        # Cooldown is managed per alert type via config.alert_cooldown_seconds
        # This method allows overriding per alert type
        self._alert_cooldown_override = getattr(self, "_alert_cooldown_override", {})
        self._alert_cooldown_override[alert_type] = cooldown_seconds
        logger.info(f"Set cooldown for {alert_type} to {cooldown_seconds} seconds")

    def _is_in_cooldown(self, alert_type: str) -> bool:
        """Check if alert type is in cooldown period."""
        if alert_type not in self._cooldown:
            return False

        # Get cooldown duration (use override if available, else config default)
        cooldown_override = getattr(self, "_alert_cooldown_override", {})
        cooldown_seconds = cooldown_override.get(alert_type, self.config.alert_cooldown_seconds)

        last_alert_time = self._cooldown[alert_type]
        elapsed = (datetime.now() - last_alert_time).total_seconds()

        return elapsed < cooldown_seconds

    def _dispatch_alert(self, alert: Alert) -> None:
        """Dispatch alert to notification channels."""
        # Note: Actual dispatch is handled by MonitoringClient's notification_channels
        # This method just logs the alert for now
        # MonitoringClient will call notification_channels.broadcast_alert()

        # Get notification channels from config
        channels = []
        if self.config.enable_dashboard_notifications:
            channels.append("dashboard")
        if self.config.enable_email_notifications and self.config.email_recipients:
            channels.append("email")
        if self.config.enable_sms_notifications and self.config.sms_recipients:
            channels.append("sms")

        logger.debug(f"Alert {alert.alert_id} ready for dispatch to channels: {channels}")
