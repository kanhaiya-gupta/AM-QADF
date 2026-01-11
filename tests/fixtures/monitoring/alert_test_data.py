"""
Test fixtures for alert test data.

Provides generators for alerts and alert history.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import uuid

from am_qadf.monitoring.alert_system import Alert


def generate_alert(
    alert_type: str = "quality_threshold",
    severity: str = "medium",
    source: str = "TestSource",
    message: str = None,
    timestamp: datetime = None,
    acknowledged: bool = False,
    metadata: Dict[str, Any] = None,
) -> Alert:
    """
    Generate a single alert.

    Args:
        alert_type: Type of alert (default: 'quality_threshold')
        severity: Alert severity ('low', 'medium', 'high', 'critical')
        source: Source component
        message: Optional message (defaults to generic message)
        timestamp: Optional timestamp (defaults to now)
        acknowledged: Whether alert is acknowledged
        metadata: Optional metadata dictionary

    Returns:
        Alert object
    """
    if timestamp is None:
        timestamp = datetime.now()

    if message is None:
        message = f"{alert_type} violation detected from {source}"

    if metadata is None:
        metadata = {
            "test_data": True,
            "metric_value": np.random.uniform(100.0, 200.0),
            "threshold": 150.0,
        }

    alert = Alert(
        alert_id=str(uuid.uuid4()),
        alert_type=alert_type,
        severity=severity,
        message=message,
        timestamp=timestamp,
        source=source,
        metadata=metadata,
        acknowledged=acknowledged,
    )

    if acknowledged:
        alert.acknowledged_by = "test_user"
        alert.acknowledged_at = timestamp

    return alert


def generate_alert_batch(
    n_alerts: int = 10,
    alert_types: List[str] = None,
    severities: List[str] = None,
    start_time: datetime = None,
    interval_seconds: float = 60.0,
) -> List[Alert]:
    """
    Generate a batch of alerts.

    Args:
        n_alerts: Number of alerts to generate
        alert_types: List of alert types (defaults to common types)
        severities: List of severities (defaults to all severities)
        start_time: Optional start time (defaults to now)
        interval_seconds: Time interval between alerts

    Returns:
        List of Alert objects
    """
    if start_time is None:
        start_time = datetime.now()

    if alert_types is None:
        alert_types = [
            "quality_threshold",
            "spc_out_of_control",
            "system_error",
            "performance_degradation",
        ]

    if severities is None:
        severities = ["low", "medium", "high", "critical"]

    alerts = []
    for i in range(n_alerts):
        timestamp = start_time + timedelta(seconds=i * interval_seconds)
        alert_type = np.random.choice(alert_types)
        severity = np.random.choice(severities)
        source = f"Source_{i % 5}"

        alert = generate_alert(
            alert_type=alert_type,
            severity=severity,
            source=source,
            timestamp=timestamp,
            acknowledged=False,
        )
        alerts.append(alert)

    return alerts


def generate_alert_history(
    duration_hours: float = 24.0, alert_rate_per_hour: float = 10.0, alert_types: List[str] = None
) -> List[Alert]:
    """
    Generate alert history over time period.

    Args:
        duration_hours: Duration in hours
        alert_rate_per_hour: Average alerts per hour
        alert_types: List of alert types

    Returns:
        List of Alert objects over time period
    """
    start_time = datetime.now() - timedelta(hours=duration_hours)
    n_alerts = int(duration_hours * alert_rate_per_hour)
    interval_seconds = (duration_hours * 3600.0) / n_alerts if n_alerts > 0 else 1.0

    return generate_alert_batch(
        n_alerts=n_alerts, alert_types=alert_types, start_time=start_time, interval_seconds=interval_seconds
    )


def generate_escalated_alert_sequence(base_alert: Alert, n_escalations: int = 3) -> List[Alert]:
    """
    Generate sequence of escalated alerts.

    Args:
        base_alert: Base alert to escalate
        n_escalations: Number of escalation steps

    Returns:
        List of alerts showing escalation sequence
    """
    severity_order = ["low", "medium", "high", "critical"]

    alerts = [base_alert]
    current_severity = base_alert.severity

    try:
        current_index = severity_order.index(current_severity)
    except ValueError:
        current_index = 0

    for i in range(n_escalations):
        if current_index < len(severity_order) - 1:
            current_index += 1
            escalated_severity = severity_order[current_index]

            escalated_alert = Alert(
                alert_id=base_alert.alert_id,  # Same alert, escalated
                alert_type=base_alert.alert_type,
                severity=escalated_severity,
                message=f"{base_alert.message} (ESCALATED)",
                timestamp=base_alert.timestamp + timedelta(minutes=15 * (i + 1)),
                source=base_alert.source,
                metadata={**base_alert.metadata, "escalated": True, "escalation_level": i + 1},
                acknowledged=base_alert.acknowledged,
            )
            alerts.append(escalated_alert)
        else:
            break

    return alerts
