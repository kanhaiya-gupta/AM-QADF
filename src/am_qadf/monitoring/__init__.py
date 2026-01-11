"""
Monitoring Module

Provides real-time monitoring capabilities including:
- Unified monitoring interface
- Alert generation and escalation
- Multi-channel notifications (Email, SMS, Dashboard)
- Threshold management
- System and process health monitoring
- Alert and notification history storage
"""

from .monitoring_client import MonitoringClient, MonitoringConfig
from .alert_system import AlertSystem, Alert
from .health_monitor import HealthMonitor, HealthStatus
from .threshold_manager import ThresholdManager, ThresholdConfig
from .notification_channels import NotificationChannels
from .monitoring_storage import MonitoringStorage

__all__ = [
    "MonitoringClient",
    "MonitoringConfig",
    "AlertSystem",
    "Alert",
    "HealthMonitor",
    "HealthStatus",
    "ThresholdManager",
    "ThresholdConfig",
    "NotificationChannels",
    "MonitoringStorage",
]
