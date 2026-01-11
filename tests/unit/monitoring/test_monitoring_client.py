"""
Unit tests for MonitoringClient.

Tests for unified monitoring interface.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from am_qadf.monitoring.monitoring_client import (
    MonitoringClient,
    MonitoringConfig,
)
from am_qadf.monitoring.threshold_manager import ThresholdConfig


class TestMonitoringConfig:
    """Test suite for MonitoringConfig dataclass."""

    @pytest.mark.unit
    def test_monitoring_config_creation_defaults(self):
        """Test creating MonitoringConfig with defaults."""
        config = MonitoringConfig()

        assert config.enable_alerts is True
        assert config.alert_check_interval_seconds == 1.0
        assert config.enable_email_notifications is False
        assert config.email_smtp_port == 587
        assert config.enable_sms_notifications is False
        assert config.enable_dashboard_notifications is True
        assert config.websocket_port == 8765
        assert config.enable_health_monitoring is True
        assert config.health_check_interval_seconds == 5.0
        assert config.alert_cooldown_seconds == 300.0

    @pytest.mark.unit
    def test_monitoring_config_custom(self):
        """Test creating MonitoringConfig with custom values."""
        config = MonitoringConfig(
            enable_alerts=False,
            alert_check_interval_seconds=2.0,
            enable_email_notifications=True,
            email_smtp_server="smtp.example.com",
            email_smtp_port=465,
            email_from_address="alerts@example.com",
            email_recipients=["admin@example.com"],
            enable_sms_notifications=True,
            sms_provider="twilio",
            sms_recipients=["+1234567890"],
            enable_dashboard_notifications=False,
            websocket_port=9000,
            enable_health_monitoring=False,
            health_check_interval_seconds=10.0,
            alert_cooldown_seconds=600.0,
        )

        assert config.enable_alerts is False
        assert config.alert_check_interval_seconds == 2.0
        assert config.enable_email_notifications is True
        assert config.email_smtp_server == "smtp.example.com"
        assert config.email_smtp_port == 465
        assert config.email_from_address == "alerts@example.com"
        assert config.email_recipients == ["admin@example.com"]
        assert config.enable_sms_notifications is True
        assert config.sms_provider == "twilio"
        assert config.sms_recipients == ["+1234567890"]
        assert config.enable_dashboard_notifications is False
        assert config.websocket_port == 9000
        assert config.enable_health_monitoring is False
        assert config.health_check_interval_seconds == 10.0
        assert config.alert_cooldown_seconds == 600.0


class TestMonitoringClient:
    """Test suite for MonitoringClient class."""

    @pytest.fixture
    def config(self):
        """Create a MonitoringConfig instance."""
        return MonitoringConfig()

    @pytest.fixture
    def client(self, config):
        """Create a MonitoringClient instance."""
        return MonitoringClient(config=config)

    @pytest.mark.unit
    def test_client_creation(self, client):
        """Test creating MonitoringClient."""
        assert client is not None
        assert client.config is not None
        assert client.alert_system is not None
        assert client.threshold_manager is not None
        assert client.health_monitor is not None
        assert client.notification_channels is not None
        assert client._is_monitoring is False

    @pytest.mark.unit
    def test_client_creation_default_config(self):
        """Test creating MonitoringClient with default config."""
        client = MonitoringClient()
        assert client.config is not None
        assert isinstance(client.config, MonitoringConfig)

    @pytest.mark.unit
    @patch("am_qadf.monitoring.health_monitor.HealthMonitor")
    def test_start_monitoring(self, mock_health_monitor, client):
        """Test starting monitoring."""
        mock_health_monitor_instance = MagicMock()
        mock_health_monitor.return_value = mock_health_monitor_instance
        client.health_monitor = mock_health_monitor_instance

        client.start_monitoring()

        assert client._is_monitoring is True
        mock_health_monitor_instance.start_monitoring.assert_called_once()
        assert client._monitoring_thread is not None

        # Cleanup
        client.stop_monitoring()

    @pytest.mark.unit
    def test_start_monitoring_already_running(self, client):
        """Test starting monitoring when already running."""
        client._is_monitoring = True

        # Should log warning but not raise error
        client.start_monitoring()

    @pytest.mark.unit
    @patch("am_qadf.monitoring.health_monitor.HealthMonitor")
    def test_stop_monitoring(self, mock_health_monitor, client):
        """Test stopping monitoring."""
        mock_health_monitor_instance = MagicMock()
        mock_health_monitor.return_value = mock_health_monitor_instance
        client.health_monitor = mock_health_monitor_instance

        client._is_monitoring = True
        client.stop_monitoring()

        assert client._is_monitoring is False
        mock_health_monitor_instance.stop_monitoring.assert_called_once()

    @pytest.mark.unit
    def test_stop_monitoring_not_running(self, client):
        """Test stopping monitoring when not running."""
        # Should log warning but not raise error
        client.stop_monitoring()

    @pytest.mark.unit
    def test_register_metric(self, client):
        """Test registering a metric."""
        threshold_config = ThresholdConfig(
            metric_name="temperature",
            threshold_type="absolute",
            lower_threshold=0.0,
            upper_threshold=100.0,
        )

        client.register_metric("temperature", threshold_config)

        # Check metric is registered
        thresholds = client.threshold_manager.get_current_thresholds()
        assert "temperature" in thresholds

    @pytest.mark.unit
    def test_update_metric(self, client):
        """Test updating metric value."""
        threshold_config = ThresholdConfig(
            metric_name="temperature",
            threshold_type="absolute",
            upper_threshold=100.0,
        )
        client.register_metric("temperature", threshold_config)

        client.update_metric("temperature", 50.0)

        metrics = client.get_current_metrics()
        assert "temperature" in metrics
        assert metrics["temperature"] == 50.0

    @pytest.mark.unit
    def test_update_metric_not_registered(self, client):
        """Test updating metric that's not registered."""
        # Should auto-register with default threshold
        client.update_metric("temperature", 50.0)

        metrics = client.get_current_metrics()
        assert "temperature" in metrics
        assert metrics["temperature"] == 50.0

    @pytest.mark.unit
    def test_update_metric_with_timestamp(self, client):
        """Test updating metric with custom timestamp."""
        threshold_config = ThresholdConfig(
            metric_name="temperature", threshold_type="absolute", lower_threshold=-float("inf"), upper_threshold=float("inf")
        )
        client.register_metric("temperature", threshold_config)

        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        client.update_metric("temperature", 50.0, timestamp)

        metrics = client.get_current_metrics()
        assert metrics["temperature"] == 50.0

    @pytest.mark.unit
    def test_get_current_metrics(self, client):
        """Test getting current metrics."""
        # Register and update metrics
        threshold_config = ThresholdConfig(
            metric_name="temperature", threshold_type="absolute", lower_threshold=-float("inf"), upper_threshold=float("inf")
        )
        client.register_metric("temperature", threshold_config)
        client.update_metric("temperature", 50.0)

        threshold_config2 = ThresholdConfig(
            metric_name="pressure", threshold_type="absolute", lower_threshold=-float("inf"), upper_threshold=float("inf")
        )
        client.register_metric("pressure", threshold_config2)
        client.update_metric("pressure", 100.0)

        metrics = client.get_current_metrics()

        assert "temperature" in metrics
        assert "pressure" in metrics
        assert metrics["temperature"] == 50.0
        assert metrics["pressure"] == 100.0

    @pytest.mark.unit
    def test_get_current_metrics_empty(self, client):
        """Test getting current metrics when none exist."""
        metrics = client.get_current_metrics()
        assert metrics == {}

    @pytest.mark.unit
    @patch("am_qadf.monitoring.health_monitor.HealthMonitor")
    def test_get_health_status(self, mock_health_monitor, client):
        """Test getting health status."""
        mock_health_monitor_instance = MagicMock()
        mock_health_monitor.return_value = mock_health_monitor_instance
        client.health_monitor = mock_health_monitor_instance

        # Mock health status
        mock_health_status = MagicMock()
        mock_health_monitor_instance.check_system_health.return_value = mock_health_status
        mock_health_monitor_instance.check_process_health.return_value = mock_health_status
        mock_health_monitor_instance.get_all_component_health.return_value = {}

        # Get all health status
        health_statuses = client.get_health_status()

        assert "system" in health_statuses

    @pytest.mark.unit
    @patch("am_qadf.monitoring.health_monitor.HealthMonitor")
    def test_get_health_status_component(self, mock_health_monitor, client):
        """Test getting health status for specific component."""
        mock_health_monitor_instance = MagicMock()
        mock_health_monitor.return_value = mock_health_monitor_instance
        client.health_monitor = mock_health_monitor_instance

        mock_health_status = MagicMock()
        mock_health_monitor_instance.check_process_health.return_value = mock_health_status

        health_statuses = client.get_health_status("component1")

        assert "component1" in health_statuses

    @pytest.mark.unit
    def test_acknowledge_alert(self, client):
        """Test acknowledging an alert."""
        # Generate an alert
        alert = client.alert_system.generate_alert(
            alert_type="test_alert", severity="medium", message="Test message", source="test"
        )

        assert alert is not None
        alert_id = alert.alert_id

        # Acknowledge alert
        client.acknowledge_alert(alert_id, "test_user")

        # Check alert is acknowledged
        active_alerts = client.alert_system.get_active_alerts()
        assert alert_id not in [a.alert_id for a in active_alerts]
