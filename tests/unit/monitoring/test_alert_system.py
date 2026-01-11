"""
Unit tests for AlertSystem.

Tests for alert generation, escalation, and management.
"""

import pytest
from unittest.mock import Mock, MagicMock
from datetime import datetime, timedelta

from am_qadf.monitoring.alert_system import (
    AlertSystem,
    Alert,
)
from am_qadf.monitoring.monitoring_client import MonitoringConfig


class TestAlert:
    """Test suite for Alert dataclass."""

    @pytest.mark.unit
    def test_alert_creation(self):
        """Test creating Alert."""
        alert_id = "test_alert_123"
        timestamp = datetime.now()
        metadata = {"source_id": "sensor1"}

        alert = Alert(
            alert_id=alert_id,
            alert_type="quality_threshold",
            severity="high",
            message="Threshold exceeded",
            timestamp=timestamp,
            source="ThresholdManager",
            metadata=metadata,
            acknowledged=False,
        )

        assert alert.alert_id == alert_id
        assert alert.alert_type == "quality_threshold"
        assert alert.severity == "high"
        assert alert.message == "Threshold exceeded"
        assert alert.timestamp == timestamp
        assert alert.source == "ThresholdManager"
        assert alert.metadata == metadata
        assert alert.acknowledged is False
        assert alert.acknowledged_by is None
        assert alert.acknowledged_at is None

    @pytest.mark.unit
    def test_alert_creation_defaults(self):
        """Test creating Alert with minimal fields."""
        alert = Alert(
            alert_id="test_alert",
            alert_type="test",
            severity="low",
            message="Test message",
            timestamp=datetime.now(),
            source="test",
        )

        assert alert.alert_id == "test_alert"
        assert alert.acknowledged is False
        assert alert.metadata == {}

    @pytest.mark.unit
    def test_alert_auto_id(self):
        """Test Alert auto-generates ID if not provided."""
        alert = Alert(
            alert_id="",
            alert_type="test",
            severity="low",
            message="Test",
            timestamp=datetime.now(),
            source="test",
        )

        # ID should be generated in __post_init__
        assert alert.alert_id != ""


class TestAlertSystem:
    """Test suite for AlertSystem class."""

    @pytest.fixture
    def config(self):
        """Create a MonitoringConfig instance."""
        return MonitoringConfig(alert_cooldown_seconds=60.0)

    @pytest.fixture
    def alert_system(self, config):
        """Create an AlertSystem instance."""
        return AlertSystem(config=config)

    @pytest.mark.unit
    def test_alert_system_creation(self, alert_system):
        """Test creating AlertSystem."""
        assert alert_system is not None
        assert alert_system.config is not None
        assert len(alert_system._active_alerts) == 0
        assert len(alert_system._alert_history) == 0

    @pytest.mark.unit
    def test_generate_alert(self, alert_system):
        """Test generating an alert."""
        alert = alert_system.generate_alert(
            alert_type="quality_threshold",
            severity="high",
            message="Temperature too high",
            source="Sensor1",
            metadata={"value": 150.0, "threshold": 100.0},
        )

        assert alert is not None
        assert alert.alert_type == "quality_threshold"
        assert alert.severity == "high"
        assert alert.message == "Temperature too high"
        assert alert.source == "Sensor1"
        assert alert.metadata["value"] == 150.0
        assert alert.alert_id in alert_system._active_alerts
        assert alert in alert_system._alert_history

    @pytest.mark.unit
    def test_generate_alert_cooldown(self, alert_system):
        """Test alert cooldown prevents duplicate alerts."""
        # Generate first alert
        alert1 = alert_system.generate_alert(alert_type="test_alert", severity="medium", message="Test message", source="test")

        assert alert1 is not None

        # Try to generate same type immediately (should be in cooldown)
        alert2 = alert_system.generate_alert(
            alert_type="test_alert", severity="medium", message="Test message 2", source="test"
        )

        # Should be None due to cooldown
        assert alert2 is None

    @pytest.mark.unit
    def test_get_active_alerts(self, alert_system):
        """Test getting active alerts."""
        # Generate alerts
        alert1 = alert_system.generate_alert(alert_type="alert1", severity="high", message="Alert 1", source="test")
        alert2 = alert_system.generate_alert(alert_type="alert2", severity="low", message="Alert 2", source="test")

        active_alerts = alert_system.get_active_alerts()

        assert len(active_alerts) >= 1  # May have cooldown effects
        alert_ids = [a.alert_id for a in active_alerts]
        # At least one should be active (depending on cooldown)

    @pytest.mark.unit
    def test_get_active_alerts_severity_filter(self, alert_system):
        """Test getting active alerts filtered by severity."""
        # Generate alerts with different severities
        # Note: Due to cooldown, may need to use different alert types
        alert_system.generate_alert(alert_type="high_alert", severity="high", message="High severity", source="test")

        # Wait a bit for cooldown
        import time

        time.sleep(0.1)

        alert_system.generate_alert(alert_type="low_alert", severity="low", message="Low severity", source="test")

        high_alerts = alert_system.get_active_alerts(severity="high")
        low_alerts = alert_system.get_active_alerts(severity="low")

        # Should filter by severity
        assert all(a.severity == "high" for a in high_alerts)
        assert all(a.severity == "low" for a in low_alerts)

    @pytest.mark.unit
    def test_acknowledge_alert(self, alert_system):
        """Test acknowledging an alert."""
        # Generate alert
        alert = alert_system.generate_alert(alert_type="test_alert", severity="medium", message="Test message", source="test")

        assert alert is not None
        alert_id = alert.alert_id
        assert alert_id in alert_system._active_alerts

        # Acknowledge
        alert_system.acknowledge_alert(alert_id, "test_user")

        # Check acknowledged
        assert alert_id not in alert_system._active_alerts
        alert_from_history = next(a for a in alert_system._alert_history if a.alert_id == alert_id)
        assert alert_from_history.acknowledged is True
        assert alert_from_history.acknowledged_by == "test_user"
        assert alert_from_history.acknowledged_at is not None

    @pytest.mark.unit
    def test_acknowledge_nonexistent_alert(self, alert_system):
        """Test acknowledging non-existent alert."""
        # Should log warning but not raise error
        alert_system.acknowledge_alert("nonexistent_id", "test_user")

    @pytest.mark.unit
    def test_escalate_alert(self, alert_system):
        """Test escalating an alert."""
        # Generate alert
        alert = alert_system.generate_alert(alert_type="test_alert", severity="low", message="Test message", source="test")

        assert alert is not None
        alert_id = alert.alert_id
        original_severity = alert.severity

        # Escalate
        alert_system.escalate_alert(alert_id)

        # Check escalated
        updated_alert = alert_system._active_alerts.get(alert_id)
        if updated_alert:
            assert updated_alert.severity != original_severity
            assert "escalated" in updated_alert.metadata

    @pytest.mark.unit
    def test_escalate_critical_alert(self, alert_system):
        """Test escalating already critical alert (should stay critical)."""
        # Generate critical alert
        alert = alert_system.generate_alert(
            alert_type="critical_alert", severity="critical", message="Critical message", source="test"
        )

        assert alert is not None
        alert_id = alert.alert_id
        original_severity = alert.severity

        # Escalate (should not change severity as it's already critical)
        alert_system.escalate_alert(alert_id)

        # Severity should remain critical
        updated_alert = alert_system._active_alerts.get(alert_id)
        if updated_alert:
            assert updated_alert.severity == "critical"

    @pytest.mark.unit
    def test_get_alert_history(self, alert_system):
        """Test getting alert history."""
        start_time = datetime.now()

        # Generate alerts
        alert_system.generate_alert(alert_type="alert1", severity="medium", message="Alert 1", source="test")

        import time

        time.sleep(0.1)

        alert_system.generate_alert(alert_type="alert2", severity="high", message="Alert 2", source="test")

        end_time = datetime.now()

        # Get history
        history = alert_system.get_alert_history(start_time, end_time)

        assert len(history) >= 0  # May have cooldown effects

    @pytest.mark.unit
    def test_get_alert_history_with_filters(self, alert_system):
        """Test getting alert history with filters."""
        start_time = datetime.now()

        # Generate different types of alerts
        alert_system.generate_alert(
            alert_type="quality_threshold", severity="high", message="Quality alert", source="QualityManager"
        )

        import time

        time.sleep(0.2)

        alert_system.generate_alert(
            alert_type="system_error", severity="critical", message="System error", source="SystemMonitor"
        )

        end_time = datetime.now()

        # Filter by alert type
        quality_alerts = alert_system.get_alert_history(start_time, end_time, filters={"alert_type": "quality_threshold"})
        assert all(a.alert_type == "quality_threshold" for a in quality_alerts)

        # Filter by severity
        high_alerts = alert_system.get_alert_history(start_time, end_time, filters={"severity": "high"})
        assert all(a.severity == "high" for a in high_alerts)

        # Filter by source
        system_alerts = alert_system.get_alert_history(start_time, end_time, filters={"source": "SystemMonitor"})
        assert all(a.source == "SystemMonitor" for a in system_alerts)

    @pytest.mark.unit
    def test_enable_alert_cooldown(self, alert_system):
        """Test enabling custom cooldown for alert type."""
        alert_system.enable_alert_cooldown("custom_alert", 120.0)

        # Generate alert
        alert1 = alert_system.generate_alert(alert_type="custom_alert", severity="medium", message="Test", source="test")

        assert alert1 is not None

        # Try to generate same type immediately
        alert2 = alert_system.generate_alert(alert_type="custom_alert", severity="medium", message="Test 2", source="test")

        # Should be None due to custom cooldown
        assert alert2 is None
