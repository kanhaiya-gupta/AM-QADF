"""
Unit tests for HealthMonitor.

Tests for system and process health monitoring with mocks.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta

from am_qadf.monitoring.health_monitor import (
    HealthMonitor,
    HealthStatus,
)


class TestHealthStatus:
    """Test suite for HealthStatus dataclass."""

    @pytest.mark.unit
    def test_health_status_creation(self):
        """Test creating HealthStatus."""
        status = HealthStatus(
            component_name="system",
            status="healthy",
            health_score=0.9,
            timestamp=datetime.now(),
            metrics={"cpu_percent": 50.0, "memory_percent": 60.0},
            issues=[],
            metadata={"check_type": "system"},
        )

        assert status.component_name == "system"
        assert status.status == "healthy"
        assert status.health_score == 0.9
        assert status.metrics["cpu_percent"] == 50.0
        assert len(status.issues) == 0

    @pytest.mark.unit
    def test_health_status_invalid_status(self):
        """Test creating HealthStatus with invalid status."""
        with pytest.raises(ValueError, match="Invalid status"):
            HealthStatus(component_name="test", status="invalid", health_score=0.5, timestamp=datetime.now())

    @pytest.mark.unit
    def test_health_status_invalid_score(self):
        """Test creating HealthStatus with invalid score."""
        with pytest.raises(ValueError, match="Health score must be between"):
            HealthStatus(component_name="test", status="healthy", health_score=1.5, timestamp=datetime.now())  # Invalid: > 1.0


class TestHealthMonitor:
    """Test suite for HealthMonitor class."""

    @pytest.fixture
    def monitor(self):
        """Create a HealthMonitor instance."""
        return HealthMonitor(check_interval_seconds=1.0)

    @pytest.mark.unit
    def test_monitor_creation(self, monitor):
        """Test creating HealthMonitor."""
        assert monitor is not None
        assert monitor.check_interval_seconds == 1.0
        assert monitor._is_monitoring is False
        assert len(monitor._components) == 0

    @pytest.mark.unit
    @patch("am_qadf.monitoring.health_monitor.PSUTIL_AVAILABLE", True)
    @patch("am_qadf.monitoring.health_monitor.psutil")
    def test_check_system_health(self, mock_psutil, monitor):
        """Test checking system health."""
        # Mock psutil functions
        mock_psutil.cpu_percent.return_value = 50.0
        mock_psutil.virtual_memory.return_value = MagicMock(
            percent=60.0, available=8 * 1024**3, total=16 * 1024**3  # 8GB  # 16GB
        )
        mock_psutil.disk_usage.return_value = MagicMock(
            percent=70.0, free=100 * 1024**3, total=500 * 1024**3  # 100GB  # 500GB
        )
        mock_psutil.net_io_counters.return_value = MagicMock(
            bytes_sent=1000000, bytes_recv=2000000, packets_sent=1000, packets_recv=2000
        )

        health_status = monitor.check_system_health()

        assert health_status is not None
        assert health_status.component_name == "system"
        assert health_status.status in ["healthy", "degraded", "unhealthy", "critical"]
        assert 0.0 <= health_status.health_score <= 1.0
        assert "cpu_percent" in health_status.metrics
        assert "memory_percent" in health_status.metrics
        assert "disk_percent" in health_status.metrics

    @pytest.mark.unit
    @patch("am_qadf.monitoring.health_monitor.PSUTIL_AVAILABLE", False)
    def test_check_system_health_no_psutil(self, monitor):
        """Test checking system health when psutil not available."""
        health_status = monitor.check_system_health()

        assert health_status is not None
        assert health_status.status == "unhealthy"
        assert health_status.health_score == 0.0
        assert "psutil not available" in health_status.issues[0]

    @pytest.mark.unit
    def test_register_component(self, monitor):
        """Test registering component for health monitoring."""

        def health_checker():
            return {"cpu_percent": 50.0, "memory_percent": 60.0}

        monitor.register_component("component1", health_checker)

        assert "component1" in monitor._components
        assert monitor._components["component1"] == health_checker

    @pytest.mark.unit
    def test_check_process_health_with_checker(self, monitor):
        """Test checking process health with custom checker."""

        def health_checker():
            return {"cpu_percent": 50.0, "error_rate": 0.01}

        monitor.register_component("component1", health_checker)

        health_status = monitor.check_process_health("component1")

        assert health_status is not None
        assert health_status.component_name == "component1"
        assert health_status.metrics["cpu_percent"] == 50.0
        assert health_status.metrics["error_rate"] == 0.01
        assert 0.0 <= health_status.health_score <= 1.0

    @pytest.mark.unit
    def test_check_process_health_no_checker(self, monitor):
        """Test checking process health without custom checker."""
        health_status = monitor.check_process_health("nonexistent_component")

        assert health_status is not None
        assert health_status.component_name == "nonexistent_component"
        # Should have default behavior
        assert health_status.status == "healthy"

    @pytest.mark.unit
    def test_start_monitoring(self, monitor):
        """Test starting health monitoring."""
        monitor.start_monitoring()

        assert monitor._is_monitoring is True
        assert monitor._monitoring_thread is not None
        assert monitor._monitoring_thread.is_alive()

        # Cleanup
        monitor.stop_monitoring()

    @pytest.mark.unit
    def test_start_monitoring_already_running(self, monitor):
        """Test starting monitoring when already running."""
        monitor._is_monitoring = True

        # Should log warning but not raise error
        monitor.start_monitoring()

    @pytest.mark.unit
    def test_stop_monitoring(self, monitor):
        """Test stopping health monitoring."""
        monitor.start_monitoring()
        assert monitor._is_monitoring is True

        monitor.stop_monitoring()

        assert monitor._is_monitoring is False

    @pytest.mark.unit
    def test_stop_monitoring_not_running(self, monitor):
        """Test stopping monitoring when not running."""
        # Should log warning but not raise error
        monitor.stop_monitoring()

    @pytest.mark.unit
    def test_get_health_history(self, monitor):
        """Test getting health history."""
        # Record some health statuses
        status1 = HealthStatus(
            component_name="component1", status="healthy", health_score=0.9, timestamp=datetime.now() - timedelta(hours=2)
        )
        status2 = HealthStatus(
            component_name="component1", status="healthy", health_score=0.8, timestamp=datetime.now() - timedelta(hours=1)
        )

        monitor._health_history["component1"] = [status1, status2]

        start_time = datetime.now() - timedelta(hours=3)
        end_time = datetime.now()

        history = monitor.get_health_history("component1", start_time, end_time)

        assert len(history) == 2

    @pytest.mark.unit
    def test_get_all_component_health(self, monitor):
        """Test getting health for all components."""

        def health_checker():
            return {"metric1": 50.0}

        monitor.register_component("component1", health_checker)

        all_health = monitor.get_all_component_health()

        assert "system" in all_health
        assert "component1" in all_health

    @pytest.mark.unit
    def test_calculate_health_score(self, monitor):
        """Test calculating health score."""
        metrics = {
            "cpu_percent": 50.0,
            "memory_percent": 60.0,
            "disk_percent": 70.0,
        }

        score = monitor.calculate_health_score(metrics)

        assert 0.0 <= score <= 1.0

    @pytest.mark.unit
    def test_calculate_health_score_with_weights(self, monitor):
        """Test calculating health score with weights."""
        metrics = {
            "cpu_percent": 50.0,
            "memory_percent": 60.0,
            "disk_percent": 70.0,
        }
        weights = {
            "cpu_percent": 0.5,
            "memory_percent": 0.3,
            "disk_percent": 0.2,
        }

        score = monitor.calculate_health_score(metrics, weights)

        assert 0.0 <= score <= 1.0
