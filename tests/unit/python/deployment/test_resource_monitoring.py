"""
Unit tests for ResourceMonitor.

Tests for system and process resource monitoring.
"""

import pytest
import time
import os
import psutil
from datetime import datetime

from am_qadf.deployment.resource_monitoring import ResourceMetrics, ResourceMonitor


class TestResourceMetrics:
    """Test suite for ResourceMetrics dataclass."""

    @pytest.mark.unit
    def test_resource_metrics_creation(self):
        """Test creating ResourceMetrics."""
        timestamp = datetime.now()
        metrics = ResourceMetrics(
            timestamp=timestamp,
            cpu_percent=50.5,
            memory_percent=60.0,
            memory_used_mb=1024.0,
            memory_available_mb=4096.0,
            disk_percent=70.0,
            disk_used_gb=500.0,
            disk_available_gb=200.0,
            network_sent_mb=10.5,
            network_recv_mb=5.2,
            process_count=100,
            thread_count=200,
        )

        assert metrics.timestamp == timestamp
        assert metrics.cpu_percent == 50.5
        assert metrics.memory_percent == 60.0
        assert metrics.memory_used_mb == 1024.0
        assert metrics.memory_available_mb == 4096.0
        assert metrics.disk_percent == 70.0
        assert metrics.disk_used_gb == 500.0
        assert metrics.disk_available_gb == 200.0
        assert metrics.network_sent_mb == 10.5
        assert metrics.network_recv_mb == 5.2
        assert metrics.process_count == 100
        assert metrics.thread_count == 200

    @pytest.mark.unit
    def test_resource_metrics_to_dict(self):
        """Test converting ResourceMetrics to dictionary."""
        timestamp = datetime.now()
        metrics = ResourceMetrics(
            timestamp=timestamp,
            cpu_percent=50.0,
            memory_percent=60.0,
            memory_used_mb=1024.0,
            memory_available_mb=4096.0,
            disk_percent=70.0,
            disk_used_gb=500.0,
            disk_available_gb=200.0,
            network_sent_mb=10.0,
            network_recv_mb=5.0,
            process_count=100,
            thread_count=200,
        )

        metrics_dict = metrics.to_dict()

        assert isinstance(metrics_dict, dict)
        assert metrics_dict["cpu_percent"] == 50.0
        assert metrics_dict["memory_percent"] == 60.0
        assert "timestamp" in metrics_dict
        assert isinstance(metrics_dict["timestamp"], str)


class TestResourceMonitor:
    """Test suite for ResourceMonitor class."""

    @pytest.mark.unit
    def test_resource_monitor_init(self):
        """Test ResourceMonitor initialization."""
        monitor = ResourceMonitor(update_interval=5.0, history_size=100)

        assert monitor.update_interval == 5.0
        assert monitor.history_size == 100
        assert monitor.monitoring_active is False
        assert len(monitor.metrics_history) == 0

    @pytest.mark.unit
    def test_get_system_metrics(self):
        """Test getting system metrics."""
        monitor = ResourceMonitor()
        metrics = monitor.get_system_metrics()

        assert isinstance(metrics, ResourceMetrics)
        assert 0 <= metrics.cpu_percent <= 100
        assert 0 <= metrics.memory_percent <= 100
        assert 0 <= metrics.disk_percent <= 100
        assert metrics.process_count >= 0
        assert metrics.thread_count >= 0
        assert len(monitor.metrics_history) == 1

    @pytest.mark.unit
    def test_get_process_metrics(self):
        """Test getting process metrics."""
        monitor = ResourceMonitor()
        current_pid = os.getpid()
        metrics = monitor.get_process_metrics(pid=current_pid)

        assert isinstance(metrics, ResourceMetrics)
        assert 0 <= metrics.cpu_percent <= 100
        assert 0 <= metrics.memory_percent <= 100
        assert metrics.thread_count >= 0

    @pytest.mark.unit
    def test_get_process_metrics_process_not_found(self):
        """Test getting process metrics for non-existent process."""
        monitor = ResourceMonitor()
        # Use a very high PID that likely doesn't exist
        # Should fallback to system metrics if process not found
        metrics = monitor.get_process_metrics(pid=99999999)
        assert isinstance(metrics, ResourceMetrics)

    @pytest.mark.unit
    def test_start_stop_monitoring(self):
        """Test starting and stopping monitoring."""
        monitor = ResourceMonitor(update_interval=0.1)
        callback_called = []

        def callback(metrics):
            callback_called.append(metrics)

        monitor.start_monitoring(callback)
        assert monitor.monitoring_active is True

        time.sleep(0.25)  # Wait for at least 2 callbacks

        monitor.stop_monitoring()
        assert monitor.monitoring_active is False

        assert len(callback_called) >= 1

    @pytest.mark.unit
    def test_start_monitoring_already_active(self):
        """Test starting monitoring when already active."""
        monitor = ResourceMonitor()

        def callback(metrics):
            pass

        monitor.start_monitoring(callback)

        with pytest.raises(RuntimeError, match="already active"):
            monitor.start_monitoring(callback)

        monitor.stop_monitoring()

    @pytest.mark.unit
    def test_get_metrics_history(self):
        """Test getting metrics history."""
        monitor = ResourceMonitor(history_size=10)

        # Generate some metrics
        for _ in range(5):
            monitor.get_system_metrics()
            time.sleep(0.01)

        history = monitor.get_metrics_history()
        assert len(history) == 5

        # Test with duration
        history_recent = monitor.get_metrics_history(duration=1.0)
        assert len(history_recent) <= 5

    @pytest.mark.unit
    def test_check_resource_limits(self):
        """Test checking resource limits."""
        monitor = ResourceMonitor()

        # Get current metrics and check with appropriate thresholds
        metrics = monitor.get_system_metrics()

        # Use very high threshold to test low usage path
        exceeded, warnings = monitor.check_resource_limits(cpu_threshold=1.0, memory_threshold=1.0, disk_threshold=1.0)

        # Should not exceed with threshold of 100%
        assert exceeded is False

        # Test with very low thresholds to trigger warnings
        exceeded, warnings = monitor.check_resource_limits(cpu_threshold=0.01, memory_threshold=0.01, disk_threshold=0.01)

        # May or may not exceed depending on actual system state
        assert isinstance(exceeded, bool)
        assert isinstance(warnings, list)
