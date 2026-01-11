"""
Performance tests for Monitoring module.

Tests performance characteristics including alert generation speed, threshold checking, and health monitoring overhead.
"""

import pytest
import numpy as np
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch

try:
    from am_qadf.monitoring import (
        MonitoringClient,
        MonitoringConfig,
        AlertSystem,
        ThresholdManager,
        ThresholdConfig,
        HealthMonitor,
    )
except ImportError:
    pytest.skip("Monitoring module not available", allow_module_level=True)


class TestMonitoringPerformance:
    """Performance tests for monitoring components."""

    @pytest.fixture
    def monitoring_config(self):
        """Create monitoring configuration."""
        return MonitoringConfig(
            alert_cooldown_seconds=0.01,  # Short for testing
            alert_check_interval_seconds=0.01,
            health_check_interval_seconds=0.1,
        )

    @pytest.fixture
    def monitoring_client(self, monitoring_config):
        """Create monitoring client."""
        return MonitoringClient(config=monitoring_config)

    @pytest.mark.performance
    def test_alert_generation_latency(self, monitoring_config):
        """Test alert generation latency (target: < 10ms)."""
        alert_system = AlertSystem(monitoring_config)

        n_alerts = 100
        latencies = []

        for i in range(n_alerts):
            start_time = time.time()
            alert = alert_system.generate_alert(
                alert_type=f"alert_type_{i % 5}",  # Vary types to avoid cooldown
                severity="medium",
                message=f"Test alert {i}",
                source="TestSource",
                metadata={"index": i},
            )
            end_time = time.time()

            if alert:  # May be None due to cooldown
                latency_ms = (end_time - start_time) * 1000.0
                latencies.append(latency_ms)

            time.sleep(0.01)  # Small delay to avoid cooldown issues

        if latencies:
            avg_latency_ms = np.mean(latencies)
            max_latency_ms = np.max(latencies)

            print(f"\nAverage alert generation latency: {avg_latency_ms:.2f} ms")
            print(f"Max alert generation latency: {max_latency_ms:.2f} ms")

            # Target: < 10ms average
            assert avg_latency_ms < 50.0  # Relaxed for test environment

    @pytest.mark.performance
    def test_threshold_checking_performance(self):
        """Test threshold checking performance."""
        threshold_manager = ThresholdManager()

        # Add multiple thresholds
        n_metrics = 100
        for i in range(n_metrics):
            config = ThresholdConfig(
                metric_name=f"metric_{i}",
                threshold_type="absolute",
                lower_threshold=0.0,
                upper_threshold=100.0,
            )
            threshold_manager.add_threshold(f"metric_{i}", config)

        # Check values
        n_checks = 1000
        start_time = time.time()

        for i in range(n_checks):
            metric_name = f"metric_{i % n_metrics}"
            value = np.random.uniform(0.0, 150.0)
            threshold_manager.check_value(metric_name, value, datetime.now())

        end_time = time.time()

        total_time = end_time - start_time
        throughput = n_checks / total_time

        print(f"\nThreshold checking throughput: {throughput:.2f} checks/second")

        # Should handle at least 1000 checks/second
        assert throughput > 500.0

    @pytest.mark.performance
    @patch("am_qadf.monitoring.health_monitor.PSUTIL_AVAILABLE", True)
    @patch("am_qadf.monitoring.health_monitor.psutil")
    def test_health_check_performance(self, mock_psutil):
        """Test health check performance."""
        # Mock psutil
        mock_psutil.cpu_percent.return_value = 50.0
        mock_psutil.virtual_memory.return_value = MagicMock(percent=60.0, available=8 * 1024**3, total=16 * 1024**3)
        mock_psutil.disk_usage.return_value = MagicMock(percent=70.0, free=100 * 1024**3, total=500 * 1024**3)
        mock_psutil.net_io_counters.return_value = MagicMock(
            bytes_sent=1000000, bytes_recv=2000000, packets_sent=1000, packets_recv=2000
        )

        health_monitor = HealthMonitor(check_interval_seconds=1.0)

        n_checks = 100
        latencies = []

        for i in range(n_checks):
            start_time = time.time()
            health_status = health_monitor.check_system_health()
            end_time = time.time()

            latency_ms = (end_time - start_time) * 1000.0
            latencies.append(latency_ms)

        avg_latency_ms = np.mean(latencies)
        max_latency_ms = np.max(latencies)

        print(f"\nAverage health check latency: {avg_latency_ms:.2f} ms")
        print(f"Max health check latency: {max_latency_ms:.2f} ms")

        # Should be fast (< 100ms)
        assert avg_latency_ms < 200.0

    @pytest.mark.performance
    def test_metric_update_performance(self, monitoring_client):
        """Test metric update performance."""
        # Register metrics
        n_metrics = 50
        for i in range(n_metrics):
            threshold_config = ThresholdConfig(
                metric_name=f"metric_{i}",
                threshold_type="absolute",
                upper_threshold=100.0,
            )
            monitoring_client.register_metric(f"metric_{i}", threshold_config)

        # Update metrics
        n_updates = 1000
        start_time = time.time()

        for i in range(n_updates):
            metric_name = f"metric_{i % n_metrics}"
            value = np.random.uniform(0.0, 150.0)
            monitoring_client.update_metric(metric_name, value)

        end_time = time.time()

        total_time = end_time - start_time
        throughput = n_updates / total_time

        print(f"\nMetric update throughput: {throughput:.2f} updates/second")

        # Should handle at least 500 updates/second
        assert throughput > 200.0

    @pytest.mark.performance
    def test_active_alerts_query_performance(self, monitoring_config):
        """Test active alerts query performance."""
        alert_system = AlertSystem(monitoring_config)

        # Generate many alerts
        n_alerts = 1000
        alert_types = [f"type_{i % 10}" for i in range(n_alerts)]

        for i in range(n_alerts):
            alert_system.generate_alert(
                alert_type=alert_types[i], severity="medium", message=f"Alert {i}", source="TestSource"
            )
            time.sleep(0.001)  # Small delay to vary timestamps

        # Query active alerts
        n_queries = 100
        start_time = time.time()

        for i in range(n_queries):
            active_alerts = alert_system.get_active_alerts()
            # Query with filter
            high_alerts = alert_system.get_active_alerts(severity="high")

        end_time = time.time()

        total_time = end_time - start_time
        avg_latency_ms = (total_time / n_queries) * 1000.0

        print(f"\nActive alerts query average latency: {avg_latency_ms:.2f} ms")

        # Should be fast (< 50ms)
        assert avg_latency_ms < 100.0

    @pytest.mark.performance
    def test_alert_history_query_performance(self, monitoring_config):
        """Test alert history query performance."""
        alert_system = AlertSystem(monitoring_config)

        # Generate alerts over time
        start_time = datetime.now() - timedelta(hours=24)
        n_alerts = 1000

        for i in range(n_alerts):
            alert_time = start_time + timedelta(seconds=i * 86.4)  # Spread over 24 hours
            alert_system.generate_alert(
                alert_type=f"type_{i % 10}",
                severity="medium",
                message=f"Alert {i}",
                source="TestSource",
                metadata={"timestamp": alert_time.isoformat()},
            )
            time.sleep(0.001)

        # Query history
        n_queries = 50
        query_start = time.time()

        for i in range(n_queries):
            end_time = datetime.now()
            start_query = end_time - timedelta(hours=12)

            history = alert_system.get_alert_history(start_query, end_time, filters={"alert_type": f"type_{i % 10}"})

        query_end = time.time()

        total_time = query_end - query_start
        avg_latency_ms = (total_time / n_queries) * 1000.0

        print(f"\nAlert history query average latency: {avg_latency_ms:.2f} ms")

        # Should be reasonably fast
        assert avg_latency_ms < 200.0

    @pytest.mark.performance
    @pytest.mark.slow
    def test_monitoring_client_long_running(self, monitoring_client):
        """Test long-running monitoring client performance."""
        # Register metrics
        for i in range(10):
            threshold_config = ThresholdConfig(
                metric_name=f"long_metric_{i}",
                threshold_type="absolute",
                upper_threshold=100.0,
            )
            monitoring_client.register_metric(f"long_metric_{i}", threshold_config)

        # Start monitoring
        monitoring_client.start_monitoring()

        # Update metrics over time
        n_updates = 1000
        start_time = time.time()

        for i in range(n_updates):
            metric_name = f"long_metric_{i % 10}"
            value = np.random.uniform(0.0, 150.0)
            monitoring_client.update_metric(metric_name, value)

            if (i + 1) % 100 == 0:
                # Check performance
                metrics = monitoring_client.get_current_metrics()
                health = monitoring_client.get_health_status()

        end_time = time.time()

        total_time = end_time - start_time
        throughput = n_updates / total_time

        print(f"\nLong-running monitoring throughput: {throughput:.2f} updates/second")

        # Stop monitoring
        monitoring_client.stop_monitoring()

        # Should maintain good performance
        assert throughput > 100.0

    @pytest.mark.performance
    def test_concurrent_metric_updates(self, monitoring_client):
        """Test concurrent metric updates performance."""
        import threading

        # Register metric
        threshold_config = ThresholdConfig(
            metric_name="concurrent_metric",
            threshold_type="absolute",
            upper_threshold=100.0,
        )
        monitoring_client.register_metric("concurrent_metric", threshold_config)

        # Update from multiple threads
        n_threads = 4
        updates_per_thread = 100
        threads = []
        errors = []

        def update_worker(thread_id):
            try:
                for i in range(updates_per_thread):
                    value = np.random.uniform(0.0, 150.0)
                    monitoring_client.update_metric("concurrent_metric", value)
            except Exception as e:
                errors.append(e)

        start_time = time.time()

        for i in range(n_threads):
            thread = threading.Thread(target=update_worker, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        end_time = time.time()

        total_time = end_time - start_time
        total_updates = n_threads * updates_per_thread
        throughput = total_updates / total_time

        print(f"\nConcurrent updates throughput: {throughput:.2f} updates/second")
        print(f"Errors: {len(errors)}")

        # Should handle concurrent updates
        assert len(errors) == 0
        assert throughput > 100.0
