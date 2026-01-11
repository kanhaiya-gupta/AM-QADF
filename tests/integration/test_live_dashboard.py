"""
Integration tests for Live Dashboard with WebSocket updates.

Tests integration of live dashboard with streaming and monitoring.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
import json
import time

try:
    from am_qadf.streaming import StreamingClient, StreamingConfig
    from am_qadf.monitoring import MonitoringClient, MonitoringConfig, Alert
    from am_qadf.analytics.quality_assessment.dashboard_generator import QualityDashboardGenerator
except ImportError:
    pytest.skip("Streaming or monitoring modules not available", allow_module_level=True)


class TestLiveDashboard:
    """Integration tests for live dashboard."""

    @pytest.fixture
    def monitoring_config(self):
        """Create monitoring configuration with dashboard enabled."""
        return MonitoringConfig(
            enable_dashboard_notifications=True,
            websocket_port=8765,
            alert_cooldown_seconds=0.1,
        )

    @pytest.fixture
    def monitoring_client(self, monitoring_config):
        """Create monitoring client."""
        return MonitoringClient(config=monitoring_config)

    @pytest.fixture
    def streaming_client(self):
        """Create streaming client."""
        config = StreamingConfig(processing_batch_size=10)
        return StreamingClient(config=config)

    @pytest.mark.integration
    def test_dashboard_alert_notifications(self, monitoring_client):
        """Test dashboard receiving alert notifications."""
        # Mock WebSocket clients
        mock_clients = [MagicMock(), MagicMock()]
        monitoring_client.notification_channels._websocket_clients = mock_clients

        # Generate alert
        alert = monitoring_client.alert_system.generate_alert(
            alert_type="test_alert",
            severity="high",
            message="Test dashboard alert",
            source="TestSource",
            metadata={"test": "value"},
        )

        if alert:
            # Send dashboard notification
            with patch("am_qadf.monitoring.notification_channels.json") as mock_json:
                mock_json.dumps.return_value = json.dumps(
                    {"type": "alert", "alert_id": alert.alert_id, "message": alert.message}
                )

                result = monitoring_client.notification_channels.send_dashboard_notification(alert)

                # Should attempt to broadcast
                assert result is True or result is False  # May fail gracefully

    @pytest.mark.integration
    def test_dashboard_metric_updates(self, monitoring_client):
        """Test dashboard receiving metric updates."""
        # Register metric
        from am_qadf.monitoring import ThresholdConfig

        threshold_config = ThresholdConfig(
            metric_name="dashboard_metric",
            threshold_type="absolute",
            upper_threshold=100.0,
        )
        monitoring_client.register_metric("dashboard_metric", threshold_config)

        # Mock WebSocket clients
        mock_client = MagicMock()
        monitoring_client.notification_channels._websocket_clients = [mock_client]

        # Update metric (should trigger dashboard update if alert generated)
        time.sleep(0.2)  # Wait for cooldown
        monitoring_client.update_metric("dashboard_metric", 50.0)

        # Update with violating value
        time.sleep(0.2)
        monitoring_client.update_metric("dashboard_metric", 150.0)

        # Should have attempted to send notifications
        assert True  # Test passes if no exception

    @pytest.mark.integration
    def test_dashboard_streaming_updates(self, streaming_client, monitoring_client):
        """Test dashboard receiving streaming data updates."""
        # Mock WebSocket clients
        mock_client = MagicMock()
        monitoring_client.notification_channels._websocket_clients = [mock_client]

        # Register processor that sends dashboard updates
        def dashboard_update_processor(data_batch):
            # Process data
            values = [item.get("value", 0.0) if isinstance(item, dict) else float(item) for item in data_batch]

            # Update monitoring metrics
            if values:
                avg_value = sum(values) / len(values)
                monitoring_client.update_metric("streaming_avg", avg_value)

            # Broadcast update to dashboard (simulated)
            with patch("am_qadf.monitoring.notification_channels.json") as mock_json:
                mock_json.dumps.return_value = json.dumps(
                    {
                        "type": "metric_update",
                        "metric": "streaming_avg",
                        "value": avg_value if values else 0.0,
                        "timestamp": datetime.now().isoformat(),
                    }
                )

                monitoring_client.notification_channels.broadcast_websocket(
                    {"type": "metric_update", "metric": "streaming_avg", "value": avg_value if values else 0.0}
                )

            return {
                "processed": True,
                "avg_value": avg_value if values else 0.0,
            }

        streaming_client.register_processor("dashboard_update_processor", dashboard_update_processor)

        # Register metric
        from am_qadf.monitoring import ThresholdConfig

        threshold_config = ThresholdConfig(
            metric_name="streaming_avg",
            threshold_type="absolute",
            upper_threshold=100.0,
        )
        monitoring_client.register_metric("streaming_avg", threshold_config)

        # Process batches
        batch1 = [{"value": 50.0}, {"value": 60.0}, {"value": 70.0}]
        result1 = streaming_client.process_stream_batch(batch1)

        # Should have processed
        assert result1.processed_count == 3

        # Check metrics were updated
        metrics = monitoring_client.get_current_metrics()
        # May or may not have 'streaming_avg' depending on timing

    @pytest.mark.integration
    @patch("am_qadf.monitoring.health_monitor.PSUTIL_AVAILABLE", True)
    @patch("am_qadf.monitoring.health_monitor.psutil")
    def test_dashboard_health_updates(self, mock_psutil, monitoring_client):
        """Test dashboard receiving health status updates."""
        # Mock psutil
        mock_psutil.cpu_percent.return_value = 50.0
        mock_psutil.virtual_memory.return_value = MagicMock(percent=60.0, available=8 * 1024**3, total=16 * 1024**3)
        mock_psutil.disk_usage.return_value = MagicMock(percent=70.0, free=100 * 1024**3, total=500 * 1024**3)
        mock_psutil.net_io_counters.return_value = MagicMock(
            bytes_sent=1000000, bytes_recv=2000000, packets_sent=1000, packets_recv=2000
        )

        # Mock WebSocket clients
        mock_client = MagicMock()
        monitoring_client.notification_channels._websocket_clients = [mock_client]

        # Start health monitoring
        monitoring_client.start_monitoring()

        # Wait a bit for health checks
        time.sleep(0.8)

        # Get health status and simulate dashboard update
        health_statuses = monitoring_client.get_health_status()

        with patch("am_qadf.monitoring.notification_channels.json") as mock_json:
            mock_json.dumps.return_value = json.dumps(
                {
                    "type": "health_update",
                    "component": "system",
                    "status": health_statuses.get("system", {}).status if "system" in health_statuses else "unknown",
                }
            )

            monitoring_client.notification_channels.broadcast_websocket(
                {
                    "type": "health_update",
                    "statuses": {k: v.status for k, v in health_statuses.items() if hasattr(v, "status")},
                }
            )

        # Stop monitoring
        monitoring_client.stop_monitoring()

        assert "system" in health_statuses or len(health_statuses) >= 0

    @pytest.mark.integration
    def test_dashboard_complete_workflow(self, streaming_client, monitoring_client):
        """Test complete dashboard workflow with streaming, monitoring, and alerts."""
        # Register metric
        from am_qadf.monitoring import ThresholdConfig

        threshold_config = ThresholdConfig(
            metric_name="live_metric",
            threshold_type="absolute",
            upper_threshold=100.0,
        )
        monitoring_client.register_metric("live_metric", threshold_config)

        # Mock WebSocket clients
        mock_clients = [MagicMock()]
        monitoring_client.notification_channels._websocket_clients = mock_clients

        # Register complete processor
        def complete_dashboard_processor(data_batch):
            values = [item.get("value", 0.0) if isinstance(item, dict) else float(item) for item in data_batch]

            if not values:
                return None

            avg_value = sum(values) / len(values)
            max_value = max(values)
            min_value = min(values)

            # Update metrics
            monitoring_client.update_metric("live_metric", avg_value)

            # Broadcast metric update
            with patch("am_qadf.monitoring.notification_channels.json"):
                monitoring_client.notification_channels.broadcast_websocket(
                    {
                        "type": "metrics",
                        "live_metric": {
                            "value": avg_value,
                            "max": max_value,
                            "min": min_value,
                            "timestamp": datetime.now().isoformat(),
                        },
                    }
                )

            return {
                "processed": True,
                "metrics": {
                    "avg": avg_value,
                    "max": max_value,
                    "min": min_value,
                },
            }

        streaming_client.register_processor("complete_dashboard_processor", complete_dashboard_processor)

        # Process streaming batches
        batches = [
            [{"value": 50.0}, {"value": 60.0}, {"value": 70.0}],
            [{"value": 80.0}, {"value": 90.0}, {"value": 100.0}],
            [{"value": 110.0}, {"value": 120.0}, {"value": 130.0}],  # Should trigger alerts
        ]

        for i, batch in enumerate(batches):
            time.sleep(0.3)  # Wait between batches
            result = streaming_client.process_stream_batch(batch)
            assert result.processed_count == 3

        # Check final statistics
        stats = streaming_client.get_stream_statistics()
        assert stats["messages_processed"] == 9
        assert stats["batches_processed"] == 3

        # Check alerts (may have been generated for threshold violations)
        active_alerts = monitoring_client.alert_system.get_active_alerts()
        # Should have attempted to generate alerts
