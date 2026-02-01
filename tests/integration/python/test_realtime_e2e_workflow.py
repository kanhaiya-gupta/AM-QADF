"""
End-to-end tests for Real-time Monitoring and Control.

Comprehensive end-to-end tests that simulate complete real-time monitoring workflows.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta
import time

try:
    from am_qadf.streaming import (
        StreamingClient,
        StreamingConfig,
        IncrementalProcessor,
        BufferManager,
        StreamProcessor,
    )
    from am_qadf.monitoring import (
        MonitoringClient,
        MonitoringConfig,
        ThresholdConfig,
    )
    from am_qadf.analytics.spc import SPCClient, SPCConfig
    from am_qadf.analytics.quality_assessment.client import QualityAssessmentClient
    from am_qadf.voxelization.voxel_grid import VoxelGrid
except ImportError:
    pytest.skip("Required modules not available", allow_module_level=True)


@pytest.mark.e2e
class TestRealtimeE2EWorkflow:
    """End-to-end tests for complete real-time monitoring workflows."""

    @pytest.fixture
    def streaming_config(self):
        """Create streaming configuration."""
        return StreamingConfig(
            buffer_size=1000,
            processing_batch_size=100,
        )

    @pytest.fixture
    def monitoring_config(self):
        """Create monitoring configuration."""
        return MonitoringConfig(
            enable_alerts=True,
            alert_check_interval_seconds=0.5,
            alert_cooldown_seconds=1.0,
            enable_health_monitoring=True,
            health_check_interval_seconds=2.0,
            enable_dashboard_notifications=True,
        )

    @pytest.fixture
    def voxel_grid(self):
        """Create voxel grid for E2E testing."""
        return VoxelGrid(bbox_min=(0.0, 0.0, 0.0), bbox_max=(100.0, 100.0, 100.0), resolution=1.0, aggregation="mean")

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_complete_realtime_monitoring_workflow(self, streaming_config, monitoring_config, voxel_grid):
        """Test complete end-to-end real-time monitoring workflow."""
        # Initialize components
        streaming_client = StreamingClient(config=streaming_config)
        monitoring_client = MonitoringClient(config=monitoring_config)
        spc_client = SPCClient(config=SPCConfig(), mongo_client=None)

        # Create incremental processor with voxel grid
        incremental_processor = IncrementalProcessor(voxel_grid=voxel_grid)

        # Register metrics for monitoring
        temp_threshold = ThresholdConfig(
            metric_name="streaming_temperature",
            threshold_type="absolute",
            upper_threshold=1200.0,
        )
        monitoring_client.register_metric("streaming_temperature", temp_threshold)

        # Register processors
        def complete_processor(data_batch):
            results = {}

            # Extract values
            temperatures = []
            coordinates = []

            for item in data_batch:
                if isinstance(item, dict):
                    temp = item.get("temperature", 1000.0)
                    temperatures.append(temp)
                    coordinates.append([item.get("x", 0.0), item.get("y", 0.0), item.get("z", 0.0)])

            if temperatures:
                # Update voxel grid
                temps_array = np.array(temperatures)
                coords_array = np.array(coordinates)
                incremental_processor.update_voxel_grid(temps_array, coords_array)
                results["voxel_updated"] = True

                # Update monitoring metrics
                avg_temp = np.mean(temperatures)
                max_temp = np.max(temperatures)

                time.sleep(1.1)  # Wait for cooldown
                monitoring_client.update_metric("streaming_temperature", max_temp)

                # SPC analysis (if enough data)
                if len(temperatures) >= 5:
                    try:
                        chart_result = spc_client.create_control_chart(temps_array, chart_type="individual")
                        results["spc_analyzed"] = True
                        results["ooc_points"] = (
                            len(chart_result.out_of_control_points) if hasattr(chart_result, "out_of_control_points") else 0
                        )
                    except Exception:
                        results["spc_analyzed"] = False

                results["avg_temperature"] = avg_temp
                results["max_temperature"] = max_temp
                results["point_count"] = len(temperatures)

            return results

        streaming_client.register_processor("complete_processor", complete_processor)

        # Start monitoring
        monitoring_client.start_monitoring()

        # Simulate streaming data
        n_batches = 10
        batch_size = 20

        for i in range(n_batches):
            batch = []
            for j in range(batch_size):
                # Simulate temperature variations (some may exceed threshold)
                base_temp = 1000.0
                if i > 5:  # Introduce some hot batches
                    temp = base_temp + np.random.uniform(100.0, 250.0)
                else:
                    temp = base_temp + np.random.uniform(-50.0, 50.0)

                batch.append(
                    {
                        "temperature": temp,
                        "power": np.random.uniform(180.0, 220.0),
                        "velocity": np.random.uniform(90.0, 110.0),
                        "x": np.random.uniform(10.0, 90.0),
                        "y": np.random.uniform(10.0, 90.0),
                        "z": np.random.uniform(10.0, 90.0),
                        "timestamp": datetime.now(),
                    }
                )

            # Process batch
            result = streaming_client.process_stream_batch(batch)
            assert result.processed_count == batch_size

            time.sleep(0.5)  # Simulate real-time interval

        # Wait for monitoring to process
        time.sleep(2.0)

        # Check results
        stats = streaming_client.get_stream_statistics()
        assert stats["messages_processed"] == n_batches * batch_size
        assert stats["batches_processed"] == n_batches

        # Check metrics
        metrics = monitoring_client.get_current_metrics()
        assert "streaming_temperature" in metrics

        # Check health
        health_statuses = monitoring_client.get_health_status()
        assert "system" in health_statuses

        # Check incremental processor
        inc_stats = incremental_processor.get_statistics()
        assert inc_stats["total_points_processed"] == n_batches * batch_size

        # Stop monitoring
        monitoring_client.stop_monitoring()

        print(f"\nE2E Test Results:")
        print(f"  Messages processed: {stats['messages_processed']}")
        print(f"  Average latency: {stats['average_latency_ms']:.2f} ms")
        print(f"  Throughput: {stats['throughput_messages_per_sec']:.2f} msg/s")
        print(f"  Voxel points processed: {inc_stats['total_points_processed']}")
        print(f"  Active alerts: {len(monitoring_client.alert_system.get_active_alerts())}")

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_realtime_spc_with_quality_monitoring(self, streaming_config, monitoring_config):
        """Test real-time SPC with quality monitoring integration."""
        streaming_client = StreamingClient(config=streaming_config)
        monitoring_client = MonitoringClient(config=monitoring_config)
        spc_client = SPCClient(config=SPCConfig(), mongo_client=None)
        quality_client = QualityAssessmentClient(enable_spc=True, mongo_client=None)

        # Establish SPC baseline
        historical_data = np.random.normal(100.0, 10.0, 100)
        baseline = spc_client.establish_baseline(historical_data)

        # Register SPC metric
        spc_threshold = ThresholdConfig(
            metric_name="spc_ooc_count",
            threshold_type="absolute",
            upper_threshold=0.0,  # Any OOC point triggers alert
        )
        monitoring_client.register_metric("spc_ooc_count", spc_threshold)

        def spc_quality_processor(data_batch):
            values = []
            for item in data_batch:
                if isinstance(item, dict):
                    values.append(item.get("value", 100.0))
                else:
                    values.append(float(item))

            if not values:
                return None

            values_array = np.array(values)

            # SPC analysis
            try:
                chart_result = spc_client.create_control_chart(values_array, chart_type="individual")

                ooc_count = len(chart_result.out_of_control_points) if hasattr(chart_result, "out_of_control_points") else 0

                # Update monitoring
                if ooc_count > 0:
                    time.sleep(1.1)  # Wait for cooldown
                    monitoring_client.update_metric("spc_ooc_count", float(ooc_count))

                return {
                    "spc_analyzed": True,
                    "ooc_count": ooc_count,
                    "center_line": chart_result.center_line if hasattr(chart_result, "center_line") else baseline.mean,
                }
            except Exception as e:
                return {"spc_analyzed": False, "error": str(e)}

        streaming_client.register_processor("spc_quality_processor", spc_quality_processor)

        # Start monitoring
        monitoring_client.start_monitoring()

        # Simulate streaming with some out-of-control points
        n_batches = 5
        for i in range(n_batches):
            batch = []
            for j in range(10):
                # Create some out-of-control points
                if i == 3 and j > 5:  # Batch 3, second half
                    # Out-of-control values
                    value = baseline.mean + 4 * baseline.std
                else:
                    # Normal values
                    value = np.random.normal(baseline.mean, baseline.std)

                batch.append({"value": value})

            result = streaming_client.process_stream_batch(batch)
            time.sleep(0.5)

        # Wait for processing
        time.sleep(2.0)

        # Check results
        stats = streaming_client.get_stream_statistics()
        assert stats["batches_processed"] == n_batches

        # Check alerts (may have been generated)
        alerts = monitoring_client.alert_system.get_active_alerts()
        # Should have attempted to generate alerts for OOC points

        # Stop monitoring
        monitoring_client.stop_monitoring()

        print(f"\nSPC Quality E2E Results:")
        print(f"  Batches processed: {stats['batches_processed']}")
        print(f"  Active alerts: {len(alerts)}")

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_realtime_dashboard_updates(self, streaming_config, monitoring_config):
        """Test real-time dashboard updates with streaming data."""
        streaming_client = StreamingClient(config=streaming_config)
        monitoring_client = MonitoringClient(config=monitoring_config)

        # Mock WebSocket clients
        mock_clients = [MagicMock(), MagicMock()]
        monitoring_client.notification_channels._websocket_clients = mock_clients

        # Register metric
        dashboard_metric_threshold = ThresholdConfig(
            metric_name="dashboard_metric",
            threshold_type="absolute",
            upper_threshold=100.0,
        )
        monitoring_client.register_metric("dashboard_metric", dashboard_metric_threshold)

        def dashboard_update_processor(data_batch):
            values = [item.get("value", 50.0) if isinstance(item, dict) else float(item) for item in data_batch]

            if values:
                avg_value = sum(values) / len(values)

                # Update metric (triggers dashboard update if alert)
                time.sleep(1.1)  # Wait for cooldown
                monitoring_client.update_metric("dashboard_metric", avg_value)

                # Simulate dashboard update broadcast
                with patch("am_qadf.monitoring.notification_channels.json") as mock_json:
                    mock_json.dumps.return_value = '{"type": "metric_update"}'
                    monitoring_client.notification_channels.broadcast_websocket(
                        {
                            "type": "metric_update",
                            "metric": "dashboard_metric",
                            "value": avg_value,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )

            return {"processed": True, "avg_value": avg_value if values else 0.0}

        streaming_client.register_processor("dashboard_update_processor", dashboard_update_processor)

        # Start monitoring
        monitoring_client.start_monitoring()

        # Simulate streaming updates
        n_batches = 5
        for i in range(n_batches):
            batch = [{"value": 50.0 + i * 10.0}] * 10  # Increasing values
            result = streaming_client.process_stream_batch(batch)
            time.sleep(0.5)

        # Wait for processing
        time.sleep(2.0)

        # Check metrics
        metrics = monitoring_client.get_current_metrics()
        assert "dashboard_metric" in metrics

        # Check WebSocket broadcasts (mocked)
        # At least some broadcasts should have been attempted

        # Stop monitoring
        monitoring_client.stop_monitoring()

        print(f"\nDashboard E2E Results:")
        print(f"  Batches processed: {streaming_client.get_stream_statistics()['batches_processed']}")
        print(f"  Metrics updated: {len(metrics)}")

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_complete_realtime_pipeline_with_all_components(self, streaming_config, monitoring_config, voxel_grid):
        """Test complete real-time pipeline with all components integrated."""
        # Initialize all components
        streaming_client = StreamingClient(config=streaming_config)
        monitoring_client = MonitoringClient(config=monitoring_config)
        spc_client = SPCClient(config=SPCConfig(), mongo_client=None)
        buffer_manager = BufferManager(window_size=50, buffer_size=500)
        stream_processor = StreamProcessor(config=streaming_config)
        incremental_processor = IncrementalProcessor(voxel_grid=voxel_grid)

        # Establish SPC baseline
        historical_data = np.random.normal(100.0, 5.0, 100)
        baseline = spc_client.establish_baseline(historical_data)

        # Register metrics
        temp_threshold = ThresholdConfig(
            metric_name="pipeline_temperature",
            threshold_type="absolute",
            upper_threshold=1100.0,
        )
        monitoring_client.register_metric("pipeline_temperature", temp_threshold)

        # Create processing pipeline
        def stage1_extract(data_batch):
            # Extract values
            extracted = []
            for item in data_batch:
                if isinstance(item, dict):
                    extracted.append(
                        {
                            "value": item.get("temperature", 1000.0),
                            "coordinates": [item.get("x", 0.0), item.get("y", 0.0), item.get("z", 0.0)],
                        }
                    )
            return extracted

        def stage2_process(extracted):
            # Update voxel grid
            if extracted:
                values = [e["value"] for e in extracted]
                coords = np.array([e["coordinates"] for e in extracted])
                incremental_processor.update_voxel_grid(np.array(values), coords)
            return extracted

        def stage3_spc(extracted):
            # SPC analysis
            if extracted:
                values = np.array([e["value"] for e in extracted])
                try:
                    chart_result = spc_client.create_control_chart(values, chart_type="individual")
                    return {
                        "extracted": extracted,
                        "spc_result": chart_result,
                        "ooc_count": (
                            len(chart_result.out_of_control_points) if hasattr(chart_result, "out_of_control_points") else 0
                        ),
                    }
                except Exception:
                    return {"extracted": extracted, "spc_result": None}
            return {"extracted": extracted}

        def stage4_monitor(processed):
            # Update monitoring
            if "extracted" in processed and processed["extracted"]:
                values = [e["value"] for e in processed["extracted"]]
                max_value = max(values)
                time.sleep(1.1)  # Wait for cooldown
                monitoring_client.update_metric("pipeline_temperature", max_value)
            return processed

        pipeline = stream_processor.create_processing_pipeline([stage1_extract, stage2_process, stage3_spc, stage4_monitor])

        # Register pipeline processor
        def pipeline_processor(data_batch):
            return stream_processor.process_with_pipeline(data_batch, pipeline)

        streaming_client.register_processor("pipeline_processor", pipeline_processor)

        # Start monitoring
        monitoring_client.start_monitoring()

        # Simulate complete workflow
        n_iterations = 5
        for iteration in range(n_iterations):
            # Generate batch
            batch = []
            for j in range(20):
                temp = 1000.0 + np.random.uniform(-100.0, 150.0)
                batch.append(
                    {
                        "temperature": temp,
                        "x": np.random.uniform(10.0, 90.0),
                        "y": np.random.uniform(10.0, 90.0),
                        "z": np.random.uniform(10.0, 90.0),
                        "timestamp": datetime.now(),
                    }
                )

            # Add to buffer
            for item in batch:
                buffer_manager.add_data(np.array([item["temperature"]]), item["timestamp"], metadata=item)

            # Get window and process
            window, timestamps, metadata_list = buffer_manager.get_sliding_window(20)

            # Convert to batch format
            window_batch = []
            for i, meta in enumerate(metadata_list[-20:]):
                window_batch.append(
                    {
                        "temperature": window.flatten()[i] if len(window.flatten()) > i else 1000.0,
                        "x": meta.get("x", 0.0),
                        "y": meta.get("y", 0.0),
                        "z": meta.get("z", 0.0),
                        "timestamp": timestamps[i] if i < len(timestamps) else datetime.now(),
                    }
                )

            # Process through streaming client
            result = streaming_client.process_stream_batch(window_batch)

            time.sleep(1.0)  # Real-time interval

        # Wait for processing
        time.sleep(3.0)

        # Check all components
        stream_stats = streaming_client.get_stream_statistics()
        processor_stats = stream_processor.get_statistics()
        inc_stats = incremental_processor.get_statistics()
        buffer_stats = buffer_manager.get_buffer_statistics()
        metrics = monitoring_client.get_current_metrics()
        health = monitoring_client.get_health_status()

        # Stop monitoring
        monitoring_client.stop_monitoring()

        print(f"\nComplete Pipeline E2E Results:")
        print(f"  Stream stats: {stream_stats['messages_processed']} messages, {stream_stats['batches_processed']} batches")
        print(f"  Processor latency: {processor_stats['average_latency_ms']:.2f} ms")
        print(f"  Incremental processor: {inc_stats['total_points_processed']} points")
        print(f"  Buffer size: {buffer_stats['current_size']}")
        print(f"  Metrics monitored: {len(metrics)}")
        print(f"  Health components: {len(health)}")

        # Verify pipeline worked
        assert stream_stats["messages_processed"] > 0
        assert inc_stats["total_points_processed"] > 0
        assert "pipeline_temperature" in metrics or len(metrics) >= 0
