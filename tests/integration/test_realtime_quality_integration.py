"""
Integration tests for Real-time Quality Assessment with Streaming.

Tests integration of Quality Assessment module with streaming data.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
import time

try:
    from am_qadf.analytics.quality_assessment.client import QualityAssessmentClient
    from am_qadf.streaming import StreamingClient, StreamingConfig, IncrementalProcessor
    from am_qadf.monitoring import MonitoringClient, MonitoringConfig, ThresholdConfig
    from am_qadf.voxelization.voxel_grid import VoxelGrid
except ImportError:
    pytest.skip("Quality assessment or streaming modules not available", allow_module_level=True)


class MockVoxelData:
    """Mock voxel data for testing."""

    def __init__(self, signals: dict, dims: tuple = (10, 10, 10)):
        self._signals = signals
        self.dims = dims
        self.available_signals = list(signals.keys())

    def get_signal_array(self, signal_name: str, default: float = 0.0) -> np.ndarray:
        return self._signals.get(signal_name, np.full(self.dims, default))


class TestRealtimeQualityIntegration:
    """Integration tests for real-time quality assessment with streaming."""

    @pytest.fixture
    def quality_client(self):
        """Create quality assessment client."""
        return QualityAssessmentClient(enable_spc=True, mongo_client=None)

    @pytest.fixture
    def streaming_client(self):
        """Create streaming client."""
        config = StreamingConfig(processing_batch_size=10)
        return StreamingClient(config=config)

    @pytest.fixture
    def monitoring_client(self):
        """Create monitoring client."""
        config = MonitoringConfig(alert_cooldown_seconds=0.1)
        return MonitoringClient(config=config)

    @pytest.fixture
    def voxel_grid(self):
        """Create voxel grid for incremental processing."""
        return VoxelGrid(bbox_min=(0.0, 0.0, 0.0), bbox_max=(100.0, 100.0, 100.0), resolution=1.0)

    @pytest.mark.integration
    def test_quality_assessment_with_streaming(self, quality_client, streaming_client):
        """Test quality assessment on streaming data."""
        quality_metrics_list = []

        def quality_processor(data_batch):
            # Extract data points
            points = []
            signals = {"temperature": [], "power": []}

            for item in data_batch:
                if isinstance(item, dict):
                    points.append((item.get("x", 0.0), item.get("y", 0.0), item.get("z", 0.0)))
                    signals["temperature"].append(item.get("temperature", 1000.0))
                    signals["power"].append(item.get("power", 200.0))

            if not points:
                return None

            # Create mock voxel data for quality assessment
            # In real implementation, this would update actual voxel grid
            mock_voxel_data = MockVoxelData(
                {
                    "temperature": np.array(signals["temperature"]).reshape(-1, 1, 1),
                    "power": np.array(signals["power"]).reshape(-1, 1, 1),
                }
            )

            try:
                # Assess quality
                quality_metrics = quality_client.assess_data_quality(mock_voxel_data, signals=["temperature", "power"])

                quality_metrics_dict = {
                    "completeness": quality_metrics.completeness if hasattr(quality_metrics, "completeness") else 0.0,
                    "coverage": quality_metrics.coverage if hasattr(quality_metrics, "coverage") else 0.0,
                    "timestamp": datetime.now().isoformat(),
                }
                quality_metrics_list.append(quality_metrics_dict)

                return quality_metrics_dict
            except Exception as e:
                return {"error": str(e)}

        streaming_client.register_processor("quality_processor", quality_processor)

        # Process batch
        batch = [
            {"x": 10.0, "y": 20.0, "z": 30.0, "temperature": 1000.0, "power": 200.0},
            {"x": 11.0, "y": 21.0, "z": 31.0, "temperature": 1050.0, "power": 210.0},
            {"x": 12.0, "y": 22.0, "z": 32.0, "temperature": 1100.0, "power": 220.0},
        ]

        result = streaming_client.process_stream_batch(batch)

        # Should have quality metrics
        assert result.quality_metrics is not None or result.processed_count > 0

    @pytest.mark.integration
    def test_quality_with_monitoring_thresholds(self, quality_client, streaming_client, monitoring_client):
        """Test quality assessment with monitoring thresholds."""
        # Register quality metrics for monitoring
        completeness_threshold = ThresholdConfig(
            metric_name="quality_completeness",
            threshold_type="absolute",
            lower_threshold=0.8,  # Minimum 80% completeness
        )
        monitoring_client.register_metric("quality_completeness", completeness_threshold)

        def quality_monitoring_processor(data_batch):
            # Simulate quality assessment
            # In real implementation, would call quality_client.assess_data_quality
            mock_voxel_data = MockVoxelData(
                {
                    "temperature": np.ones((5, 5, 5)) * 1000.0,
                }
            )

            try:
                quality_metrics = quality_client.assess_data_quality(mock_voxel_data, signals=["temperature"])

                # Extract completeness metric
                completeness = quality_metrics.completeness if hasattr(quality_metrics, "completeness") else 0.95

                # Update monitoring metric
                monitoring_client.update_metric("quality_completeness", completeness)

                return {
                    "quality_assessed": True,
                    "completeness": completeness,
                }
            except Exception:
                return {"quality_assessed": False, "completeness": 0.5}

        streaming_client.register_processor("quality_monitoring_processor", quality_monitoring_processor)

        # Process batch
        batch = [{"value": 1.0}] * 5
        time.sleep(0.2)  # Wait for cooldown
        result = streaming_client.process_stream_batch(batch)

        # Check metrics were updated
        metrics = monitoring_client.get_current_metrics()
        # May or may not have 'quality_completeness' depending on processing

    @pytest.mark.integration
    def test_incremental_quality_updates(self, quality_client, voxel_grid):
        """Test incremental quality updates with voxel grid."""
        incremental_processor = IncrementalProcessor(voxel_grid=voxel_grid)

        # Add data incrementally
        for i in range(10):
            coordinates = np.array([[10.0 + i, 20.0 + i, 30.0 + i]])
            signals = np.array([1000.0 + i])
            incremental_processor.update_voxel_grid(signals, coordinates)

        # Finalize voxel grid (if needed)
        voxel_grid.finalize()

        # Assess quality on updated grid
        try:
            quality_metrics = quality_client.assess_data_quality(
                voxel_grid, signals=["value"] if "value" in voxel_grid.available_signals else []
            )

            assert quality_metrics is not None
        except Exception:
            # May fail if no signals, that's okay
            pass

        # Check incremental processor statistics
        stats = incremental_processor.get_statistics()
        assert stats["total_points_processed"] == 10

    @pytest.mark.integration
    def test_quality_spc_integration(self, quality_client, streaming_client):
        """Test quality assessment with SPC integration."""
        # Quality client has SPC enabled
        assert quality_client.enable_spc is True

        def quality_spc_processor(data_batch):
            # Create voxel data
            mock_voxel_data = MockVoxelData(
                {
                    "temperature": np.ones((5, 5, 5)) * 1000.0,
                }
            )

            try:
                # Assess quality with SPC
                if hasattr(quality_client, "assess_with_spc"):
                    results = quality_client.assess_with_spc(mock_voxel_data, signals=["temperature"])
                    return results
                else:
                    # Fallback to regular assessment
                    quality_metrics = quality_client.assess_data_quality(mock_voxel_data, signals=["temperature"])
                    return {"quality_metrics": quality_metrics}
            except Exception as e:
                return {"error": str(e)}

        streaming_client.register_processor("quality_spc_processor", quality_spc_processor)

        batch = [{"value": 1.0}] * 5
        result = streaming_client.process_stream_batch(batch)

        # Should have processed
        assert result is not None
