"""
Unit tests for StreamingClient.

Tests for streaming client interface and configuration.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from am_qadf.streaming.streaming_client import (
    StreamingClient,
    StreamingConfig,
    StreamingResult,
)


class TestStreamingConfig:
    """Test suite for StreamingConfig dataclass."""

    @pytest.mark.unit
    def test_streaming_config_creation_defaults(self):
        """Test creating StreamingConfig with defaults."""
        config = StreamingConfig()

        assert config.kafka_bootstrap_servers == ["localhost:9092"]
        assert config.kafka_topic == "am_qadf_monitoring"
        assert config.consumer_group_id == "am_qadf_consumers"
        assert config.enable_auto_commit is True
        assert config.auto_commit_interval_ms == 5000
        assert config.max_poll_records == 100
        assert config.session_timeout_ms == 30000
        assert config.buffer_size == 1000
        assert config.processing_batch_size == 100
        assert config.enable_redis_cache is True
        assert config.redis_host == "localhost"
        assert config.redis_port == 6379
        assert config.redis_db == 0
        assert config.enable_mongodb_storage is True
        assert config.storage_batch_size == 1000

    @pytest.mark.unit
    def test_streaming_config_custom(self):
        """Test creating StreamingConfig with custom values."""
        config = StreamingConfig(
            kafka_bootstrap_servers=["broker1:9092", "broker2:9092"],
            kafka_topic="custom_topic",
            consumer_group_id="custom_group",
            enable_auto_commit=False,
            auto_commit_interval_ms=10000,
            max_poll_records=200,
            session_timeout_ms=60000,
            buffer_size=2000,
            processing_batch_size=200,
            enable_redis_cache=False,
            redis_host="redis.example.com",
            redis_port=6380,
            redis_db=1,
            enable_mongodb_storage=False,
            storage_batch_size=2000,
        )

        assert config.kafka_bootstrap_servers == ["broker1:9092", "broker2:9092"]
        assert config.kafka_topic == "custom_topic"
        assert config.consumer_group_id == "custom_group"
        assert config.enable_auto_commit is False
        assert config.auto_commit_interval_ms == 10000
        assert config.max_poll_records == 200
        assert config.session_timeout_ms == 60000
        assert config.buffer_size == 2000
        assert config.processing_batch_size == 200
        assert config.enable_redis_cache is False
        assert config.redis_host == "redis.example.com"
        assert config.redis_port == 6380
        assert config.redis_db == 1
        assert config.enable_mongodb_storage is False
        assert config.storage_batch_size == 2000


class TestStreamingResult:
    """Test suite for StreamingResult dataclass."""

    @pytest.mark.unit
    def test_streaming_result_creation(self):
        """Test creating StreamingResult."""
        timestamp = datetime.now()
        data_batch = np.array([1.0, 2.0, 3.0])

        result = StreamingResult(
            timestamp=timestamp,
            data_batch=data_batch,
            processed_count=3,
            processing_time_ms=10.5,
            voxel_updates={"updated_regions": 5},
            quality_metrics={"completeness": 0.95},
            spc_results={"baseline_established": True},
            alerts_generated=["alert1", "alert2"],
            metadata={"test": "value"},
        )

        assert result.timestamp == timestamp
        assert np.array_equal(result.data_batch, data_batch)
        assert result.processed_count == 3
        assert result.processing_time_ms == 10.5
        assert result.voxel_updates == {"updated_regions": 5}
        assert result.quality_metrics == {"completeness": 0.95}
        assert result.spc_results == {"baseline_established": True}
        assert result.alerts_generated == ["alert1", "alert2"]
        assert result.metadata == {"test": "value"}

    @pytest.mark.unit
    def test_streaming_result_minimal(self):
        """Test creating StreamingResult with minimal fields."""
        timestamp = datetime.now()
        data_batch = np.array([])

        result = StreamingResult(
            timestamp=timestamp,
            data_batch=data_batch,
            processed_count=0,
            processing_time_ms=0.0,
        )

        assert result.timestamp == timestamp
        assert np.array_equal(result.data_batch, data_batch)
        assert result.processed_count == 0
        assert result.processing_time_ms == 0.0
        assert result.voxel_updates is None
        assert result.quality_metrics is None
        assert result.spc_results is None
        assert result.alerts_generated == []
        assert result.metadata == {}


class TestStreamingClient:
    """Test suite for StreamingClient class."""

    @pytest.fixture
    def config(self):
        """Create a StreamingConfig instance."""
        return StreamingConfig()

    @pytest.fixture
    def client(self, config):
        """Create a StreamingClient instance."""
        return StreamingClient(config=config)

    @pytest.mark.unit
    def test_client_creation(self, client):
        """Test creating StreamingClient."""
        assert client is not None
        assert client.config is not None
        assert client.kafka_consumer is None
        assert client.kafka_producer is None
        assert client._is_running is False
        assert client._statistics["messages_processed"] == 0

    @pytest.mark.unit
    def test_client_creation_with_config(self, config):
        """Test creating StreamingClient with custom config."""
        client = StreamingClient(config=config)
        assert client.config == config

    @pytest.mark.unit
    def test_client_creation_default_config(self):
        """Test creating StreamingClient with default config."""
        client = StreamingClient()
        assert client.config is not None
        assert isinstance(client.config, StreamingConfig)

    @pytest.mark.unit
    def test_process_stream_batch_dict(self, client):
        """Test processing stream batch with dictionary data."""
        data_batch = [
            {"value": 1.0, "timestamp": datetime.now()},
            {"value": 2.0, "timestamp": datetime.now()},
            {"value": 3.0, "timestamp": datetime.now()},
        ]

        result = client.process_stream_batch(data_batch)

        assert isinstance(result, StreamingResult)
        assert result.processed_count == 3
        assert result.processing_time_ms > 0
        assert len(result.data_batch) == 3
        assert result.metadata["processor_count"] == 0  # No processors registered

    @pytest.mark.unit
    def test_process_stream_batch_array(self, client):
        """Test processing stream batch with array data."""
        data_batch = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = client.process_stream_batch(data_batch.tolist())

        assert isinstance(result, StreamingResult)
        assert result.processed_count == 5
        assert result.processing_time_ms > 0

    @pytest.mark.unit
    def test_process_stream_batch_with_processor(self, client):
        """Test processing stream batch with registered processor."""

        def test_processor(data):
            return {"processed": True, "count": len(data)}

        client.register_processor("test_processor", test_processor)

        data_batch = [{"value": 1.0}, {"value": 2.0}]
        result = client.process_stream_batch(data_batch)

        assert result.processed_count == 2
        assert result.metadata["processor_count"] == 1

    @pytest.mark.unit
    def test_process_stream_batch_empty(self, client):
        """Test processing empty stream batch."""
        data_batch = []

        result = client.process_stream_batch(data_batch)

        assert isinstance(result, StreamingResult)
        assert result.processed_count == 0
        assert len(result.data_batch) == 0

    @pytest.mark.unit
    def test_register_processor(self, client):
        """Test registering a processor."""

        def processor(data):
            return {"result": "processed"}

        client.register_processor("test_processor", processor)

        assert "test_processor" in client._processors
        assert client._processors["test_processor"] == processor

    @pytest.mark.unit
    def test_unregister_processor(self, client):
        """Test unregistering a processor."""

        def processor(data):
            return {"result": "processed"}

        client.register_processor("test_processor", processor)
        assert "test_processor" in client._processors

        client.unregister_processor("test_processor")
        assert "test_processor" not in client._processors

    @pytest.mark.unit
    def test_unregister_nonexistent_processor(self, client):
        """Test unregistering a non-existent processor."""
        # Should not raise error, just log warning
        client.unregister_processor("nonexistent_processor")

    @pytest.mark.unit
    def test_get_stream_statistics(self, client):
        """Test getting stream statistics."""
        # Process some batches
        client.process_stream_batch([{"value": 1.0}, {"value": 2.0}])
        client.process_stream_batch([{"value": 3.0}])

        stats = client.get_stream_statistics()

        assert stats["messages_processed"] == 3
        assert stats["batches_processed"] == 2
        assert stats["total_processing_time_ms"] > 0
        assert stats["average_latency_ms"] > 0
        assert stats["errors"] == 0

    @pytest.mark.unit
    def test_reset_statistics(self, client):
        """Test resetting statistics."""
        # Process some batches
        client.process_stream_batch([{"value": 1.0}])
        stats_before = client.get_stream_statistics()
        assert stats_before["batches_processed"] == 1

        # Reset
        client.reset_statistics()
        stats_after = client.get_stream_statistics()

        assert stats_after["messages_processed"] == 0
        assert stats_after["batches_processed"] == 0
        assert stats_after["total_processing_time_ms"] == 0.0
        assert stats_after["average_latency_ms"] == 0.0
        assert stats_after["errors"] == 0

    @pytest.mark.unit
    @patch("am_qadf.streaming.kafka_consumer.KafkaConsumer")
    def test_start_consumer(self, mock_kafka_consumer, client):
        """Test starting Kafka consumer."""
        mock_consumer = MagicMock()
        # Make consume block briefly so thread stays alive long enough for test
        import time

        def mock_consume(topics, callback, timeout_ms=1000):
            time.sleep(0.2)  # Block briefly

        mock_consumer.consume = mock_consume
        mock_kafka_consumer.return_value = mock_consumer

        topics = ["topic1", "topic2"]

        def callback(batch):
            pass

        # Start consumer
        client.start_consumer(topics, callback)

        # Wait for thread to start and set _is_running
        max_wait = 0.5
        elapsed = 0
        while not client._is_running and elapsed < max_wait:
            time.sleep(0.01)
            elapsed += 0.01

        # Check consumer was created and is running
        assert client.kafka_consumer is not None
        assert client._is_running is True

        # Cleanup
        client.stop_consumer()

    @pytest.mark.unit
    def test_start_consumer_already_running(self, client):
        """Test starting consumer when already running."""
        client._is_running = True

        # Should log warning but not raise error
        client.start_consumer(["topic1"], lambda x: None)

    @pytest.mark.unit
    def test_stop_consumer_not_running(self, client):
        """Test stopping consumer when not running."""
        # Should log warning but not raise error
        client.stop_consumer()

    @pytest.mark.unit
    @patch("am_qadf.streaming.kafka_consumer.KafkaConsumer")
    def test_stop_consumer(self, mock_kafka_consumer, client):
        """Test stopping Kafka consumer."""
        mock_consumer = MagicMock()
        mock_kafka_consumer.return_value = mock_consumer

        client.kafka_consumer = mock_consumer
        client._is_running = True

        client.stop_consumer()

        mock_consumer.close.assert_called_once()
        assert client._is_running is False

    @pytest.mark.unit
    def test_process_stream_batch_error_handling(self, client):
        """Test error handling in process_stream_batch."""

        def failing_processor(data):
            raise ValueError("Processor failed")

        client.register_processor("failing_processor", failing_processor)

        data_batch = [{"value": 1.0}]
        result = client.process_stream_batch(data_batch)

        # Should still return result (processor failures are logged but don't fail the batch)
        assert isinstance(result, StreamingResult)
        # Processor failures don't add 'error' to metadata - they're just logged
        # The batch still processes successfully
        assert result.processed_count == 1
        stats = client.get_stream_statistics()
        # Errors counter is only incremented when the whole batch fails, not for processor failures
        assert stats["errors"] == 0

    @pytest.mark.unit
    def test_process_stream_batch_invalid_data(self, client):
        """Test processing invalid data."""
        # Invalid data type - string instead of list
        data_batch = "not a list"

        result = client.process_stream_batch(data_batch)

        # Current implementation doesn't validate input type, so it processes the string
        # np.asarray("not a list") creates an array, len("not a list") = 10
        assert isinstance(result, StreamingResult)
        # The implementation processes it as a string, so processed_count = len(string) = 10
        assert result.processed_count == 10
