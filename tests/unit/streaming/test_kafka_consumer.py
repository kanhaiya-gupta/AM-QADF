"""
Unit tests for KafkaConsumer.

Tests for Kafka consumer with mock backends.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from am_qadf.streaming.kafka_consumer import KafkaConsumer
from am_qadf.streaming.streaming_client import StreamingConfig


class TestKafkaConsumer:
    """Test suite for KafkaConsumer class."""

    @pytest.fixture
    def config(self):
        """Create a StreamingConfig instance."""
        return StreamingConfig(
            kafka_bootstrap_servers=["localhost:9092"],
            consumer_group_id="test_group",
        )

    @pytest.mark.unit
    @patch("am_qadf.streaming.kafka_consumer.CONFLUENT_KAFKA_AVAILABLE", True)
    @patch("am_qadf.streaming.kafka_consumer.Consumer")
    def test_consumer_creation_confluent(self, mock_consumer_class, config):
        """Test creating KafkaConsumer with confluent-kafka backend."""
        mock_consumer = MagicMock()
        mock_consumer_class.return_value = mock_consumer

        consumer = KafkaConsumer(config)

        assert consumer is not None
        assert consumer.use_confluent is True
        assert consumer.config == config

    @pytest.mark.unit
    @patch("am_qadf.streaming.kafka_consumer.CONFLUENT_KAFKA_AVAILABLE", False)
    @patch("am_qadf.streaming.kafka_consumer.KAFKA_PYTHON_AVAILABLE", True)
    @patch("am_qadf.streaming.kafka_consumer.KafkaPythonConsumer")
    def test_consumer_creation_kafka_python(self, mock_consumer_class, config):
        """Test creating KafkaConsumer with kafka-python backend."""
        mock_consumer = MagicMock()
        mock_consumer_class.return_value = mock_consumer

        consumer = KafkaConsumer(config)

        assert consumer is not None
        assert consumer.use_confluent is False
        assert consumer.config == config

    @pytest.mark.unit
    @patch("am_qadf.streaming.kafka_consumer.CONFLUENT_KAFKA_AVAILABLE", False)
    @patch("am_qadf.streaming.kafka_consumer.KAFKA_PYTHON_AVAILABLE", False)
    def test_consumer_creation_no_backend(self, config):
        """Test creating KafkaConsumer when no backend available."""
        with pytest.raises(ImportError, match="Neither kafka-python nor confluent-kafka is available"):
            KafkaConsumer(config)

    @pytest.mark.unit
    @patch("am_qadf.streaming.kafka_consumer.CONFLUENT_KAFKA_AVAILABLE", True)
    @patch("am_qadf.streaming.kafka_consumer.Consumer")
    def test_consume_confluent(self, mock_consumer_class, config):
        """Test consuming messages with confluent-kafka."""
        mock_consumer = MagicMock()
        mock_consumer_class.return_value = mock_consumer

        # Mock message polling - return None after first message to avoid infinite loop
        mock_message = MagicMock()
        mock_message.error.return_value = None
        mock_message.topic.return_value = "test_topic"
        mock_message.partition.return_value = 0
        mock_message.offset.return_value = 123
        mock_message.timestamp.return_value = (1, 1234567890000)
        mock_message.key.return_value = b"test_key"
        mock_message.value.return_value = b'{"test": "value"}'

        # Poll returns one message, then consistently returns None
        poll_count = [0]

        def poll_side_effect(timeout):
            poll_count[0] += 1
            if poll_count[0] == 1:
                return mock_message
            # After first message, return None to allow loop to continue but not hang
            return None

        mock_consumer.poll.side_effect = poll_side_effect

        consumer = KafkaConsumer(config)

        callback_called = []

        def callback(batch):
            callback_called.append(batch)

        # Run consume in a thread and stop it quickly
        import threading
        import time

        def run_consume():
            try:
                consumer._consume_confluent(["test_topic"], callback, timeout_ms=10)
            except Exception:
                pass  # Expected to stop

        # Start consumer in thread
        consume_thread = threading.Thread(target=run_consume, daemon=True)
        consume_thread.start()

        # Wait briefly for consumer to initialize and process
        time.sleep(0.05)

        # Stop the consumer loop
        consumer._is_running = False

        # Wait for thread to finish (with short timeout)
        consume_thread.join(timeout=0.5)

        # Consumer should have been initialized
        assert consumer.consumer is not None
        # Verify Consumer was called (created)
        mock_consumer_class.assert_called_once()
        # Verify subscribe was called
        mock_consumer.subscribe.assert_called_once_with(["test_topic"])

    @pytest.mark.unit
    @patch("am_qadf.streaming.kafka_consumer.CONFLUENT_KAFKA_AVAILABLE", True)
    @patch("am_qadf.streaming.kafka_consumer.Consumer")
    def test_pause_resume(self, mock_consumer_class, config):
        """Test pausing and resuming consumer."""
        mock_consumer = MagicMock()
        mock_consumer_class.return_value = mock_consumer

        consumer = KafkaConsumer(config, use_confluent=True)
        consumer.consumer = mock_consumer

        consumer.pause()
        assert consumer._paused is True

        consumer.resume()
        assert consumer._paused is False

    @pytest.mark.unit
    @patch("am_qadf.streaming.kafka_consumer.CONFLUENT_KAFKA_AVAILABLE", True)
    @patch("am_qadf.streaming.kafka_consumer.Consumer")
    def test_commit_offset(self, mock_consumer_class, config):
        """Test committing offsets."""
        mock_consumer = MagicMock()
        mock_consumer_class.return_value = mock_consumer

        consumer = KafkaConsumer(config, use_confluent=True)
        consumer.consumer = mock_consumer

        consumer.commit_offset()

        mock_consumer.commit.assert_called_once()

    @pytest.mark.unit
    @patch("am_qadf.streaming.kafka_consumer.CONFLUENT_KAFKA_AVAILABLE", True)
    @patch("am_qadf.streaming.kafka_consumer.Consumer")
    def test_seek_to_beginning(self, mock_consumer_class, config):
        """Test seeking to beginning of partitions."""
        mock_consumer = MagicMock()
        mock_consumer_class.return_value = mock_consumer

        # Mock assignment
        mock_partition = MagicMock()
        mock_consumer.assignment.return_value = [mock_partition]

        consumer = KafkaConsumer(config, use_confluent=True)
        consumer.consumer = mock_consumer

        consumer.seek_to_beginning()

        mock_consumer.seek.assert_called()

    @pytest.mark.unit
    @patch("am_qadf.streaming.kafka_consumer.CONFLUENT_KAFKA_AVAILABLE", True)
    @patch("am_qadf.streaming.kafka_consumer.Consumer")
    def test_close(self, mock_consumer_class, config):
        """Test closing consumer."""
        mock_consumer = MagicMock()
        mock_consumer_class.return_value = mock_consumer

        consumer = KafkaConsumer(config, use_confluent=True)
        consumer.consumer = mock_consumer
        consumer._is_running = True

        consumer.close()

        assert consumer._is_running is False
        mock_consumer.close.assert_called_once()
