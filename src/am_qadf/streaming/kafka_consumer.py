"""
Kafka Consumer

Kafka consumer implementation with error handling and rebalancing.
Supports both kafka-python and confluent-kafka backends.
"""

from typing import Dict, List, Optional, Callable, Any
import logging
import threading
import time
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import Kafka libraries
try:
    from kafka import KafkaConsumer as KafkaPythonConsumer
    from kafka.errors import KafkaError

    KAFKA_PYTHON_AVAILABLE = True
except ImportError:
    KAFKA_PYTHON_AVAILABLE = False
    KafkaPythonConsumer = None
    KafkaError = None

try:
    from confluent_kafka import Consumer, KafkaError as ConfluentKafkaError, KafkaException

    CONFLUENT_KAFKA_AVAILABLE = True
except ImportError:
    CONFLUENT_KAFKA_AVAILABLE = False
    Consumer = None
    ConfluentKafkaError = None
    KafkaException = None

from .streaming_client import StreamingConfig


class KafkaConsumer:
    """
    Kafka consumer with automatic error handling.

    Supports both kafka-python and confluent-kafka backends.
    Uses confluent-kafka by default if available (better performance),
    falls back to kafka-python otherwise.
    """

    def __init__(self, config: StreamingConfig, use_confluent: Optional[bool] = None):
        """
        Initialize Kafka consumer.

        Args:
            config: StreamingConfig with Kafka settings
            use_confluent: Force use of confluent-kafka (True) or kafka-python (False).
                          If None, auto-selects based on availability (prefers confluent-kafka).
        """
        self.config = config
        self.consumer = None
        self._is_running = False
        self._paused = False
        self._lock = threading.Lock()

        # Auto-select backend if not specified
        if use_confluent is None:
            use_confluent = CONFLUENT_KAFKA_AVAILABLE
        elif use_confluent and not CONFLUENT_KAFKA_AVAILABLE:
            logger.warning("confluent-kafka requested but not available, falling back to kafka-python")
            use_confluent = False
        elif not use_confluent and not KAFKA_PYTHON_AVAILABLE:
            logger.warning("kafka-python requested but not available, falling back to confluent-kafka")
            use_confluent = True

        self.use_confluent = use_confluent and CONFLUENT_KAFKA_AVAILABLE

        if not self.use_confluent and not KAFKA_PYTHON_AVAILABLE:
            raise ImportError(
                "Neither kafka-python nor confluent-kafka is available. "
                "Please install one: pip install kafka-python or pip install confluent-kafka"
            )

        logger.info(f"Initialized Kafka consumer with backend: {'confluent-kafka' if self.use_confluent else 'kafka-python'}")

    def consume(self, topics: List[str], callback: Callable, timeout_ms: int = 1000) -> None:
        """
        Consume messages from Kafka topics.

        Args:
            topics: List of topic names to consume from
            callback: Callback function called with message batches
            timeout_ms: Poll timeout in milliseconds
        """
        if self.use_confluent:
            self._consume_confluent(topics, callback, timeout_ms)
        else:
            self._consume_kafka_python(topics, callback, timeout_ms)

    def _consume_confluent(self, topics: List[str], callback: Callable, timeout_ms: int) -> None:
        """Consume using confluent-kafka backend."""
        try:
            # Build consumer config
            consumer_config = {
                "bootstrap.servers": ",".join(self.config.kafka_bootstrap_servers),
                "group.id": self.config.consumer_group_id,
                "auto.offset.reset": "earliest",
                "enable.auto.commit": self.config.enable_auto_commit,
                "auto.commit.interval.ms": self.config.auto_commit_interval_ms,
                "session.timeout.ms": self.config.session_timeout_ms,
                "max.poll.records": self.config.max_poll_records,
            }

            self.consumer = Consumer(consumer_config)
            self.consumer.subscribe(topics)

            self._is_running = True
            logger.info(f"Started confluent-kafka consumer for topics: {topics}")

            batch = []
            last_batch_time = time.time()
            batch_timeout = 0.1  # 100ms batch window

            while self._is_running:
                if self._paused:
                    time.sleep(0.1)
                    continue

                try:
                    # Poll for messages
                    msg = self.consumer.poll(timeout=timeout_ms / 1000.0)

                    if msg is None:
                        # Process accumulated batch if timeout
                        if batch and (time.time() - last_batch_time) > batch_timeout:
                            try:
                                callback(batch)
                            except Exception as e:
                                logger.error(f"Callback error: {e}")
                            batch = []
                        continue

                    if msg.error():
                        if msg.error().code() == ConfluentKafkaError._PARTITION_EOF:
                            logger.debug(f"Reached end of partition: {msg.topic()}[{msg.partition()}]")
                        else:
                            logger.error(f"Consumer error: {msg.error()}")
                        continue

                    # Parse message
                    try:
                        import json

                        value = json.loads(msg.value().decode("utf-8")) if msg.value() else {}
                    except Exception:
                        value = {"raw": msg.value().decode("utf-8") if msg.value() else ""}

                    message_data = {
                        "topic": msg.topic(),
                        "partition": msg.partition(),
                        "offset": msg.offset(),
                        "timestamp": (
                            datetime.fromtimestamp(msg.timestamp()[1] / 1000) if msg.timestamp()[1] else datetime.now()
                        ),
                        "key": msg.key().decode("utf-8") if msg.key() else None,
                        "value": value,
                    }

                    batch.append(message_data)

                    # Process batch if full or timeout
                    if len(batch) >= self.config.processing_batch_size:
                        try:
                            callback(batch)
                        except Exception as e:
                            logger.error(f"Callback error: {e}")
                        batch = []
                        last_batch_time = time.time()

                except KafkaException as e:
                    logger.error(f"Kafka exception: {e}")
                    if not self._is_running:
                        break
                    time.sleep(0.1)
                except Exception as e:
                    logger.error(f"Unexpected error in consumer loop: {e}")
                    if not self._is_running:
                        break
                    time.sleep(0.1)

            # Process remaining batch
            if batch:
                try:
                    callback(batch)
                except Exception as e:
                    logger.error(f"Callback error: {e}")

        except Exception as e:
            logger.error(f"Error starting confluent-kafka consumer: {e}")
            raise
        finally:
            if self.consumer:
                try:
                    self.consumer.close()
                except Exception as e:
                    logger.error(f"Error closing confluent-kafka consumer: {e}")

    def _consume_kafka_python(self, topics: List[str], callback: Callable, timeout_ms: int) -> None:
        """Consume using kafka-python backend."""
        try:
            self.consumer = KafkaPythonConsumer(
                *topics,
                bootstrap_servers=self.config.kafka_bootstrap_servers,
                group_id=self.config.consumer_group_id,
                enable_auto_commit=self.config.enable_auto_commit,
                auto_commit_interval_ms=self.config.auto_commit_interval_ms,
                max_poll_records=self.config.max_poll_records,
                session_timeout_ms=self.config.session_timeout_ms,
                value_deserializer=lambda m: m.decode("utf-8") if m else None,
                key_deserializer=lambda k: k.decode("utf-8") if k else None,
                consumer_timeout_ms=timeout_ms,
            )

            self._is_running = True
            logger.info(f"Started kafka-python consumer for topics: {topics}")

            batch = []
            last_batch_time = time.time()
            batch_timeout = 0.1  # 100ms batch window

            while self._is_running:
                if self._paused:
                    time.sleep(0.1)
                    continue

                try:
                    # Poll for messages
                    msg_pack = self.consumer.poll(timeout_ms=timeout_ms, max_records=self.config.max_poll_records)

                    if not msg_pack:
                        # Process accumulated batch if timeout
                        if batch and (time.time() - last_batch_time) > batch_timeout:
                            try:
                                callback(batch)
                            except Exception as e:
                                logger.error(f"Callback error: {e}")
                            batch = []
                        continue

                    # Process messages
                    for tp, messages in msg_pack.items():
                        for message in messages:
                            try:
                                import json

                                value = json.loads(message.value) if message.value else {}
                            except Exception:
                                value = {"raw": message.value} if message.value else {}

                            message_data = {
                                "topic": tp.topic,
                                "partition": tp.partition,
                                "offset": message.offset,
                                "timestamp": (
                                    datetime.fromtimestamp(message.timestamp / 1000)
                                    if hasattr(message, "timestamp") and message.timestamp
                                    else datetime.now()
                                ),
                                "key": message.key,
                                "value": value,
                            }

                            batch.append(message_data)

                    # Process batch if full or timeout
                    if len(batch) >= self.config.processing_batch_size:
                        try:
                            callback(batch)
                        except Exception as e:
                            logger.error(f"Callback error: {e}")
                        batch = []
                        last_batch_time = time.time()

                except KafkaError as e:
                    logger.error(f"Kafka error: {e}")
                    if not self._is_running:
                        break
                    time.sleep(0.1)
                except Exception as e:
                    logger.error(f"Unexpected error in consumer loop: {e}")
                    if not self._is_running:
                        break
                    time.sleep(0.1)

            # Process remaining batch
            if batch:
                try:
                    callback(batch)
                except Exception as e:
                    logger.error(f"Callback error: {e}")

        except Exception as e:
            logger.error(f"Error starting kafka-python consumer: {e}")
            raise
        finally:
            if self.consumer:
                try:
                    self.consumer.close()
                except Exception as e:
                    logger.error(f"Error closing kafka-python consumer: {e}")

    def pause(self) -> None:
        """Pause consumption."""
        with self._lock:
            self._paused = True
            if self.consumer:
                if self.use_confluent:
                    # confluent-kafka doesn't have pause, use flag
                    pass
                else:
                    try:
                        self.consumer.pause()
                    except Exception as e:
                        logger.error(f"Error pausing consumer: {e}")
            logger.info("Consumer paused")

    def resume(self) -> None:
        """Resume consumption."""
        with self._lock:
            self._paused = False
            if self.consumer:
                if self.use_confluent:
                    # confluent-kafka doesn't have pause, use flag
                    pass
                else:
                    try:
                        self.consumer.resume()
                    except Exception as e:
                        logger.error(f"Error resuming consumer: {e}")
            logger.info("Consumer resumed")

    def commit_offset(self) -> None:
        """Manually commit offsets."""
        if self.consumer:
            try:
                if self.use_confluent:
                    self.consumer.commit()
                else:
                    self.consumer.commit()
                logger.debug("Offsets committed")
            except Exception as e:
                logger.error(f"Error committing offsets: {e}")

    def seek_to_beginning(self) -> None:
        """Seek to beginning of partitions."""
        if self.consumer:
            try:
                if self.use_confluent:
                    # Get partitions and seek to beginning
                    for topic_partition in self.consumer.assignment():
                        self.consumer.seek(topic_partition, 0)
                else:
                    self.consumer.seek_to_beginning()
                logger.info("Seeked to beginning of partitions")
            except Exception as e:
                logger.error(f"Error seeking to beginning: {e}")

    def close(self) -> None:
        """Close consumer."""
        with self._lock:
            self._is_running = False
            if self.consumer:
                try:
                    if self.use_confluent:
                        self.consumer.close()
                    else:
                        self.consumer.close()
                    logger.info("Consumer closed")
                except Exception as e:
                    logger.error(f"Error closing consumer: {e}")
