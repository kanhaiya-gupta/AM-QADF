"""
Kafka Producer

Kafka producer implementation for testing and data publishing.
Supports both kafka-python and confluent-kafka backends.
"""

from typing import Dict, List, Optional, Any
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import Kafka libraries
try:
    from kafka import KafkaProducer as KafkaPythonProducer
    from kafka.errors import KafkaError

    KAFKA_PYTHON_AVAILABLE = True
except ImportError:
    KAFKA_PYTHON_AVAILABLE = False
    KafkaPythonProducer = None
    KafkaError = None

try:
    from confluent_kafka import Producer, KafkaError as ConfluentKafkaError, KafkaException

    CONFLUENT_KAFKA_AVAILABLE = True
except ImportError:
    CONFLUENT_KAFKA_AVAILABLE = False
    Producer = None
    ConfluentKafkaError = None
    KafkaException = None

from .streaming_client import StreamingConfig


class KafkaProducer:
    """
    Kafka producer for publishing messages.

    Supports both kafka-python and confluent-kafka backends.
    Uses confluent-kafka by default if available (better performance),
    falls back to kafka-python otherwise.
    """

    def __init__(self, config: StreamingConfig, use_confluent: Optional[bool] = None):
        """
        Initialize Kafka producer.

        Args:
            config: StreamingConfig with Kafka settings
            use_confluent: Force use of confluent-kafka (True) or kafka-python (False).
                          If None, auto-selects based on availability (prefers confluent-kafka).
        """
        self.config = config
        self.producer = None

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

        # Initialize producer
        if self.use_confluent:
            self._init_confluent()
        else:
            self._init_kafka_python()

        logger.info(f"Initialized Kafka producer with backend: {'confluent-kafka' if self.use_confluent else 'kafka-python'}")

    def _init_confluent(self) -> None:
        """Initialize confluent-kafka producer."""
        try:
            producer_config = {
                "bootstrap.servers": ",".join(self.config.kafka_bootstrap_servers),
            }
            self.producer = Producer(producer_config)
        except Exception as e:
            logger.error(f"Error initializing confluent-kafka producer: {e}")
            raise

    def _init_kafka_python(self) -> None:
        """Initialize kafka-python producer."""
        try:
            self.producer = KafkaPythonProducer(
                bootstrap_servers=self.config.kafka_bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
            )
        except Exception as e:
            logger.error(f"Error initializing kafka-python producer: {e}")
            raise

    def produce(self, topic: str, value: Any, key: Optional[str] = None, partition: Optional[int] = None) -> None:
        """
        Produce a message to a Kafka topic.

        Args:
            topic: Topic name
            value: Message value (will be JSON-serialized)
            key: Optional message key
            partition: Optional partition number
        """
        if not self.producer:
            raise RuntimeError("Producer not initialized")

        try:
            if self.use_confluent:
                self._produce_confluent(topic, value, key, partition)
            else:
                self._produce_kafka_python(topic, value, key, partition)
        except Exception as e:
            logger.error(f"Error producing message to topic {topic}: {e}")
            raise

    def _produce_confluent(self, topic: str, value: Any, key: Optional[str], partition: Optional[int]) -> None:
        """Produce using confluent-kafka backend."""
        try:
            # Prepare message
            if isinstance(value, (dict, list)):
                message_value = json.dumps(value).encode("utf-8")
            elif isinstance(value, str):
                message_value = value.encode("utf-8")
            else:
                message_value = str(value).encode("utf-8")

            message_key = key.encode("utf-8") if key else None

            # Produce message
            self.producer.produce(
                topic, value=message_value, key=message_key, partition=partition, callback=self._delivery_callback
            )

            # Poll to handle delivery callbacks
            self.producer.poll(0)

        except Exception as e:
            logger.error(f"Error in confluent-kafka produce: {e}")
            raise

    def _produce_kafka_python(self, topic: str, value: Any, key: Optional[str], partition: Optional[int]) -> None:
        """Produce using kafka-python backend."""
        try:
            future = self.producer.send(topic, value=value, key=key, partition=partition)

            # Wait for send to complete (with timeout)
            future.get(timeout=10)

        except KafkaError as e:
            logger.error(f"Kafka error in produce: {e}")
            raise
        except Exception as e:
            logger.error(f"Error in kafka-python produce: {e}")
            raise

    def _delivery_callback(self, err, msg) -> None:
        """Delivery callback for confluent-kafka."""
        if err is not None:
            logger.error(f"Message delivery failed: {err}")
        else:
            logger.debug(f"Message delivered to {msg.topic()}[{msg.partition()}]@offset {msg.offset()}")

    def flush(self, timeout: float = 10.0) -> None:
        """
        Flush pending messages.

        Args:
            timeout: Maximum time to wait for flush (seconds)
        """
        if not self.producer:
            return

        try:
            if self.use_confluent:
                self.producer.flush(timeout=int(timeout))
            else:
                self.producer.flush(timeout=timeout)
            logger.debug("Producer flushed")
        except Exception as e:
            logger.error(f"Error flushing producer: {e}")

    def close(self) -> None:
        """Close producer."""
        if self.producer:
            try:
                if self.use_confluent:
                    self.flush()
                    self.producer = None
                else:
                    self.producer.close()
                    self.producer = None
                logger.info("Producer closed")
            except Exception as e:
                logger.error(f"Error closing producer: {e}")
