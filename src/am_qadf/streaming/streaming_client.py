"""
Streaming Client

Main streaming interface for real-time data processing.
Provides Kafka integration, incremental processing, and stream statistics.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
import logging
from datetime import datetime
import threading
import time

logger = logging.getLogger(__name__)


@dataclass
class StreamingConfig:
    """Configuration for streaming operations."""

    kafka_bootstrap_servers: List[str] = field(default_factory=lambda: ["localhost:9092"])
    kafka_topic: str = "am_qadf_monitoring"
    consumer_group_id: str = "am_qadf_consumers"
    enable_auto_commit: bool = True
    auto_commit_interval_ms: int = 5000
    max_poll_records: int = 100
    session_timeout_ms: int = 30000
    buffer_size: int = 1000  # Buffer size for temporal windows
    processing_batch_size: int = 100  # Batch size for incremental processing
    enable_redis_cache: bool = True
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    enable_mongodb_storage: bool = True
    storage_batch_size: int = 1000  # Batch size for MongoDB writes


@dataclass
class StreamingResult:
    """Result of streaming data processing."""

    timestamp: datetime
    data_batch: np.ndarray
    processed_count: int
    processing_time_ms: float
    voxel_updates: Optional[Dict[str, Any]] = None
    quality_metrics: Optional[Dict[str, float]] = None
    spc_results: Optional[Dict[str, Any]] = None
    alerts_generated: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class StreamingClient:
    """
    Main streaming client interface.

    Provides:
    - Kafka consumer/producer integration
    - Stream processing with callbacks
    - Stream statistics and monitoring
    - Custom processor registration
    """

    def __init__(self, config: Optional[StreamingConfig] = None):
        """
        Initialize streaming client.

        Args:
            config: Optional StreamingConfig for default settings
        """
        self.config = config if config is not None else StreamingConfig()
        self.kafka_consumer = None
        self.kafka_producer = None
        self._processors: Dict[str, Callable] = {}
        self._is_running = False
        self._consumer_thread: Optional[threading.Thread] = None
        self._statistics = {
            "messages_processed": 0,
            "batches_processed": 0,
            "total_processing_time_ms": 0.0,
            "average_latency_ms": 0.0,
            "throughput_messages_per_sec": 0.0,
            "last_update_time": datetime.now(),
            "errors": 0,
        }
        self._lock = threading.Lock()

        logger.info("StreamingClient initialized")

    def start_consumer(self, topics: List[str], callback: Callable) -> None:
        """
        Start Kafka consumer and process messages.

        Args:
            topics: List of Kafka topics to consume from
            callback: Callback function for each message batch
        """
        if self._is_running:
            logger.warning("Consumer is already running")
            return

        try:
            from .kafka_consumer import KafkaConsumer

            self.kafka_consumer = KafkaConsumer(self.config)

            def consumer_loop():
                """Consumer loop running in separate thread."""
                self._is_running = True
                try:
                    self.kafka_consumer.consume(topics, callback)
                except Exception as e:
                    logger.error(f"Error in consumer loop: {e}")
                    self._statistics["errors"] += 1
                finally:
                    self._is_running = False

            self._consumer_thread = threading.Thread(target=consumer_loop, daemon=True)
            self._consumer_thread.start()
            logger.info(f"Started Kafka consumer for topics: {topics}")
        except ImportError as e:
            logger.error(f"Kafka consumer not available: {e}")
            raise RuntimeError("Kafka consumer not available. Install kafka-python or confluent-kafka.")
        except Exception as e:
            logger.error(f"Failed to start consumer: {e}")
            raise

    def stop_consumer(self) -> None:
        """Stop Kafka consumer."""
        if not self._is_running:
            logger.warning("Consumer is not running")
            return

        self._is_running = False
        if self.kafka_consumer:
            try:
                self.kafka_consumer.close()
            except Exception as e:
                logger.error(f"Error closing consumer: {e}")

        if self._consumer_thread and self._consumer_thread.is_alive():
            self._consumer_thread.join(timeout=5.0)

        logger.info("Kafka consumer stopped")

    def process_stream_batch(self, data_batch: List[Dict]) -> StreamingResult:
        """
        Process a batch of streaming data.

        Args:
            data_batch: List of data dictionaries from stream

        Returns:
            StreamingResult with processing results
        """
        start_time = time.time()
        timestamp = datetime.now()

        try:
            # Convert to numpy array if needed
            if data_batch and isinstance(data_batch[0], dict):
                # Extract numeric data from dictionaries
                data_array = np.array([item.get("value", item.get("data", 0.0)) for item in data_batch])
            else:
                data_array = np.asarray(data_batch)

            # Apply registered processors
            voxel_updates = None
            quality_metrics = None
            spc_results = None
            alerts = []

            for processor_name, processor in self._processors.items():
                try:
                    result = processor(data_batch)
                    if processor_name == "voxel_processor":
                        voxel_updates = result
                    elif processor_name == "quality_processor":
                        quality_metrics = result
                    elif processor_name == "spc_processor":
                        spc_results = result
                    elif processor_name == "alert_processor":
                        alerts.extend(result if isinstance(result, list) else [result])
                except Exception as e:
                    logger.warning(f"Processor {processor_name} failed: {e}")

            processing_time_ms = (time.time() - start_time) * 1000.0

            # Update statistics
            with self._lock:
                self._statistics["messages_processed"] += len(data_batch)
                self._statistics["batches_processed"] += 1
                self._statistics["total_processing_time_ms"] += processing_time_ms

                # Calculate average latency
                total_time = self._statistics["total_processing_time_ms"]
                batches = self._statistics["batches_processed"]
                self._statistics["average_latency_ms"] = total_time / batches if batches > 0 else 0.0

                # Calculate throughput
                elapsed_time = (datetime.now() - self._statistics["last_update_time"]).total_seconds()
                if elapsed_time > 0:
                    messages = self._statistics["messages_processed"]
                    self._statistics["throughput_messages_per_sec"] = messages / elapsed_time

                self._statistics["last_update_time"] = datetime.now()

            result = StreamingResult(
                timestamp=timestamp,
                data_batch=data_array,
                processed_count=len(data_batch),
                processing_time_ms=processing_time_ms,
                voxel_updates=voxel_updates,
                quality_metrics=quality_metrics,
                spc_results=spc_results,
                alerts_generated=alerts,
                metadata={"processor_count": len(self._processors)},
            )

            return result

        except Exception as e:
            logger.error(f"Error processing stream batch: {e}")
            with self._lock:
                self._statistics["errors"] += 1

            # Return error result
            return StreamingResult(
                timestamp=timestamp,
                data_batch=np.array([]),
                processed_count=0,
                processing_time_ms=(time.time() - start_time) * 1000.0,
                metadata={"error": str(e)},
            )

    def get_stream_statistics(self) -> Dict[str, Any]:
        """
        Get streaming statistics (throughput, latency, etc.).

        Returns:
            Dictionary with streaming statistics
        """
        with self._lock:
            return self._statistics.copy()

    def register_processor(self, processor_type: str, processor: Callable) -> None:
        """
        Register custom stream processor.

        Args:
            processor_type: Type/name of processor ('voxel_processor', 'quality_processor', etc.)
            processor: Callable that processes data batches
        """
        self._processors[processor_type] = processor
        logger.info(f"Registered processor: {processor_type}")

    def unregister_processor(self, processor_type: str) -> None:
        """
        Unregister stream processor.

        Args:
            processor_type: Type/name of processor to remove
        """
        if processor_type in self._processors:
            del self._processors[processor_type]
            logger.info(f"Unregistered processor: {processor_type}")
        else:
            logger.warning(f"Processor {processor_type} not found")

    def reset_statistics(self) -> None:
        """Reset streaming statistics."""
        with self._lock:
            self._statistics = {
                "messages_processed": 0,
                "batches_processed": 0,
                "total_processing_time_ms": 0.0,
                "average_latency_ms": 0.0,
                "throughput_messages_per_sec": 0.0,
                "last_update_time": datetime.now(),
                "errors": 0,
            }
        logger.info("Statistics reset")
