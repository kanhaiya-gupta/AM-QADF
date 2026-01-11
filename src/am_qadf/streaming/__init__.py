"""
Streaming Module

Provides real-time data streaming capabilities including:
- Kafka consumer/producer integration
- Incremental processing of streaming data
- Buffer management with temporal windows
- Low-latency stream processing
- Stream data storage (Redis/MongoDB)
"""

from .streaming_client import StreamingClient, StreamingConfig, StreamingResult
from .incremental_processor import IncrementalProcessor
from .buffer_manager import BufferManager
from .stream_processor import StreamProcessor
from .stream_storage import StreamStorage

# Lazy imports for optional dependencies
try:
    from .kafka_consumer import KafkaConsumer

    KAFKA_CONSUMER_AVAILABLE = True
except ImportError:
    KAFKA_CONSUMER_AVAILABLE = False
    KafkaConsumer = None

try:
    from .kafka_producer import KafkaProducer

    KAFKA_PRODUCER_AVAILABLE = True
except ImportError:
    KAFKA_PRODUCER_AVAILABLE = False
    KafkaProducer = None

__all__ = [
    "StreamingClient",
    "StreamingConfig",
    "StreamingResult",
    "IncrementalProcessor",
    "BufferManager",
    "StreamProcessor",
    "StreamStorage",
]

if KAFKA_CONSUMER_AVAILABLE:
    __all__.append("KafkaConsumer")

if KAFKA_PRODUCER_AVAILABLE:
    __all__.append("KafkaProducer")
