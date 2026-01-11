"""
Test fixtures for streaming module.

Provides test data generators for streaming tests.
"""

from .kafka_test_data import (
    generate_kafka_message,
    generate_kafka_message_batch,
    generate_streaming_data_point,
    generate_streaming_batch,
)

__all__ = [
    "generate_kafka_message",
    "generate_kafka_message_batch",
    "generate_streaming_data_point",
    "generate_streaming_batch",
]
