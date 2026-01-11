"""
Test fixtures for Kafka test data.

Provides generators for Kafka messages and streaming data.
"""

import numpy as np
from typing import Dict, List, Any
from datetime import datetime, timedelta
import json
import uuid


def generate_kafka_message(
    topic: str = "am_qadf_monitoring",
    value: Dict[str, Any] = None,
    key: str = None,
    timestamp: datetime = None,
    partition: int = 0,
    offset: int = 0,
) -> Dict[str, Any]:
    """
    Generate a single Kafka message dictionary.

    Args:
        topic: Kafka topic name
        value: Message value dictionary (defaults to sample monitoring data)
        key: Optional message key
        timestamp: Optional timestamp (defaults to now)
        partition: Partition number
        offset: Offset number

    Returns:
        Dictionary representing Kafka message
    """
    if timestamp is None:
        timestamp = datetime.now()

    if value is None:
        value = {
            "sensor_id": f"sensor_{uuid.uuid4().hex[:8]}",
            "temperature": np.random.normal(1000.0, 50.0),
            "power": np.random.normal(200.0, 10.0),
            "velocity": np.random.normal(100.0, 5.0),
            "x": np.random.uniform(0.0, 100.0),
            "y": np.random.uniform(0.0, 100.0),
            "z": np.random.uniform(0.0, 100.0),
            "timestamp": timestamp.isoformat(),
        }

    return {
        "topic": topic,
        "partition": partition,
        "offset": offset,
        "timestamp": timestamp,
        "key": key or f"message_{offset}",
        "value": value,
    }


def generate_kafka_message_batch(
    n_messages: int = 10, topic: str = "am_qadf_monitoring", start_time: datetime = None, interval_seconds: float = 0.1
) -> List[Dict[str, Any]]:
    """
    Generate a batch of Kafka messages.

    Args:
        n_messages: Number of messages to generate
        topic: Kafka topic name
        start_time: Optional start time (defaults to now)
        interval_seconds: Time interval between messages

    Returns:
        List of Kafka message dictionaries
    """
    if start_time is None:
        start_time = datetime.now()

    messages = []
    for i in range(n_messages):
        timestamp = start_time + timedelta(seconds=i * interval_seconds)
        message = generate_kafka_message(
            topic=topic,
            timestamp=timestamp,
            offset=i,
        )
        messages.append(message)

    return messages


def generate_streaming_data_point(
    signal_types: List[str] = None, coordinates: tuple = None, timestamp: datetime = None
) -> Dict[str, Any]:
    """
    Generate a single streaming data point.

    Args:
        signal_types: List of signal types to include (defaults to common signals)
        coordinates: Optional (x, y, z) coordinates (defaults to random)
        timestamp: Optional timestamp (defaults to now)

    Returns:
        Dictionary with streaming data point
    """
    if timestamp is None:
        timestamp = datetime.now()

    if signal_types is None:
        signal_types = ["temperature", "power", "velocity"]

    if coordinates is None:
        coordinates = (
            np.random.uniform(0.0, 100.0),
            np.random.uniform(0.0, 100.0),
            np.random.uniform(0.0, 100.0),
        )

    data_point = {
        "x": coordinates[0],
        "y": coordinates[1],
        "z": coordinates[2],
        "timestamp": timestamp,
    }

    # Add signal values
    signal_ranges = {
        "temperature": (800.0, 1200.0),
        "power": (150.0, 250.0),
        "velocity": (80.0, 120.0),
        "density": (0.95, 1.0),
        "laser_power": (180.0, 220.0),
        "scan_speed": (700.0, 900.0),
    }

    for signal_type in signal_types:
        if signal_type in signal_ranges:
            min_val, max_val = signal_ranges[signal_type]
            data_point[signal_type] = np.random.uniform(min_val, max_val)
        else:
            data_point[signal_type] = np.random.uniform(0.0, 100.0)

    return data_point


def generate_streaming_batch(
    n_points: int = 10, signal_types: List[str] = None, start_time: datetime = None, interval_seconds: float = 0.1
) -> List[Dict[str, Any]]:
    """
    Generate a batch of streaming data points.

    Args:
        n_points: Number of data points
        signal_types: List of signal types
        start_time: Optional start time (defaults to now)
        interval_seconds: Time interval between points

    Returns:
        List of streaming data point dictionaries
    """
    if start_time is None:
        start_time = datetime.now()

    batch = []
    for i in range(n_points):
        timestamp = start_time + timedelta(seconds=i * interval_seconds)
        data_point = generate_streaming_data_point(signal_types=signal_types, timestamp=timestamp)
        batch.append(data_point)

    return batch


def generate_streaming_data_with_anomalies(
    n_points: int = 100, anomaly_rate: float = 0.1, signal_types: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Generate streaming data with anomalies.

    Args:
        n_points: Number of data points
        anomaly_rate: Rate of anomalies (0.0-1.0)
        signal_types: List of signal types

    Returns:
        List of streaming data points with anomalies marked
    """
    batch = generate_streaming_batch(n_points, signal_types)

    n_anomalies = int(n_points * anomaly_rate)
    anomaly_indices = np.random.choice(n_points, n_anomalies, replace=False)

    for idx in anomaly_indices:
        # Create anomaly by adding large deviation
        point = batch[idx]
        for signal in signal_types or ["temperature", "power"]:
            if signal in point:
                # Add 3-5 sigma deviation
                point[signal] *= np.random.uniform(1.5, 3.0)
                point["is_anomaly"] = True
                point["anomaly_type"] = f"{signal}_deviation"

    return batch


def generate_time_series_stream(
    duration_seconds: float = 60.0, sample_rate: float = 10.0, signal_types: List[str] = None  # samples per second
) -> List[Dict[str, Any]]:
    """
    Generate time-series stream data.

    Args:
        duration_seconds: Duration of stream in seconds
        sample_rate: Samples per second
        signal_types: List of signal types

    Returns:
        List of streaming data points
    """
    n_points = int(duration_seconds * sample_rate)
    interval_seconds = 1.0 / sample_rate

    return generate_streaming_batch(n_points=n_points, signal_types=signal_types, interval_seconds=interval_seconds)
