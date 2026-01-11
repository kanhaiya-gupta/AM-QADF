"""
Test fixtures and data for AM-QADF tests.

Contains test data, mock objects, and fixture utilities.
"""

import pytest
from pathlib import Path

# Import voxel data fixtures
try:
    from .voxel_data import (
        load_small_voxel_grid,
        load_medium_voxel_grid,
        load_large_voxel_grid,
    )

    VOXEL_FIXTURES_AVAILABLE = True
except ImportError:
    VOXEL_FIXTURES_AVAILABLE = False


@pytest.fixture
def small_voxel_grid():
    """Pytest fixture for small voxel grid (10x10x10)."""
    if not VOXEL_FIXTURES_AVAILABLE:
        pytest.skip("Voxel fixtures not available")
    return load_small_voxel_grid()


@pytest.fixture
def medium_voxel_grid():
    """Pytest fixture for medium voxel grid (50x50x50)."""
    if not VOXEL_FIXTURES_AVAILABLE:
        pytest.skip("Voxel fixtures not available")
    return load_medium_voxel_grid()


@pytest.fixture
def large_voxel_grid():
    """Pytest fixture for large voxel grid (100x100x100)."""
    if not VOXEL_FIXTURES_AVAILABLE:
        pytest.skip("Voxel fixtures not available")
    return load_large_voxel_grid()


# Import streaming and monitoring fixtures
try:
    from .streaming import (
        generate_kafka_message,
        generate_kafka_message_batch,
        generate_streaming_data_point,
        generate_streaming_batch,
        generate_streaming_data_with_anomalies,
        generate_time_series_stream,
    )

    STREAMING_FIXTURES_AVAILABLE = True
except ImportError:
    STREAMING_FIXTURES_AVAILABLE = False
    generate_kafka_message = None
    generate_kafka_message_batch = None
    generate_streaming_data_point = None
    generate_streaming_batch = None

try:
    from .monitoring import (
        generate_alert,
        generate_alert_batch,
        generate_alert_history,
        generate_health_status,
        generate_health_history,
        generate_health_metrics,
    )

    MONITORING_FIXTURES_AVAILABLE = True
except ImportError:
    MONITORING_FIXTURES_AVAILABLE = False
    generate_alert = None
    generate_alert_batch = None
    generate_health_status = None


# Export commonly used fixtures
__all__ = [
    "small_voxel_grid",
    "medium_voxel_grid",
    "large_voxel_grid",
    "load_small_voxel_grid",
    "load_medium_voxel_grid",
    "load_large_voxel_grid",
]

# Add streaming fixtures if available
if STREAMING_FIXTURES_AVAILABLE:
    __all__.extend(
        [
            "generate_kafka_message",
            "generate_kafka_message_batch",
            "generate_streaming_data_point",
            "generate_streaming_batch",
        ]
    )

# Add monitoring fixtures if available
if MONITORING_FIXTURES_AVAILABLE:
    __all__.extend(
        [
            "generate_alert",
            "generate_alert_batch",
            "generate_health_status",
        ]
    )
