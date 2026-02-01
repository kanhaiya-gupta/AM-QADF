"""
Unit tests for BufferManager.

Tests for temporal window and buffer management.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta

from am_qadf.streaming.buffer_manager import (
    BufferManager,
    DataPoint,
)


class TestDataPoint:
    """Test suite for DataPoint dataclass."""

    @pytest.mark.unit
    def test_data_point_creation(self):
        """Test creating DataPoint."""
        data = np.array([1.0, 2.0, 3.0])
        timestamp = datetime.now()
        metadata = {"source": "test"}

        point = DataPoint(data=data, timestamp=timestamp, metadata=metadata)

        assert np.array_equal(point.data, data)
        assert point.timestamp == timestamp
        assert point.metadata == metadata

    @pytest.mark.unit
    def test_data_point_defaults(self):
        """Test creating DataPoint with defaults."""
        data = np.array([1.0])
        timestamp = datetime.now()

        point = DataPoint(data=data, timestamp=timestamp)

        assert np.array_equal(point.data, data)
        assert point.timestamp == timestamp
        assert point.metadata == {}


class TestBufferManager:
    """Test suite for BufferManager class."""

    @pytest.fixture
    def buffer_manager(self):
        """Create a BufferManager instance."""
        return BufferManager(window_size=10, buffer_size=100)

    @pytest.mark.unit
    def test_buffer_manager_creation(self, buffer_manager):
        """Test creating BufferManager."""
        assert buffer_manager is not None
        assert buffer_manager.window_size == 10
        assert buffer_manager.buffer_size == 100
        assert buffer_manager.get_size() == 0

    @pytest.mark.unit
    def test_add_data(self, buffer_manager):
        """Test adding data to buffer."""
        data = np.array([1.0, 2.0, 3.0])
        timestamp = datetime.now()

        buffer_manager.add_data(data, timestamp)

        assert buffer_manager.get_size() == 1
        stats = buffer_manager.get_buffer_statistics()
        assert stats["total_added"] == 1
        assert stats["oldest_timestamp"] == timestamp.isoformat()

    @pytest.mark.unit
    def test_add_data_no_timestamp(self, buffer_manager):
        """Test adding data without timestamp."""
        data = np.array([1.0, 2.0])

        buffer_manager.add_data(data)

        assert buffer_manager.get_size() == 1
        stats = buffer_manager.get_buffer_statistics()
        assert stats["oldest_timestamp"] is not None

    @pytest.mark.unit
    def test_add_data_with_metadata(self, buffer_manager):
        """Test adding data with metadata."""
        data = np.array([1.0])
        metadata = {"source": "test", "signal": "temperature"}

        buffer_manager.add_data(data, metadata=metadata)

        assert buffer_manager.get_size() == 1

    @pytest.mark.unit
    def test_get_window(self, buffer_manager):
        """Test getting window from buffer."""
        # Add data
        for i in range(15):
            data = np.array([float(i)])
            buffer_manager.add_data(data, datetime.now())

        # Get window (should return last 10 points)
        window = buffer_manager.get_window()

        assert len(window) == 10  # window_size is 10
        assert buffer_manager.get_size() == 15  # Buffer still has all data

    @pytest.mark.unit
    def test_get_window_empty(self, buffer_manager):
        """Test getting window from empty buffer."""
        window = buffer_manager.get_window()
        assert len(window) == 0

    @pytest.mark.unit
    def test_get_window_custom_size(self, buffer_manager):
        """Test getting window with custom size."""
        # Add data
        for i in range(20):
            buffer_manager.add_data(np.array([float(i)]))

        # Get custom window size
        window = buffer_manager.get_window(window_size=5)

        assert len(window) == 5

    @pytest.mark.unit
    def test_get_sliding_window(self, buffer_manager):
        """Test getting sliding window."""
        # Add data
        timestamps = []
        for i in range(10):
            ts = datetime.now()
            timestamps.append(ts)
            buffer_manager.add_data(np.array([float(i)]), ts)

        window, window_timestamps, metadata_list = buffer_manager.get_sliding_window(5)

        assert len(window) == 5
        assert len(window_timestamps) == 5
        assert len(metadata_list) == 5
        assert window_timestamps == timestamps[-5:]

    @pytest.mark.unit
    def test_get_time_window(self, buffer_manager):
        """Test getting time-based window."""
        base_time = datetime.now()

        # Add data with different timestamps (ensure recent data)
        for i in range(5):
            ts = base_time - timedelta(seconds=4 - i)  # From 4s ago to now
            buffer_manager.add_data(np.array([float(i)]), ts)

        # Get window for last 3 seconds (should include last 3 items)
        window, timestamps = buffer_manager.get_time_window(3.0)

        assert len(window) > 0  # Should have recent data
        # All timestamps should be within the last 3 seconds from now
        cutoff_time = datetime.now() - timedelta(seconds=3)
        assert all(ts >= cutoff_time for ts in timestamps)

    @pytest.mark.unit
    def test_flush_buffer(self, buffer_manager):
        """Test flushing buffer."""
        # Add data
        timestamps = []
        for i in range(5):
            ts = datetime.now()
            timestamps.append(ts)
            buffer_manager.add_data(np.array([float(i)]), ts, metadata={"index": i})

        data, flushed_timestamps, metadata = buffer_manager.flush_buffer()

        assert buffer_manager.get_size() == 0
        assert len(data) > 0
        assert len(flushed_timestamps) == 5
        assert len(metadata) == 5
        assert metadata[0]["index"] == 0

    @pytest.mark.unit
    def test_flush_buffer_empty(self, buffer_manager):
        """Test flushing empty buffer."""
        data, timestamps, metadata = buffer_manager.flush_buffer()

        assert len(data) == 0
        assert len(timestamps) == 0
        assert len(metadata) == 0

    @pytest.mark.unit
    def test_clear_old_data(self, buffer_manager):
        """Test clearing old data."""
        base_time = datetime.now()

        # Add old data
        for i in range(5):
            ts = base_time - timedelta(seconds=100 + i)
            buffer_manager.add_data(np.array([float(i)]), ts)

        # Add recent data
        for i in range(5):
            ts = base_time - timedelta(seconds=i)
            buffer_manager.add_data(np.array([float(i + 5)]), ts)

        assert buffer_manager.get_size() == 10

        # Clear data older than 50 seconds
        removed = buffer_manager.clear_old_data(50.0)

        assert removed == 5  # Removed old data
        assert buffer_manager.get_size() == 5  # Kept recent data

    @pytest.mark.unit
    def test_get_buffer_statistics(self, buffer_manager):
        """Test getting buffer statistics."""
        # Add data
        timestamps = []
        for i in range(20):
            ts = datetime.now() - timedelta(seconds=20 - i)
            timestamps.append(ts)
            buffer_manager.add_data(np.array([float(i)]), ts)

        stats = buffer_manager.get_buffer_statistics()

        assert stats["current_size"] == 20
        assert stats["max_size"] == 100
        assert stats["utilization_percent"] == 20.0
        assert stats["total_added"] == 20
        assert stats["total_removed"] == 0
        assert stats["oldest_timestamp"] is not None
        assert stats["newest_timestamp"] is not None
        assert stats["age_span_seconds"] > 0

    @pytest.mark.unit
    def test_reset(self, buffer_manager):
        """Test resetting buffer."""
        # Add data
        for i in range(10):
            buffer_manager.add_data(np.array([float(i)]))

        assert buffer_manager.get_size() == 10

        # Reset
        buffer_manager.reset()

        assert buffer_manager.get_size() == 0
        stats = buffer_manager.get_buffer_statistics()
        assert stats["total_added"] == 0
        assert stats["total_removed"] == 0
        assert stats["oldest_timestamp"] is None
        assert stats["newest_timestamp"] is None

    @pytest.mark.unit
    def test_buffer_overflow(self, buffer_manager):
        """Test buffer overflow handling."""
        buffer_manager = BufferManager(window_size=10, buffer_size=5)  # Small buffer

        # Add more data than buffer size
        for i in range(10):
            buffer_manager.add_data(np.array([float(i)]))

        # Buffer should handle overflow (deque with maxlen)
        assert buffer_manager.get_size() == 5  # Should be capped at buffer_size
        stats = buffer_manager.get_buffer_statistics()
        assert stats["overflow_count"] > 0

    @pytest.mark.unit
    def test_is_full(self, buffer_manager):
        """Test checking if buffer is full."""
        buffer_manager = BufferManager(window_size=10, buffer_size=5)

        assert buffer_manager.is_full() is False

        # Fill buffer
        for i in range(5):
            buffer_manager.add_data(np.array([float(i)]))

        assert buffer_manager.is_full() is True

    @pytest.mark.unit
    def test_different_data_shapes(self, buffer_manager):
        """Test handling different data shapes."""
        # Add data with different shapes
        buffer_manager.add_data(np.array([1.0, 2.0, 3.0]))  # 1D
        buffer_manager.add_data(np.array([[1.0, 2.0], [3.0, 4.0]]))  # 2D
        buffer_manager.add_data(np.array([5.0]))  # Scalar-like

        # Should handle gracefully - different shapes get flattened and concatenated
        window = buffer_manager.get_window()
        # When shapes differ, arrays are flattened: [1,2,3] + [1,2,3,4] + [5] = 8 elements
        assert len(window) == 8
