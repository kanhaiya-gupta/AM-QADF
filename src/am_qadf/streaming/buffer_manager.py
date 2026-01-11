"""
Buffer Manager

Temporal window and buffer management for streaming data.
Provides sliding windows, time-based windowing, and automatic buffer overflow handling.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
from collections import deque
import threading

logger = logging.getLogger(__name__)


@dataclass
class DataPoint:
    """Data point with timestamp."""

    data: np.ndarray
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class BufferManager:
    """
    Manage temporal windows and buffers for streaming data.

    Provides:
    - Sliding window support
    - Time-based windowing
    - Automatic buffer overflow handling
    - Memory-efficient circular buffers
    - Buffer statistics
    """

    def __init__(self, window_size: int, buffer_size: int = 1000):
        """
        Initialize buffer manager.

        Args:
            window_size: Size of temporal window (number of samples)
            buffer_size: Maximum buffer size (default: 1000)
        """
        self.window_size = window_size
        self.buffer_size = buffer_size

        # Circular buffer using deque (more efficient for FIFO)
        self._buffer: deque = deque(maxlen=buffer_size)

        # Thread safety
        self._lock = threading.Lock()

        # Statistics
        self._stats = {
            "total_added": 0,
            "total_removed": 0,
            "overflow_count": 0,
            "oldest_timestamp": None,
            "newest_timestamp": None,
        }

        logger.info(f"BufferManager initialized with window_size={window_size}, buffer_size={buffer_size}")

    def add_data(
        self, data: np.ndarray, timestamp: Optional[datetime] = None, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add data to buffer.

        Args:
            data: Data array to add
            timestamp: Optional timestamp (defaults to now)
            metadata: Optional metadata dictionary
        """
        if timestamp is None:
            timestamp = datetime.now()

        data_point = DataPoint(data=np.asarray(data), timestamp=timestamp, metadata=metadata or {})

        with self._lock:
            # Check for overflow (buffer is full)
            was_full = len(self._buffer) >= self.buffer_size

            # Add to buffer (deque automatically handles maxlen)
            self._buffer.append(data_point)

            # Update statistics
            self._stats["total_added"] += 1

            if was_full:
                self._stats["overflow_count"] += 1
                logger.warning(f"Buffer overflow: buffer_size={self.buffer_size} exceeded")

            # Update timestamp statistics
            if self._stats["oldest_timestamp"] is None or timestamp < self._stats["oldest_timestamp"]:
                self._stats["oldest_timestamp"] = timestamp
            if self._stats["newest_timestamp"] is None or timestamp > self._stats["newest_timestamp"]:
                self._stats["newest_timestamp"] = timestamp

    def get_window(self, window_size: Optional[int] = None) -> np.ndarray:
        """
        Get current temporal window.

        Args:
            window_size: Optional window size (uses self.window_size if None)

        Returns:
            Array of data from window (stacked if multiple points)
        """
        with self._lock:
            if not self._buffer:
                return np.array([])

            size = window_size if window_size is not None else self.window_size
            size = min(size, len(self._buffer))

            # Get last N points
            window_points = list(self._buffer)[-size:]

            if not window_points:
                return np.array([])

            # Stack data arrays
            data_arrays = [point.data for point in window_points]

            # Handle different shapes
            try:
                if all(arr.shape == data_arrays[0].shape for arr in data_arrays):
                    # Same shape, stack
                    return np.stack(data_arrays)
                else:
                    # Different shapes, concatenate flattened
                    return np.concatenate([arr.flatten() for arr in data_arrays])
            except Exception as e:
                logger.warning(f"Error stacking window data: {e}, returning concatenated")
                return np.concatenate([arr.flatten() for arr in data_arrays])

    def get_sliding_window(self, window_size: int) -> Tuple[np.ndarray, List[datetime], List[Dict[str, Any]]]:
        """
        Get sliding window of specified size.

        Args:
            window_size: Size of sliding window

        Returns:
            Tuple of (data_array, timestamps_list, metadata_list)
        """
        with self._lock:
            if not self._buffer:
                return np.array([]), [], []

            size = min(window_size, len(self._buffer))
            window_points = list(self._buffer)[-size:]

            data_arrays = [point.data for point in window_points]
            timestamps = [point.timestamp for point in window_points]
            metadata_list = [point.metadata for point in window_points]

            # Stack data
            try:
                if all(arr.shape == data_arrays[0].shape for arr in data_arrays):
                    data_array = np.stack(data_arrays)
                else:
                    data_array = np.concatenate([arr.flatten() for arr in data_arrays])
            except Exception as e:
                logger.warning(f"Error stacking sliding window data: {e}")
                data_array = np.concatenate([arr.flatten() for arr in data_arrays])

            return data_array, timestamps, metadata_list

    def get_time_window(self, duration_seconds: float) -> Tuple[np.ndarray, List[datetime]]:
        """
        Get time-based window (all data within duration_seconds from now).

        Args:
            duration_seconds: Duration of time window in seconds

        Returns:
            Tuple of (data_array, timestamps_list)
        """
        with self._lock:
            if not self._buffer:
                return np.array([]), []

            cutoff_time = datetime.now() - timedelta(seconds=duration_seconds)

            # Get all points within time window
            window_points = [point for point in self._buffer if point.timestamp >= cutoff_time]

            if not window_points:
                return np.array([]), []

            data_arrays = [point.data for point in window_points]
            timestamps = [point.timestamp for point in window_points]

            # Stack data
            try:
                if all(arr.shape == data_arrays[0].shape for arr in data_arrays):
                    data_array = np.stack(data_arrays)
                else:
                    data_array = np.concatenate([arr.flatten() for arr in data_arrays])
            except Exception as e:
                logger.warning(f"Error stacking time window data: {e}")
                data_array = np.concatenate([arr.flatten() for arr in data_arrays])

            return data_array, timestamps

    def flush_buffer(self) -> Tuple[np.ndarray, List[datetime], List[Dict[str, Any]]]:
        """
        Flush buffer and return all data.

        Returns:
            Tuple of (data_array, timestamps_list, metadata_list)
        """
        with self._lock:
            if not self._buffer:
                return np.array([]), [], []

            points = list(self._buffer)

            data_arrays = [point.data for point in points]
            timestamps = [point.timestamp for point in points]
            metadata_list = [point.metadata for point in points]

            # Stack data
            try:
                if all(arr.shape == data_arrays[0].shape for arr in data_arrays):
                    data_array = np.stack(data_arrays)
                else:
                    data_array = np.concatenate([arr.flatten() for arr in data_arrays])
            except Exception as e:
                logger.warning(f"Error stacking flushed data: {e}")
                data_array = np.concatenate([arr.flatten() for arr in data_arrays])

            # Clear buffer
            self._buffer.clear()
            self._stats["total_removed"] += len(points)
            self._stats["oldest_timestamp"] = None
            self._stats["newest_timestamp"] = None

            logger.info(f"Flushed buffer: {len(points)} points")

            return data_array, timestamps, metadata_list

    def clear_old_data(self, max_age_seconds: float) -> int:
        """
        Clear data older than max_age_seconds.

        Args:
            max_age_seconds: Maximum age in seconds

        Returns:
            Number of points removed
        """
        with self._lock:
            if not self._buffer:
                return 0

            cutoff_time = datetime.now() - timedelta(seconds=max_age_seconds)

            # Remove old points
            initial_size = len(self._buffer)
            self._buffer = deque([point for point in self._buffer if point.timestamp >= cutoff_time], maxlen=self.buffer_size)

            removed = initial_size - len(self._buffer)

            if removed > 0:
                self._stats["total_removed"] += removed
                logger.info(f"Cleared {removed} old data points (age > {max_age_seconds}s)")

                # Update timestamp statistics
                if self._buffer:
                    self._stats["oldest_timestamp"] = min(point.timestamp for point in self._buffer)
                    self._stats["newest_timestamp"] = max(point.timestamp for point in self._buffer)
                else:
                    self._stats["oldest_timestamp"] = None
                    self._stats["newest_timestamp"] = None

            return removed

    def get_buffer_statistics(self) -> Dict[str, Any]:
        """
        Get buffer statistics (size, age, etc.).

        Returns:
            Dictionary with buffer statistics
        """
        with self._lock:
            stats = self._stats.copy()

            stats["current_size"] = len(self._buffer)
            stats["max_size"] = self.buffer_size
            stats["utilization_percent"] = (len(self._buffer) / self.buffer_size * 100) if self.buffer_size > 0 else 0.0

            if self._buffer:
                age_span = (self._stats["newest_timestamp"] - self._stats["oldest_timestamp"]).total_seconds()
                stats["age_span_seconds"] = age_span
                stats["average_age_seconds"] = age_span / len(self._buffer) if len(self._buffer) > 0 else 0.0
            else:
                stats["age_span_seconds"] = 0.0
                stats["average_age_seconds"] = 0.0

            stats["oldest_timestamp"] = (
                self._stats["oldest_timestamp"].isoformat() if self._stats["oldest_timestamp"] else None
            )
            stats["newest_timestamp"] = (
                self._stats["newest_timestamp"].isoformat() if self._stats["newest_timestamp"] else None
            )

            return stats

    def reset(self) -> None:
        """Reset buffer and statistics."""
        with self._lock:
            self._buffer.clear()
            self._stats = {
                "total_added": 0,
                "total_removed": 0,
                "overflow_count": 0,
                "oldest_timestamp": None,
                "newest_timestamp": None,
            }
            logger.info("BufferManager reset")

    def get_size(self) -> int:
        """Get current buffer size."""
        with self._lock:
            return len(self._buffer)

    def is_full(self) -> bool:
        """Check if buffer is full."""
        with self._lock:
            return len(self._buffer) >= self.buffer_size
