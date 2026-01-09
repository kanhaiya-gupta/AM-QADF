"""
Connection Pool

Connection pooling utilities for database connections.
"""

import logging
from typing import Dict, Any, Optional
from threading import Lock
from queue import Queue, Empty

logger = logging.getLogger(__name__)


class ConnectionPool:
    """
    Generic connection pool for database connections.

    Manages a pool of reusable database connections to improve performance.
    """

    def __init__(self, factory_func, max_size: int = 10, min_size: int = 2):
        """
        Initialize connection pool.

        Args:
            factory_func: Function to create new connections
            max_size: Maximum pool size
            min_size: Minimum pool size
        """
        self.factory_func = factory_func
        self.max_size = max_size
        self.min_size = min_size
        self._pool: Queue = Queue(maxsize=max_size)
        self._lock = Lock()
        self._created = 0

        # Pre-populate pool
        self._initialize_pool()

    def _initialize_pool(self):
        """Initialize pool with minimum connections."""
        for _ in range(self.min_size):
            try:
                conn = self.factory_func()
                self._pool.put(conn)
                self._created += 1
            except Exception as e:
                logger.error(f"Failed to create initial connection: {e}")

    def get_connection(self) -> Any:
        """
        Get connection from pool.

        Returns:
            Database connection

        Raises:
            RuntimeError: If unable to get connection
        """
        try:
            # Try to get from pool
            conn = self._pool.get_nowait()

            # Check if connection is still valid
            if hasattr(conn, "is_connected") and not conn.is_connected():
                # Connection is dead, create new one
                conn = self.factory_func()
                self._created += 1

            return conn

        except Empty:
            # Pool is empty, create new connection if under max
            with self._lock:
                if self._created < self.max_size:
                    conn = self.factory_func()
                    self._created += 1
                    return conn
                else:
                    # Wait for connection to become available
                    return self._pool.get(timeout=30)

    def return_connection(self, conn: Any):
        """
        Return connection to pool.

        Args:
            conn: Connection to return
        """
        try:
            self._pool.put_nowait(conn)
        except Exception as e:
            logger.warning(f"Failed to return connection to pool: {e}")
            # Connection is discarded

    def close_all(self):
        """Close all connections in pool."""
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                if hasattr(conn, "close"):
                    conn.close()
            except Empty:
                break

        self._created = 0

    def __enter__(self):
        """Context manager entry."""
        return self.get_connection()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # Note: This is a simplified version
        # In practice, you'd need to track which connection was borrowed
        pass
