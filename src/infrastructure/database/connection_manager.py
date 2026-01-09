"""
Connection Manager

Centralized management of database connections.
Provides singleton pattern for connection reuse across the application.
"""

import logging
from typing import Optional, Dict, Any
from threading import Lock

from ..config.database_config import DatabaseConfig, get_database_configs
from .mongodb_client import MongoDBClient

logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    Centralized connection manager for all database connections.

    Manages connections to MongoDB, Cassandra, Neo4j, Elasticsearch, etc.
    Uses singleton pattern to ensure connections are reused.
    """

    _instance: Optional["ConnectionManager"] = None
    _lock: Lock = Lock()

    def __init__(self, config: Optional[DatabaseConfig] = None, env_name: str = "development"):
        """
        Initialize connection manager.

        Args:
            config: Database configuration (if None, loads from env)
            env_name: Environment name for loading config
        """
        if config is None:
            config = get_database_configs(env_name)

        self.config = config
        self._mongodb_client: Optional[MongoDBClient] = None
        self._connections: Dict[str, Any] = {}
        self._initialized = False

    @classmethod
    def get_instance(cls, env_name: str = "development") -> "ConnectionManager":
        """
        Get singleton instance of ConnectionManager.

        Args:
            env_name: Environment name

        Returns:
            ConnectionManager instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(env_name=env_name)
        return cls._instance

    def initialize(self):
        """Initialize all database connections."""
        if self._initialized:
            logger.warning("ConnectionManager already initialized")
            return

        logger.info("Initializing database connections...")

        # Initialize MongoDB
        if self.config.mongodb:
            try:
                self._mongodb_client = MongoDBClient(self.config.mongodb)
                self._connections["mongodb"] = self._mongodb_client
                logger.info("MongoDB connection initialized")
            except Exception as e:
                logger.error(f"Failed to initialize MongoDB connection: {e}")

        # TODO: Initialize other databases (Cassandra, Neo4j, etc.)
        # if self.config.cassandra:
        #     ...
        # if self.config.neo4j:
        #     ...

        self._initialized = True
        logger.info("Database connections initialized")

    def get_mongodb_client(self) -> Optional[MongoDBClient]:
        """
        Get MongoDB client.

        Returns:
            MongoDBClient instance or None if not configured
        """
        if not self._initialized:
            self.initialize()

        return self._mongodb_client

    def get_connection(self, db_type: str) -> Optional[Any]:
        """
        Get database connection by type.

        Args:
            db_type: Database type ('mongodb', 'cassandra', 'neo4j', etc.)

        Returns:
            Database client instance or None
        """
        if not self._initialized:
            self.initialize()

        return self._connections.get(db_type)

    def close_all(self):
        """Close all database connections."""
        logger.info("Closing all database connections...")

        for db_type, connection in self._connections.items():
            try:
                if hasattr(connection, "close"):
                    connection.close()
                logger.info(f"Closed {db_type} connection")
            except Exception as e:
                logger.error(f"Error closing {db_type} connection: {e}")

        self._connections.clear()
        self._mongodb_client = None
        self._initialized = False
        logger.info("All connections closed")

    def health_check(self) -> Dict[str, Dict[str, Any]]:
        """
        Perform health check on all connections.

        Returns:
            Dictionary mapping database type to health status dictionary
        """
        from .health_check import HealthChecker

        checker = HealthChecker(self)
        return checker.check_all()


# Global singleton instance
_global_manager: Optional[ConnectionManager] = None


def get_connection_manager(env_name: str = "development") -> ConnectionManager:
    """
    Get global connection manager instance.

    Args:
        env_name: Environment name

    Returns:
        ConnectionManager instance
    """
    global _global_manager

    if _global_manager is None:
        _global_manager = ConnectionManager.get_instance(env_name)
        _global_manager.initialize()

    return _global_manager
