"""
Health Check

Health check utilities for database connections in Docker environments.
"""

import logging
from typing import Dict, Any, List
from .connection_manager import ConnectionManager

logger = logging.getLogger(__name__)


class HealthChecker:
    """
    Health checker for database connections.

    Provides health check functionality for all database connections,
    useful for Docker health checks and monitoring.
    """

    def __init__(self, connection_manager: ConnectionManager):
        """
        Initialize health checker.

        Args:
            connection_manager: ConnectionManager instance
        """
        self.connection_manager = connection_manager

    def check_mongodb(self) -> Dict[str, Any]:
        """
        Check MongoDB health.

        Returns:
            Health status dictionary
        """
        mongodb_client = self.connection_manager.get_mongodb_client()

        if mongodb_client is None:
            return {"status": "not_configured", "message": "MongoDB not configured"}

        return mongodb_client.health_check()

    def check_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Check health of all database connections.

        Returns:
            Dictionary mapping database type to health status
        """
        results = {}

        # Check MongoDB
        try:
            results["mongodb"] = self.check_mongodb()
        except Exception as e:
            results["mongodb"] = {"status": "error", "error": str(e)}

        # TODO: Add checks for other databases
        # results['cassandra'] = self.check_cassandra()
        # results['neo4j'] = self.check_neo4j()
        # results['elasticsearch'] = self.check_elasticsearch()

        return results

    def is_healthy(self) -> bool:
        """
        Check if all configured databases are healthy.

        Returns:
            True if all databases are healthy, False otherwise
        """
        results = self.check_all()

        for db_type, status in results.items():
            if status.get("status") not in ("healthy", "not_configured"):
                return False

        return True


def check_all_connections(env_name: str = "development") -> Dict[str, Dict[str, Any]]:
    """
    Convenience function to check all database connections.

    Args:
        env_name: Environment name

    Returns:
        Dictionary mapping database type to health status
    """
    from .connection_manager import get_connection_manager

    manager = get_connection_manager(env_name)
    checker = HealthChecker(manager)
    return checker.check_all()
