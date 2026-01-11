"""
Monitoring Storage

Alert and notification history storage in MongoDB.
Provides storage for alerts, notifications, and health history.
"""

from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
import threading

logger = logging.getLogger(__name__)

# Try to import MongoDB
try:
    import pymongo
    from pymongo.collection import Collection

    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    pymongo = None
    Collection = None

from .alert_system import Alert
from .health_monitor import HealthStatus


class MonitoringStorage:
    """
    Alert and notification history storage.

    Provides:
    - Alert history storage
    - Notification history storage
    - Health status history storage
    - Time-series queries
    - Efficient indexing
    """

    def __init__(
        self,
        mongo_client: Optional[Any] = None,
        database_name: str = "am_qadf",
        alerts_collection: str = "monitoring_alerts",
        notifications_collection: str = "monitoring_notifications",
        health_collection: str = "monitoring_health",
    ):
        """
        Initialize monitoring storage.

        Args:
            mongo_client: MongoDB client instance
            database_name: Database name
            alerts_collection: Collection name for alerts
            notifications_collection: Collection name for notifications
            health_collection: Collection name for health status
        """
        self.mongo_client = mongo_client
        self.database_name = database_name

        # Collections
        self._alerts_collection: Optional[Collection] = None
        self._notifications_collection: Optional[Collection] = None
        self._health_collection: Optional[Collection] = None

        if self.mongo_client is not None:
            try:
                # Get database
                if hasattr(self.mongo_client, "get_database"):
                    db = self.mongo_client.get_database(database_name)
                elif hasattr(self.mongo_client, "get_collection"):
                    # Assume it's a database
                    db = self.mongo_client
                else:
                    # Assume it's a pymongo database
                    db = self.mongo_client[database_name] if isinstance(self.mongo_client, dict) else self.mongo_client

                # Get collections
                if hasattr(db, "get_collection"):
                    self._alerts_collection = db.get_collection(alerts_collection)
                    self._notifications_collection = db.get_collection(notifications_collection)
                    self._health_collection = db.get_collection(health_collection)
                else:
                    self._alerts_collection = db[alerts_collection]
                    self._notifications_collection = db[notifications_collection]
                    self._health_collection = db[health_collection]

                # Create indexes
                self._create_indexes()

                logger.info(f"MonitoringStorage initialized with database: {database_name}")
            except Exception as e:
                logger.warning(f"Could not initialize MongoDB collections: {e}")
        else:
            logger.warning("MongoDB client not provided, storage disabled")

        self._lock = threading.Lock()

    def _create_indexes(self) -> None:
        """Create indexes for efficient queries."""
        try:
            # Alerts indexes
            if self._alerts_collection:
                self._alerts_collection.create_index([("timestamp", 1)])
                self._alerts_collection.create_index([("alert_type", 1)])
                self._alerts_collection.create_index([("severity", 1)])
                self._alerts_collection.create_index([("acknowledged", 1)])
                self._alerts_collection.create_index([("timestamp", 1), ("severity", 1)])

            # Notifications indexes
            if self._notifications_collection:
                self._notifications_collection.create_index([("timestamp", 1)])
                self._notifications_collection.create_index([("channel", 1)])
                self._notifications_collection.create_index([("alert_id", 1)])

            # Health indexes
            if self._health_collection:
                self._health_collection.create_index([("timestamp", 1)])
                self._health_collection.create_index([("component_name", 1)])
                self._health_collection.create_index([("status", 1)])
                self._health_collection.create_index([("component_name", 1), ("timestamp", 1)])

            logger.info("Created indexes for monitoring collections")
        except Exception as e:
            logger.warning(f"Error creating indexes: {e}")

    def save_alert(self, alert: Alert) -> bool:
        """
        Save alert to storage.

        Args:
            alert: Alert object to save

        Returns:
            True if saved successfully, False otherwise
        """
        if self._alerts_collection is None:
            logger.debug("MongoDB not available, alert not saved")
            return False

        try:
            # Convert alert to document
            doc = {
                "alert_id": alert.alert_id,
                "alert_type": alert.alert_type,
                "severity": alert.severity,
                "message": alert.message,
                "timestamp": alert.timestamp,
                "source": alert.source,
                "metadata": alert.metadata,
                "acknowledged": alert.acknowledged,
                "acknowledged_by": alert.acknowledged_by,
                "acknowledged_at": alert.acknowledged_at,
                "created_at": datetime.utcnow(),
            }

            # Insert or update
            self._alerts_collection.update_one({"alert_id": alert.alert_id}, {"$set": doc}, upsert=True)

            logger.debug(f"Saved alert: {alert.alert_id}")
            return True

        except Exception as e:
            logger.error(f"Error saving alert: {e}")
            return False

    def get_alert(self, alert_id: str) -> Optional[Dict]:
        """
        Get alert by ID.

        Args:
            alert_id: Alert ID

        Returns:
            Alert document or None if not found
        """
        if self._alerts_collection is None:
            return None

        try:
            doc = self._alerts_collection.find_one({"alert_id": alert_id})
            return doc
        except Exception as e:
            logger.error(f"Error getting alert: {e}")
            return None

    def query_alerts(
        self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None, filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Query alerts from storage.

        Args:
            start_time: Optional start time
            end_time: Optional end time
            filters: Optional additional filters

        Returns:
            List of alert documents
        """
        if self._alerts_collection is None:
            return []

        try:
            # Build query
            query = {}

            if start_time or end_time:
                query["timestamp"] = {}
                if start_time:
                    query["timestamp"]["$gte"] = start_time
                if end_time:
                    query["timestamp"]["$lte"] = end_time

            if filters:
                query.update(filters)

            # Execute query
            cursor = self._alerts_collection.find(query).sort("timestamp", -1)  # Most recent first
            results = list(cursor)

            logger.debug(f"Queried alerts: {len(results)} found")
            return results

        except Exception as e:
            logger.error(f"Error querying alerts: {e}")
            return []

    def save_notification(
        self, alert_id: str, channel: str, success: bool, timestamp: Optional[datetime] = None, metadata: Optional[Dict] = None
    ) -> bool:
        """
        Save notification record to storage.

        Args:
            alert_id: Alert ID
            channel: Notification channel ('email', 'sms', 'dashboard')
            success: Whether notification was successful
            timestamp: Optional timestamp (defaults to now)
            metadata: Optional metadata

        Returns:
            True if saved successfully, False otherwise
        """
        if self._notifications_collection is None:
            logger.debug("MongoDB not available, notification not saved")
            return False

        try:
            doc = {
                "alert_id": alert_id,
                "channel": channel,
                "success": success,
                "timestamp": timestamp or datetime.utcnow(),
                "metadata": metadata or {},
                "created_at": datetime.utcnow(),
            }

            self._notifications_collection.insert_one(doc)

            logger.debug(f"Saved notification: {channel} for alert {alert_id}")
            return True

        except Exception as e:
            logger.error(f"Error saving notification: {e}")
            return False

    def query_notifications(
        self,
        alert_id: Optional[str] = None,
        channel: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Dict]:
        """
        Query notifications from storage.

        Args:
            alert_id: Optional alert ID filter
            channel: Optional channel filter
            start_time: Optional start time
            end_time: Optional end time

        Returns:
            List of notification documents
        """
        if self._notifications_collection is None:
            return []

        try:
            # Build query
            query = {}

            if alert_id:
                query["alert_id"] = alert_id
            if channel:
                query["channel"] = channel
            if start_time or end_time:
                query["timestamp"] = {}
                if start_time:
                    query["timestamp"]["$gte"] = start_time
                if end_time:
                    query["timestamp"]["$lte"] = end_time

            # Execute query
            cursor = self._notifications_collection.find(query).sort("timestamp", -1)
            results = list(cursor)

            logger.debug(f"Queried notifications: {len(results)} found")
            return results

        except Exception as e:
            logger.error(f"Error querying notifications: {e}")
            return []

    def save_health_status(self, health_status: HealthStatus) -> bool:
        """
        Save health status to storage.

        Args:
            health_status: HealthStatus object to save

        Returns:
            True if saved successfully, False otherwise
        """
        if self._health_collection is None:
            logger.debug("MongoDB not available, health status not saved")
            return False

        try:
            # Convert health status to document
            doc = {
                "component_name": health_status.component_name,
                "status": health_status.status,
                "health_score": health_status.health_score,
                "timestamp": health_status.timestamp,
                "metrics": health_status.metrics,
                "issues": health_status.issues,
                "metadata": health_status.metadata,
                "created_at": datetime.utcnow(),
            }

            self._health_collection.insert_one(doc)

            logger.debug(f"Saved health status: {health_status.component_name}")
            return True

        except Exception as e:
            logger.error(f"Error saving health status: {e}")
            return False

    def query_health_history(self, component_name: str, start_time: datetime, end_time: datetime) -> List[Dict]:
        """
        Query health history for component.

        Args:
            component_name: Component name
            start_time: Start time
            end_time: End time

        Returns:
            List of health status documents
        """
        if self._health_collection is None:
            return []

        try:
            # Build query
            query = {"component_name": component_name, "timestamp": {"$gte": start_time, "$lte": end_time}}

            # Execute query
            cursor = self._health_collection.find(query).sort("timestamp", 1)  # Chronological order
            results = list(cursor)

            logger.debug(f"Queried health history for {component_name}: {len(results)} records found")
            return results

        except Exception as e:
            logger.error(f"Error querying health history: {e}")
            return []

    def delete_old_data(self, older_than: datetime, collection: Optional[str] = None) -> Dict[str, int]:
        """
        Delete old data from storage.

        Args:
            older_than: Delete data older than this time
            collection: Optional collection name ('alerts', 'notifications', 'health', or None for all)

        Returns:
            Dictionary mapping collection names to deleted counts
        """
        results = {}

        try:
            if collection is None or collection == "alerts":
                if self._alerts_collection:
                    result = self._alerts_collection.delete_many({"timestamp": {"$lt": older_than}})
                    results["alerts"] = result.deleted_count

            if collection is None or collection == "notifications":
                if self._notifications_collection:
                    result = self._notifications_collection.delete_many({"timestamp": {"$lt": older_than}})
                    results["notifications"] = result.deleted_count

            if collection is None or collection == "health":
                if self._health_collection:
                    result = self._health_collection.delete_many({"timestamp": {"$lt": older_than}})
                    results["health"] = result.deleted_count

            logger.info(f"Deleted old data: {results}")
            return results

        except Exception as e:
            logger.error(f"Error deleting old data: {e}")
            return results

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get storage statistics.

        Returns:
            Dictionary with statistics
        """
        stats = {
            "mongodb_available": self._alerts_collection is not None,
            "database_name": self.database_name,
        }

        try:
            if self._alerts_collection:
                stats["alerts_count"] = self._alerts_collection.count_documents({})
                stats["unacknowledged_alerts"] = self._alerts_collection.count_documents({"acknowledged": False})

            if self._notifications_collection:
                stats["notifications_count"] = self._notifications_collection.count_documents({})

            if self._health_collection:
                stats["health_records_count"] = self._health_collection.count_documents({})
        except Exception as e:
            logger.warning(f"Error getting statistics: {e}")
            stats["error"] = str(e)

        return stats
