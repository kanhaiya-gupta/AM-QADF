"""
Stream Storage

Store stream data in Redis (caching) and MongoDB (persistence).
Provides caching for recent data, batch storage, and time-series queries.
"""

from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
import json
import threading

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

try:
    import pymongo
    from pymongo.collection import Collection

    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    pymongo = None
    Collection = None


class StreamStorage:
    """
    Store stream data in Redis and MongoDB.

    Provides:
    - Redis caching for recent data
    - MongoDB persistence for long-term storage
    - Time-series indexing for efficient queries
    - Batch operations for performance
    """

    def __init__(
        self,
        redis_client: Optional[Any] = None,
        mongo_client: Optional[Any] = None,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        collection_name: str = "stream_data",
    ):
        """
        Initialize stream storage.

        Args:
            redis_client: Optional Redis client instance
            mongo_client: Optional MongoDB client instance
            redis_host: Redis host (if creating new client)
            redis_port: Redis port (if creating new client)
            redis_db: Redis database number (if creating new client)
            collection_name: MongoDB collection name
        """
        self.redis_client = redis_client
        self.mongo_client = mongo_client
        self.collection_name = collection_name

        # Initialize Redis if not provided
        if self.redis_client is None and REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(
                    host=redis_host, port=redis_port, db=redis_db, decode_responses=False  # Keep binary for flexibility
                )
                # Test connection
                self.redis_client.ping()
                logger.info(f"Connected to Redis at {redis_host}:{redis_port}")
            except Exception as e:
                logger.warning(f"Could not connect to Redis: {e}")
                self.redis_client = None
        elif self.redis_client is None:
            logger.warning("Redis not available, caching disabled")

        # MongoDB collection
        self._collection: Optional[Collection] = None
        if self.mongo_client is not None:
            try:
                if hasattr(self.mongo_client, "get_database"):
                    db = self.mongo_client.get_database("am_qadf")
                    self._collection = db.get_collection(collection_name)
                elif hasattr(self.mongo_client, "get_collection"):
                    self._collection = self.mongo_client.get_collection(collection_name)
                else:
                    # Assume it's a pymongo database or collection
                    if isinstance(self.mongo_client, Collection):
                        self._collection = self.mongo_client
                    else:
                        self._collection = self.mongo_client[collection_name]

                logger.info(f"MongoDB collection initialized: {collection_name}")
            except Exception as e:
                logger.warning(f"Could not initialize MongoDB collection: {e}")

        self._lock = threading.Lock()

        logger.info("StreamStorage initialized")

    def cache_recent_data(self, key: str, data: Any, ttl_seconds: int = 3600) -> None:
        """
        Cache recent data in Redis.

        Args:
            key: Cache key
            data: Data to cache (will be JSON-serialized)
            ttl_seconds: Time-to-live in seconds (default: 1 hour)
        """
        if self.redis_client is None:
            logger.debug("Redis not available, skipping cache")
            return

        try:
            # Serialize data
            if isinstance(data, (dict, list)):
                serialized = json.dumps(data).encode("utf-8")
            elif isinstance(data, str):
                serialized = data.encode("utf-8")
            elif isinstance(data, bytes):
                serialized = data
            else:
                # Try to serialize as JSON
                serialized = json.dumps(str(data)).encode("utf-8")

            # Store in Redis with TTL
            self.redis_client.setex(key, ttl_seconds, serialized)
            logger.debug(f"Cached data with key: {key} (TTL: {ttl_seconds}s)")

        except Exception as e:
            logger.error(f"Error caching data in Redis: {e}")

    def get_cached_data(self, key: str) -> Optional[Any]:
        """
        Get cached data from Redis.

        Args:
            key: Cache key

        Returns:
            Cached data (deserialized) or None if not found
        """
        if self.redis_client is None:
            return None

        try:
            # Get from Redis
            serialized = self.redis_client.get(key)

            if serialized is None:
                return None

            # Deserialize
            if isinstance(serialized, bytes):
                try:
                    data = json.loads(serialized.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    # Return as string if not JSON
                    data = serialized.decode("utf-8")
            else:
                data = serialized

            logger.debug(f"Retrieved cached data with key: {key}")
            return data

        except Exception as e:
            logger.error(f"Error retrieving cached data from Redis: {e}")
            return None

    def store_batch(self, batch_data: List[Dict], collection_name: Optional[str] = None) -> int:
        """
        Store batch of data in MongoDB.

        Args:
            batch_data: List of data dictionaries to store
            collection_name: Optional collection name (uses default if None)

        Returns:
            Number of documents inserted
        """
        if self._collection is None and self.mongo_client is None:
            logger.warning("MongoDB not available, cannot store batch")
            return 0

        if not batch_data:
            logger.warning("Empty batch, nothing to store")
            return 0

        try:
            # Get collection
            collection = self._collection
            if collection_name and collection_name != self.collection_name:
                # Use different collection
                if hasattr(self.mongo_client, "get_database"):
                    db = self.mongo_client.get_database("am_qadf")
                    collection = db.get_collection(collection_name)
                else:
                    collection = self.mongo_client[collection_name]

            # Prepare documents (add timestamp if not present)
            documents = []
            for item in batch_data:
                doc = dict(item)
                if "_id" not in doc:
                    # Add timestamp if not present
                    if "timestamp" not in doc:
                        doc["timestamp"] = datetime.utcnow()
                    # MongoDB will generate _id automatically
                documents.append(doc)

            # Insert batch
            result = collection.insert_many(documents)
            inserted_count = len(result.inserted_ids)

            logger.info(f"Stored batch: {inserted_count} documents in collection '{collection_name or self.collection_name}'")
            return inserted_count

        except Exception as e:
            logger.error(f"Error storing batch in MongoDB: {e}")
            return 0

    def query_stream_history(
        self, start_time: datetime, end_time: datetime, filters: Optional[Dict] = None, collection_name: Optional[str] = None
    ) -> List[Dict]:
        """
        Query stream history from MongoDB.

        Args:
            start_time: Start time for query
            end_time: End time for query
            filters: Optional additional filters
            collection_name: Optional collection name

        Returns:
            List of documents matching query
        """
        if self._collection is None and self.mongo_client is None:
            logger.warning("MongoDB not available, cannot query history")
            return []

        try:
            # Get collection
            collection = self._collection
            if collection_name and collection_name != self.collection_name:
                if hasattr(self.mongo_client, "get_database"):
                    db = self.mongo_client.get_database("am_qadf")
                    collection = db.get_collection(collection_name)
                else:
                    collection = self.mongo_client[collection_name]

            # Build query
            query = {"timestamp": {"$gte": start_time, "$lte": end_time}}

            # Add additional filters
            if filters:
                query.update(filters)

            # Execute query
            cursor = collection.find(query).sort("timestamp", 1)  # Sort by timestamp ascending
            results = list(cursor)

            logger.info(f"Queried stream history: {len(results)} documents found")
            return results

        except Exception as e:
            logger.error(f"Error querying stream history from MongoDB: {e}")
            return []

    def create_time_series_index(self, collection_name: Optional[str] = None) -> bool:
        """
        Create time-series index for efficient queries.

        Args:
            collection_name: Optional collection name

        Returns:
            True if index created successfully, False otherwise
        """
        if self._collection is None and self.mongo_client is None:
            logger.warning("MongoDB not available, cannot create index")
            return False

        try:
            # Get collection
            collection = self._collection
            if collection_name and collection_name != self.collection_name:
                if hasattr(self.mongo_client, "get_database"):
                    db = self.mongo_client.get_database("am_qadf")
                    collection = db.get_collection(collection_name)
                else:
                    collection = self.mongo_client[collection_name]

            # Create index on timestamp
            collection.create_index([("timestamp", 1)])

            # Create compound index with other common fields
            try:
                collection.create_index([("timestamp", 1), ("topic", 1)])
                collection.create_index([("timestamp", 1), ("source", 1)])
            except Exception:
                # Some fields might not exist, that's okay
                pass

            logger.info(f"Created time-series indexes for collection '{collection_name or self.collection_name}'")
            return True

        except Exception as e:
            logger.error(f"Error creating time-series index: {e}")
            return False

    def delete_old_data(self, older_than: datetime, collection_name: Optional[str] = None) -> int:
        """
        Delete data older than specified time.

        Args:
            older_than: Delete data older than this time
            collection_name: Optional collection name

        Returns:
            Number of documents deleted
        """
        if self._collection is None and self.mongo_client is None:
            logger.warning("MongoDB not available, cannot delete data")
            return 0

        try:
            # Get collection
            collection = self._collection
            if collection_name and collection_name != self.collection_name:
                if hasattr(self.mongo_client, "get_database"):
                    db = self.mongo_client.get_database("am_qadf")
                    collection = db.get_collection(collection_name)
                else:
                    collection = self.mongo_client[collection_name]

            # Delete old documents
            result = collection.delete_many({"timestamp": {"$lt": older_than}})
            deleted_count = result.deleted_count

            logger.info(f"Deleted {deleted_count} old documents from collection '{collection_name or self.collection_name}'")
            return deleted_count

        except Exception as e:
            logger.error(f"Error deleting old data: {e}")
            return 0

    def get_statistics(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get storage statistics.

        Args:
            collection_name: Optional collection name

        Returns:
            Dictionary with statistics
        """
        stats = {
            "redis_available": self.redis_client is not None,
            "mongodb_available": self._collection is not None,
            "collection_name": collection_name or self.collection_name,
        }

        # MongoDB statistics
        if self._collection is not None:
            try:
                collection = self._collection
                if collection_name and collection_name != self.collection_name:
                    if hasattr(self.mongo_client, "get_database"):
                        db = self.mongo_client.get_database("am_qadf")
                        collection = db.get_collection(collection_name)
                    else:
                        collection = self.mongo_client[collection_name]

                stats["document_count"] = collection.count_documents({})
                stats["indexes"] = list(collection.list_indexes())
            except Exception as e:
                logger.warning(f"Error getting MongoDB statistics: {e}")
                stats["document_count"] = None
                stats["indexes"] = []

        # Redis statistics
        if self.redis_client is not None:
            try:
                info = self.redis_client.info("memory")
                stats["redis_memory_used"] = info.get("used_memory_human", "unknown")
                stats["redis_keys"] = self.redis_client.dbsize()
            except Exception as e:
                logger.warning(f"Error getting Redis statistics: {e}")
                stats["redis_memory_used"] = None
                stats["redis_keys"] = None

        return stats
