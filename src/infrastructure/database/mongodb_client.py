"""
MongoDB Client Wrapper

MongoDB-specific client wrapper with connection pooling and health checks.
"""

import logging
from typing import Optional, Dict, Any
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from gridfs import GridFS

from ..config.database_config import MongoDBConfig

logger = logging.getLogger(__name__)


class MongoDBClient:
    """
    MongoDB client wrapper with connection pooling and error handling.

    Provides a clean interface for MongoDB operations with automatic
    connection management and health checks.
    """

    def __init__(self, config: MongoDBConfig):
        """
        Initialize MongoDB client.

        Args:
            config: MongoDB configuration
        """
        self.config = config
        self._client: Optional[MongoClient] = None
        self._database: Optional[Database] = None
        self._gridfs: Optional[GridFS] = None
        self._connected = False

        self._connect()

    def _connect(self):
        """Establish MongoDB connection."""
        try:
            # Build connection options
            options = {
                "maxPoolSize": self.config.max_pool_size,
                "minPoolSize": self.config.min_pool_size,
                "maxIdleTimeMS": self.config.max_idle_time_ms,
                "connectTimeoutMS": self.config.connect_timeout_ms,
                "socketTimeoutMS": self.config.socket_timeout_ms,
                "serverSelectionTimeoutMS": self.config.server_selection_timeout_ms,
                "retryWrites": self.config.retry_writes,
                "retryReads": self.config.retry_reads,
            }

            # Add SSL options if enabled
            if self.config.ssl:
                options["ssl"] = True
                if self.config.ssl_certfile:
                    options["ssl_certfile"] = self.config.ssl_certfile
                if self.config.ssl_keyfile:
                    options["ssl_keyfile"] = self.config.ssl_keyfile
                if self.config.ssl_ca_certs:
                    options["ssl_ca_certs"] = self.config.ssl_ca_certs

            # Create client
            self._client = MongoClient(self.config.url, **options)

            # Test connection
            self._client.admin.command("ping")

            # Get database
            self._database = self._client[self.config.database]

            # Initialize GridFS
            self._gridfs = GridFS(self._database)

            self._connected = True
            logger.info(f"MongoDB connected to {self.config.database}")

        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            self._connected = False
            raise
        except Exception as e:
            logger.error(f"Unexpected error connecting to MongoDB: {e}")
            self._connected = False
            raise

    @property
    def client(self) -> MongoClient:
        """
        Get MongoDB client instance.

        Returns:
            MongoClient instance

        Raises:
            RuntimeError: If not connected
        """
        if not self._connected or self._client is None:
            raise RuntimeError("MongoDB client not connected. Call connect() first.")
        return self._client

    @property
    def database(self) -> Database:
        """
        Get MongoDB database instance.

        Returns:
            Database instance

        Raises:
            RuntimeError: If not connected
        """
        if not self._connected or self._database is None:
            raise RuntimeError("MongoDB database not available. Call connect() first.")
        return self._database

    def get_collection(self, collection_name: str) -> Collection:
        """
        Get MongoDB collection.

        Args:
            collection_name: Name of the collection

        Returns:
            Collection instance
        """
        return self.database[collection_name]

    def is_connected(self) -> bool:
        """
        Check if MongoDB is connected.

        Returns:
            True if connected, False otherwise
        """
        if not self._connected or self._client is None:
            return False

        try:
            self._client.admin.command("ping")
            return True
        except Exception:
            self._connected = False
            return False

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on MongoDB connection.

        Returns:
            Dictionary with health status information
        """
        try:
            if not self.is_connected():
                return {"status": "unhealthy", "error": "Not connected to MongoDB"}

            # Get server info
            server_info = self._client.server_info()

            # Get database stats
            db_stats = self.database.command("dbStats")

            return {
                "status": "healthy",
                "server_version": server_info.get("version"),
                "database": self.config.database,
                "collections": db_stats.get("collections", 0),
                "data_size": db_stats.get("dataSize", 0),
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def store_file(self, data: bytes, filename: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store a file in GridFS.

        Args:
            data: File data as bytes
            filename: Filename for the file
            metadata: Optional metadata dictionary

        Returns:
            GridFS file ID as string
        """
        if not self.is_connected() or self._gridfs is None:
            raise RuntimeError("MongoDB client not connected. Cannot store file.")

        try:
            file_id = self._gridfs.put(data, filename=filename, metadata=metadata or {})
            logger.debug(f"Stored file in GridFS: {filename} (ID: {file_id})")
            return str(file_id)
        except Exception as e:
            logger.error(f"Failed to store file in GridFS: {e}")
            raise

    def get_file(self, file_id: str) -> Optional[bytes]:
        """
        Retrieve a file from GridFS.

        Args:
            file_id: GridFS file ID (as string or ObjectId)

        Returns:
            File data as bytes, or None if not found
        """
        if not self.is_connected() or self._gridfs is None:
            raise RuntimeError("MongoDB client not connected. Cannot retrieve file.")

        try:
            from bson import ObjectId

            # Convert string to ObjectId if needed
            file_id_obj: Any = file_id
            if isinstance(file_id, str):
                try:
                    file_id_obj = ObjectId(file_id)
                except Exception:
                    # Try to find by filename if ObjectId conversion fails
                    grid_file = self._gridfs.find_one({"filename": file_id})
                    if grid_file:
                        return grid_file.read()
                    return None

            grid_file = self._gridfs.get(file_id_obj)
            return grid_file.read()
        except Exception as e:
            logger.warning(f"Failed to retrieve file from GridFS: {e}")
            return None

    def delete_file(self, file_id: str) -> bool:
        """
        Delete a file from GridFS.

        Args:
            file_id: GridFS file ID (as string or ObjectId)

        Returns:
            True if deleted, False if not found
        """
        if not self.is_connected() or self._gridfs is None:
            raise RuntimeError("MongoDB client not connected. Cannot delete file.")

        try:
            from bson import ObjectId

            # Convert string to ObjectId if needed
            file_id_obj: Any = file_id
            if isinstance(file_id, str):
                try:
                    file_id_obj = ObjectId(file_id)
                except Exception:
                    return False

            self._gridfs.delete(file_id_obj)
            logger.debug(f"Deleted file from GridFS: {file_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to delete file from GridFS: {e}")
            return False

    def close(self):
        """Close MongoDB connection."""
        if self._client:
            try:
                self._client.close()
                logger.info("MongoDB connection closed")
            except Exception as e:
                logger.error(f"Error closing MongoDB connection: {e}")
            finally:
                self._client = None
                self._database = None
                self._gridfs = None
                self._connected = False

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
