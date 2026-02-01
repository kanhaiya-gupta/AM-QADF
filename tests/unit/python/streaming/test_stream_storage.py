"""
Unit tests for StreamStorage.

Tests for Redis and MongoDB stream storage with mocks.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta

from am_qadf.streaming.stream_storage import StreamStorage


class TestStreamStorage:
    """Test suite for StreamStorage class."""

    @pytest.fixture
    def mock_redis_client(self):
        """Create a mock Redis client."""
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True
        mock_redis.setex = MagicMock()
        mock_redis.get = MagicMock(return_value=None)
        mock_redis.dbsize.return_value = 100
        return mock_redis

    @pytest.fixture
    def mock_mongo_client(self):
        """Create a mock MongoDB client."""
        mock_mongo = MagicMock()
        mock_collection = MagicMock()
        mock_collection.insert_many.return_value = MagicMock(inserted_ids=["id1", "id2"])
        mock_collection.find.return_value = []
        mock_collection.count_documents.return_value = 0
        mock_collection.list_indexes.return_value = []
        mock_collection.delete_many.return_value = MagicMock(deleted_count=0)
        mock_collection.create_index.return_value = "index_name"

        # Mock database and collection access
        mock_db = MagicMock()
        mock_db.get_collection.return_value = mock_collection
        mock_mongo.get_database.return_value = mock_db

        return mock_mongo

    @pytest.mark.unit
    @patch("am_qadf.streaming.stream_storage.REDIS_AVAILABLE", True)
    @patch("am_qadf.streaming.stream_storage.redis")
    def test_storage_creation_with_redis(self, mock_redis_module, mock_redis_client):
        """Test creating StreamStorage with Redis."""
        mock_redis_module.Redis.return_value = mock_redis_client

        storage = StreamStorage()

        assert storage is not None
        assert storage.redis_client is not None

    @pytest.mark.unit
    def test_storage_creation_with_mocks(self, mock_redis_client, mock_mongo_client):
        """Test creating StreamStorage with provided clients."""
        storage = StreamStorage(redis_client=mock_redis_client, mongo_client=mock_mongo_client)

        assert storage is not None
        assert storage.redis_client == mock_redis_client
        assert storage.mongo_client == mock_mongo_client

    @pytest.mark.unit
    def test_cache_recent_data(self, mock_redis_client):
        """Test caching recent data in Redis."""
        storage = StreamStorage(redis_client=mock_redis_client)

        data = {"test": "value", "number": 123}
        storage.cache_recent_data("test_key", data, ttl_seconds=3600)

        mock_redis_client.setex.assert_called_once()
        call_args = mock_redis_client.setex.call_args
        assert call_args[0][0] == "test_key"
        assert call_args[0][1] == 3600

    @pytest.mark.unit
    def test_cache_recent_data_no_redis(self):
        """Test caching when Redis not available."""
        storage = StreamStorage(redis_client=None)

        # Should not raise error, just skip caching
        storage.cache_recent_data("test_key", {"data": 123})

    @pytest.mark.unit
    def test_get_cached_data(self, mock_redis_client):
        """Test getting cached data from Redis."""
        import json

        mock_redis_client.get.return_value = json.dumps({"test": "value"}).encode("utf-8")

        storage = StreamStorage(redis_client=mock_redis_client)

        data = storage.get_cached_data("test_key")

        assert data == {"test": "value"}
        mock_redis_client.get.assert_called_once_with("test_key")

    @pytest.mark.unit
    def test_get_cached_data_not_found(self, mock_redis_client):
        """Test getting cached data that doesn't exist."""
        mock_redis_client.get.return_value = None

        storage = StreamStorage(redis_client=mock_redis_client)

        data = storage.get_cached_data("nonexistent_key")

        assert data is None

    @pytest.mark.unit
    def test_store_batch(self, mock_mongo_client):
        """Test storing batch in MongoDB."""
        storage = StreamStorage(mongo_client=mock_mongo_client)

        batch_data = [
            {"value": 1.0, "timestamp": datetime.now()},
            {"value": 2.0, "timestamp": datetime.now()},
        ]

        inserted_count = storage.store_batch(batch_data)

        assert inserted_count == 2

    @pytest.mark.unit
    def test_store_batch_empty(self, mock_mongo_client):
        """Test storing empty batch."""
        storage = StreamStorage(mongo_client=mock_mongo_client)

        inserted_count = storage.store_batch([])

        assert inserted_count == 0

    @pytest.mark.unit
    def test_store_batch_no_mongodb(self):
        """Test storing batch when MongoDB not available."""
        storage = StreamStorage(mongo_client=None)

        inserted_count = storage.store_batch([{"data": 123}])

        assert inserted_count == 0

    @pytest.mark.unit
    def test_query_stream_history(self, mock_mongo_client):
        """Test querying stream history."""
        # Mock query results
        mock_doc1 = {"_id": "1", "value": 1.0, "timestamp": datetime.now()}
        mock_doc2 = {"_id": "2", "value": 2.0, "timestamp": datetime.now()}

        # Create a mock cursor that supports .sort() chaining
        mock_cursor = MagicMock()
        mock_cursor.sort.return_value = [mock_doc1, mock_doc2]  # sort() returns list directly

        mock_collection = mock_mongo_client.get_database().get_collection()
        # Setup: find() returns cursor, cursor.sort() returns list
        mock_collection.find.return_value = mock_cursor

        storage = StreamStorage(mongo_client=mock_mongo_client)

        start_time = datetime.now() - timedelta(hours=1)
        end_time = datetime.now()

        results = storage.query_stream_history(start_time, end_time)

        assert len(results) == 2
        mock_collection.find.assert_called_once()

    @pytest.mark.unit
    def test_create_time_series_index(self, mock_mongo_client):
        """Test creating time-series index."""
        storage = StreamStorage(mongo_client=mock_mongo_client)

        result = storage.create_time_series_index()

        assert result is True
        mock_collection = mock_mongo_client.get_database().get_collection()
        assert mock_collection.create_index.called

    @pytest.mark.unit
    def test_delete_old_data(self, mock_mongo_client):
        """Test deleting old data."""
        mock_collection = mock_mongo_client.get_database().get_collection()
        mock_collection.delete_many.return_value = MagicMock(deleted_count=5)

        storage = StreamStorage(mongo_client=mock_mongo_client)

        older_than = datetime.now() - timedelta(days=30)
        deleted_count = storage.delete_old_data(older_than)

        assert deleted_count == 5
        mock_collection.delete_many.assert_called_once()

    @pytest.mark.unit
    def test_get_statistics(self, mock_redis_client, mock_mongo_client):
        """Test getting storage statistics."""
        mock_redis_client.info.return_value = {"used_memory_human": "10MB"}
        mock_collection = mock_mongo_client.get_database().get_collection()
        mock_collection.count_documents.return_value = 1000
        mock_collection.list_indexes.return_value = []

        storage = StreamStorage(redis_client=mock_redis_client, mongo_client=mock_mongo_client)

        stats = storage.get_statistics()

        assert stats["redis_available"] is True
        assert stats["mongodb_available"] is True
        assert stats["document_count"] == 1000
        assert "redis_memory_used" in stats
        assert "redis_keys" in stats
