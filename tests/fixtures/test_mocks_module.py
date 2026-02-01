"""
Tests for mocks fixture module.

Tests the mock classes in tests/fixtures/mocks/
"""

import pytest
import numpy as np

try:
    from tests.fixtures.mocks import (
        MockMongoClient,
        MockCollection,
        MockCursor,
        MockInsertResult,
        MockUnifiedQueryClient,
        MockHatchingClient,
        MockLaserClient,
        MockCTClient,
        MockISPMClient,
        MockSTLClient,
        MockQueryResult,
    )

    MOCKS_AVAILABLE = True
except ImportError:
    MOCKS_AVAILABLE = False


@pytest.mark.skipif(not MOCKS_AVAILABLE, reason="Mocks module not available")
class TestMongoDBMocks:
    """Tests for MongoDB mock classes."""

    def test_mock_insert_result(self):
        """Test MockInsertResult."""
        result = MockInsertResult("test_id_123")
        assert result.inserted_id == "test_id_123"

    def test_mock_cursor(self):
        """Test MockCursor."""
        docs = [{"id": 1}, {"id": 2}, {"id": 3}]
        cursor = MockCursor(docs)

        # Test iteration
        results = list(cursor)
        assert len(results) == 3
        assert results[0]["id"] == 1

        # Test to_list
        cursor2 = MockCursor(docs)
        assert len(cursor2.to_list()) == 3
        assert len(cursor2.to_list(length=2)) == 2

    def test_mock_collection(self):
        """Test MockCollection."""
        collection = MockCollection("test_collection")

        # Test insert
        doc = {"name": "test"}
        result = collection.insert_one(doc)
        assert result.inserted_id is not None

        # Test find by _id (the way MockCollection.find_one works)
        found = collection.find_one({"_id": result.inserted_id})
        assert found is not None
        assert found["name"] == "test"

    def test_mock_mongo_client(self):
        """Test MockMongoClient."""
        client = MockMongoClient()
        # MockMongoClient uses get_collection directly, not get_database
        collection = client.get_collection("test_collection")
        assert collection is not None
        assert isinstance(collection, MockCollection)


@pytest.mark.skipif(not MOCKS_AVAILABLE, reason="Mocks module not available")
class TestQueryClientMocks:
    """Tests for query client mock classes."""

    def test_mock_query_result(self):
        """Test MockQueryResult."""
        points = np.array([[1, 2, 3], [4, 5, 6]])
        signals = {"power": np.array([10, 20])}
        metadata = {"model_id": "test"}

        result = MockQueryResult(points=points, signals=signals, metadata=metadata)
        assert result.points.shape == (2, 3)
        assert "power" in result.signals
        assert result.metadata["model_id"] == "test"

    def test_mock_query_result_defaults(self):
        """Test MockQueryResult with defaults."""
        result = MockQueryResult()
        assert result.points.shape == (0, 3)
        assert result.signals == {}
        assert result.metadata == {}

    def test_mock_stl_client(self):
        """Test MockSTLClient."""
        client = MockSTLClient()
        bbox = client.get_model_bounding_box()
        assert bbox is not None

        metadata = client.get_model_metadata()
        assert metadata is not None

    def test_mock_hatching_client(self):
        """Test MockHatchingClient."""
        client = MockHatchingClient()
        layers = client.get_layers()
        assert layers is not None

        result = client.query(model_id="test")
        assert result is not None

    def test_mock_laser_client(self):
        """Test MockLaserClient."""
        client = MockLaserClient()
        result = client.query(model_id="test")
        assert result is not None
        assert hasattr(result, "points")

    def test_mock_ct_client(self):
        """Test MockCTClient."""
        client = MockCTClient()
        result = client.query(model_id="test")
        assert result is not None

    def test_mock_unified_query_client(self):
        """Test MockUnifiedQueryClient."""
        client = MockUnifiedQueryClient()
        result = client.query(model_id="test")
        assert result is not None
