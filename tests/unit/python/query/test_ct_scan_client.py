"""
Unit tests for CTScanClient.

Tests for CT scan data query functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from am_qadf.query.ct_scan_client import CTScanClient
from am_qadf.query.base_query_client import (
    SpatialQuery,
    TemporalQuery,
    QueryResult,
    SignalType,
)


class TestCTScanClient:
    """Test suite for CTScanClient."""

    @pytest.mark.unit
    def test_ct_scan_client_creation_in_memory(self):
        """Test creating CTScanClient with in-memory data."""
        ct_volume = np.random.rand(10, 10, 10)
        ct_spacing = (0.1, 0.1, 0.1)
        ct_origin = (0.0, 0.0, 0.0)

        client = CTScanClient(
            ct_volume=ct_volume,
            ct_spacing=ct_spacing,
            ct_origin=ct_origin,
            use_mongodb=False,
        )

        assert client.use_mongodb is False
        assert client.data_source == "in_memory"
        assert np.array_equal(client.ct_volume, ct_volume)
        assert client.ct_spacing == ct_spacing
        assert client.ct_origin == ct_origin

    @pytest.mark.unit
    def test_ct_scan_client_creation_mongodb(self):
        """Test creating CTScanClient with MongoDB."""
        mock_mongo = Mock()
        client = CTScanClient(mongo_client=mock_mongo, use_mongodb=True)

        assert client.use_mongodb is True
        assert client.data_source == "mongodb_warehouse"
        assert client.mongo_client == mock_mongo
        assert SignalType.DENSITY in client._available_signals

    @pytest.mark.unit
    def test_ct_scan_client_query_mongodb_empty(self, mock_mongodb_client):
        """Test querying MongoDB with empty result."""
        client = CTScanClient(mongo_client=mock_mongodb_client, use_mongodb=True)

        # Mock MongoDB query - return empty document (no data but document exists)
        mock_collection = mock_mongodb_client.db["ct_scans"]
        # Create a real dict (not MagicMock) to ensure .get() works correctly
        test_doc = dict(
            {
                "model_id": "test_model",
                "points": [],
                "defect_count": 0,  # Explicitly set to integer
                "defect_locations": [],  # Explicitly set to list
                "metadata": {
                    "statistics": {
                        "spacing": [0.67, 0.67, 0.67],
                        "origin": [0.0, 0.0, 0.0],
                        "dimensions": [30, 30, 30],
                    }
                },
                "coordinate_system": {},
                # No density_values_gridfs_id or porosity_map_gridfs_id fields
            }
        )
        mock_collection.find_one = Mock(return_value=test_doc)
        # Ensure get_file returns None to avoid decompression issues
        mock_mongodb_client.get_file = Mock(return_value=None)

        # MongoDB queries require component_id
        spatial = SpatialQuery(component_id="test_model")
        result = client.query(spatial=spatial)

        assert isinstance(result, QueryResult)
        assert len(result.points) == 0
        # With empty document, should return empty result, not error

    @pytest.mark.unit
    def test_ct_scan_client_query_in_memory(self):
        """Test querying in-memory CT scan data."""
        ct_volume = np.random.rand(5, 5, 5) * 1000.0  # Density values
        ct_spacing = (0.5, 0.5, 0.5)
        ct_origin = (0.0, 0.0, 0.0)

        client = CTScanClient(
            ct_volume=ct_volume,
            ct_spacing=ct_spacing,
            ct_origin=ct_origin,
            use_mongodb=False,
        )

        result = client.query()

        assert isinstance(result, QueryResult)
        assert len(result.points) > 0
        assert "density" in result.signals or "DENSITY" in result.signals

    @pytest.mark.unit
    def test_ct_scan_client_query_with_spatial(self, mock_mongodb_client):
        """Test querying with spatial constraints."""
        client = CTScanClient(mongo_client=mock_mongodb_client, use_mongodb=True)

        # Mock MongoDB query - return doc without gridfs_id to avoid decompression
        mock_collection = mock_mongodb_client.db["ct_scans"]
        # Create a real dict (not MagicMock) to ensure .get() works correctly
        test_doc = dict(
            {
                "model_id": "test_model",
                "points": [],
                "defect_count": 0,  # Explicitly set to integer
                "defect_locations": [],  # Explicitly set to list
                "metadata": {
                    "statistics": {
                        "spacing": [0.67, 0.67, 0.67],
                        "origin": [0.0, 0.0, 0.0],
                        "dimensions": [30, 30, 30],
                    }
                },
                "coordinate_system": {},
                # No density_values_gridfs_id or porosity_map_gridfs_id fields
            }
        )
        mock_collection.find_one = Mock(return_value=test_doc)
        # Ensure get_file returns None to avoid decompression issues
        mock_mongodb_client.get_file = Mock(return_value=None)

        spatial = SpatialQuery(
            component_id="test_model",
            bbox_min=(0.0, 0.0, 0.0),
            bbox_max=(10.0, 10.0, 10.0),
        )

        result = client.query(spatial=spatial)

        assert isinstance(result, QueryResult)

    @pytest.mark.unit
    def test_ct_scan_client_query_with_temporal(self, mock_mongodb_client):
        """Test querying with temporal constraints."""
        client = CTScanClient(mongo_client=mock_mongodb_client, use_mongodb=True)

        # Mock MongoDB query - return doc without gridfs_id to avoid decompression
        mock_collection = mock_mongodb_client.db["ct_scans"]
        # Create a real dict (not MagicMock) to ensure .get() works correctly
        test_doc = dict(
            {
                "model_id": "test_model",
                "points": [],
                "defect_count": 0,  # Explicitly set to integer
                "defect_locations": [],  # Explicitly set to list
                "metadata": {
                    "statistics": {
                        "spacing": [0.67, 0.67, 0.67],
                        "origin": [0.0, 0.0, 0.0],
                        "dimensions": [30, 30, 30],
                    }
                },
                "coordinate_system": {},
                # No density_values_gridfs_id or porosity_map_gridfs_id fields
            }
        )
        mock_collection.find_one = Mock(return_value=test_doc)
        # Ensure get_file returns None to avoid decompression issues
        mock_mongodb_client.get_file = Mock(return_value=None)

        spatial = SpatialQuery(component_id="test_model")
        temporal = TemporalQuery(layer_start=0, layer_end=10)

        result = client.query(spatial=spatial, temporal=temporal)

        assert isinstance(result, QueryResult)

    @pytest.mark.unit
    def test_ct_scan_client_error_handling(self, mock_mongodb_client):
        """Test error handling in CTScanClient."""
        client = CTScanClient(mongo_client=mock_mongodb_client, use_mongodb=True)

        # Mock MongoDB to raise an error
        mock_collection = mock_mongodb_client.db["ct_scans"]
        mock_collection.find.side_effect = Exception("Database error")

        with pytest.raises(Exception):
            client.query()
