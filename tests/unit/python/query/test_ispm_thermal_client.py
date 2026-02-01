"""
Unit tests for ISPMThermalClient.

ISPMThermalClient uses C++ (mongocxx) for MongoDB queries; no Python fallback.
Tests require am_qadf_native to be built (skip entire module if not).
"""

import pytest
from unittest.mock import Mock, MagicMock

pytest.importorskip("am_qadf_native", reason="ISPMThermalClient requires C++ (mongocxx); build am_qadf_native")

from am_qadf.query.ispm_thermal_client import ISPMThermalClient
from am_qadf.query.base_query_client import (
    SpatialQuery,
    TemporalQuery,
    QueryResult,
    SignalType,
)


class TestISPMThermalClient:
    """Test suite for ISPMThermalClient."""

    @pytest.mark.unit
    def test_ispm_thermal_client_creation(self):
        """Test creating ISPMThermalClient."""
        mock_mongo = Mock()
        client = ISPMThermalClient(mongo_client=mock_mongo, use_mongodb=True)

        assert client.use_mongodb is True
        assert client.data_source == "mongodb_warehouse"
        assert client.mongo_client == mock_mongo
        assert client.streaming_enabled is False

    @pytest.mark.unit
    def test_ispm_thermal_client_creation_requires_mongodb(self):
        """Test that use_mongodb=True is required."""
        with pytest.raises(ValueError, match="MongoDB backend required"):
            ISPMThermalClient(use_mongodb=False)

    @pytest.mark.unit
    def test_ispm_thermal_client_available_signals(self):
        """Test available signals in ISPMThermalClient."""
        mock_mongo = Mock()
        client = ISPMThermalClient(mongo_client=mock_mongo, use_mongodb=True)
        available = client._available_signals
        assert SignalType.THERMAL in available
        assert SignalType.TEMPERATURE in available
        assert SignalType.POWER in available
        assert SignalType.VELOCITY in available

    @pytest.mark.unit
    def test_ispm_thermal_client_query_requires_spatial(self, mock_mongodb_client):
        """Test that query() requires spatial with component_id."""
        client = ISPMThermalClient(mongo_client=mock_mongodb_client, use_mongodb=True)
        with pytest.raises(ValueError, match="component_id|Spatial query"):
            client.query()

    @pytest.mark.unit
    def test_ispm_thermal_client_query_with_spatial(self, mock_mongodb_client, query_or_skip_mongodb):
        """Test querying with spatial constraints (C++ mongocxx path). Skips if MongoDB not available."""
        client = ISPMThermalClient(mongo_client=mock_mongodb_client, use_mongodb=True)
        spatial = SpatialQuery(
            component_id="test_model",
            bbox_min=(0.0, 0.0, 0.0),
            bbox_max=(10.0, 10.0, 10.0),
        )
        result = query_or_skip_mongodb(client, spatial)
        assert isinstance(result, QueryResult)
        assert isinstance(result.points, list)
        assert isinstance(result.signals, dict)

    @pytest.mark.unit
    def test_ispm_thermal_client_set_mongo_client(self):
        """Test set_mongo_client switches to MongoDB mode."""
        client = ISPMThermalClient(mongo_client=Mock(), use_mongodb=True)
        mock_mongo = Mock()
        client.set_mongo_client(mock_mongo)
        assert client.mongo_client == mock_mongo
        assert client.use_mongodb is True
