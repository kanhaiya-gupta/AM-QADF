"""
Unit tests for LaserMonitoringClient.

LaserMonitoringClient uses C++ (mongocxx) for MongoDB queries; no Python fallback.
These tests require am_qadf_native to be built (skip entire module if not).
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch

pytest.importorskip("am_qadf_native", reason="LaserMonitoringClient MongoDB path requires C++ (mongocxx); build am_qadf_native")

from am_qadf.query.laser_monitoring_client import LaserMonitoringClient
from am_qadf.query.base_query_client import (
    SpatialQuery,
    TemporalQuery,
    QueryResult,
    SignalType,
)


class TestLaserMonitoringClient:
    """Test suite for LaserMonitoringClient."""

    @pytest.mark.unit
    def test_laser_monitoring_client_creation_generated(self):
        """Test creating LaserMonitoringClient with generated data."""
        client = LaserMonitoringClient(
            stl_part=None,
            generated_layers=[],
            generated_build_styles={},
            use_mongodb=False,
        )

        assert client.use_mongodb is False
        assert client.data_source == "hatching_data"
        assert client.generated_layers == []
        assert client.generated_build_styles == {}

    @pytest.mark.unit
    def test_laser_monitoring_client_creation_mongodb(self):
        """Test creating LaserMonitoringClient with MongoDB."""
        mock_mongo = Mock()
        client = LaserMonitoringClient(mongo_client=mock_mongo, use_mongodb=True)

        assert client.use_mongodb is True
        assert client.data_source == "mongodb_warehouse"
        assert client.mongo_client == mock_mongo

    @pytest.mark.unit
    def test_laser_monitoring_client_query_mongodb(self, mock_mongodb_client, query_or_skip_mongodb):
        """Query with use_mongodb=True uses C++ (mongocxx) client. Skips if MongoDB not available."""
        client = LaserMonitoringClient(mongo_client=mock_mongodb_client, use_mongodb=True)
        spatial = SpatialQuery(component_id="test_model")
        result = query_or_skip_mongodb(client, spatial)
        assert isinstance(result, QueryResult)

    @pytest.mark.unit
    def test_laser_monitoring_client_query_with_spatial(self, mock_mongodb_client, query_or_skip_mongodb):
        """Test querying with spatial constraints (C++ mongocxx path). Skips if MongoDB not available."""
        client = LaserMonitoringClient(mongo_client=mock_mongodb_client, use_mongodb=True)
        spatial = SpatialQuery(
            component_id="test_model",
            bbox_min=(0.0, 0.0, 0.0),
            bbox_max=(10.0, 10.0, 10.0),
        )
        result = query_or_skip_mongodb(client, spatial)
        assert isinstance(result, QueryResult)
        assert isinstance(result.points, list)

    @pytest.mark.unit
    def test_laser_monitoring_client_set_data(self):
        """Test set_data updates generated data."""
        client = LaserMonitoringClient(use_mongodb=False)
        client.set_data(generated_layers=[1, 2, 3])
        assert client.generated_layers == [1, 2, 3]

    @pytest.mark.unit
    def test_laser_monitoring_client_set_mongo_client(self):
        """Test set_mongo_client switches to MongoDB mode."""
        client = LaserMonitoringClient(use_mongodb=False)
        mock_mongo = Mock()
        client.set_mongo_client(mock_mongo)
        assert client.use_mongodb is True
        assert client.mongo_client == mock_mongo
