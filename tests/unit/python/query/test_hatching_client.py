"""
Unit tests for HatchingClient.

HatchingClient uses C++ (mongocxx) for all queries; no Python fallback.
These tests require am_qadf_native to be built (skip entire module if not).
Query tests skip when MongoDB is not available or not authenticated (no credentials in repo).
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch

pytest.importorskip("am_qadf_native", reason="HatchingClient requires C++ (mongocxx); build am_qadf_native")

from am_qadf.query.hatching_client import HatchingClient
from am_qadf.query.base_query_client import (
    SpatialQuery,
    TemporalQuery,
    QueryResult,
    SignalType,
)


class TestHatchingClient:
    """Test suite for HatchingClient."""

    @pytest.mark.unit
    def test_hatching_client_creation(self):
        """Test creating HatchingClient (uses C++ for queries; no collection_name)."""
        mock_mongo = Mock()
        client = HatchingClient(mongo_client=mock_mongo)

        assert client.mongo_client == mock_mongo
        # data_source defaults to None if not provided
        assert client.data_source is None or client.data_source == "mongodb_warehouse"

    @pytest.mark.unit
    def test_hatching_client_query_empty(self, mock_mongodb_client, query_or_skip_mongodb):
        """Test querying with empty queries (C++ mongocxx path). Skips if MongoDB not available."""
        client = HatchingClient(mongo_client=mock_mongodb_client)
        spatial = SpatialQuery(component_id="test_model")
        result = query_or_skip_mongodb(client, spatial)
        assert isinstance(result, QueryResult)
        assert isinstance(result.points, list)
        assert isinstance(result.signals, dict)

    @pytest.mark.unit
    def test_hatching_client_query_with_spatial(self, mock_mongodb_client, query_or_skip_mongodb):
        """Test querying with spatial constraints (C++ mongocxx path). Skips if MongoDB not available."""
        client = HatchingClient(mongo_client=mock_mongodb_client)
        spatial = SpatialQuery(
            component_id="test_model",
            bbox_min=(0.0, 0.0, 0.0),
            bbox_max=(10.0, 10.0, 10.0),
        )
        result = query_or_skip_mongodb(client, spatial)
        assert isinstance(result, QueryResult)
        assert isinstance(result.points, list)

    @pytest.mark.unit
    def test_hatching_client_query_with_temporal(self, mock_mongodb_client, query_or_skip_mongodb):
        """Test querying with temporal constraints (C++ mongocxx path). Skips if MongoDB not available."""
        client = HatchingClient(mongo_client=mock_mongodb_client)
        spatial = SpatialQuery(component_id="test_model")
        temporal = TemporalQuery(layer_start=0, layer_end=10)
        result = query_or_skip_mongodb(client, spatial, temporal=temporal)
        assert isinstance(result, QueryResult)

    @pytest.mark.unit
    def test_hatching_client_query_with_signal_types(self, mock_mongodb_client, query_or_skip_mongodb):
        """Test querying with specific signal types (C++ mongocxx path). Skips if MongoDB not available."""
        client = HatchingClient(mongo_client=mock_mongodb_client)
        spatial = SpatialQuery(component_id="test_model")
        signal_types = [SignalType.VELOCITY, SignalType.POWER]
        result = query_or_skip_mongodb(client, spatial, signal_types=signal_types)
        assert isinstance(result, QueryResult)
        assert isinstance(result.signals, dict)

    @pytest.mark.unit
    def test_hatching_client_query_model_id(self, mock_mongodb_client, query_or_skip_mongodb):
        """Test querying with model ID (C++ mongocxx path). Skips if MongoDB not available."""
        client = HatchingClient(mongo_client=mock_mongodb_client)
        spatial = SpatialQuery(component_id="test_model_123")
        result = query_or_skip_mongodb(client, spatial)
        assert isinstance(result, QueryResult)

    @pytest.mark.unit
    def test_hatching_client_error_handling(self, mock_mongodb_client):
        """Test that query() without spatial raises (component_id required)."""
        client = HatchingClient(mongo_client=mock_mongodb_client)
        with pytest.raises(ValueError, match="component_id|Spatial query"):
            client.query()
