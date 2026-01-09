"""
Unit tests for UnifiedQueryClient.

Tests for unified multi-source query functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from am_qadf.query.unified_query_client import UnifiedQueryClient
from am_qadf.query.base_query_client import (
    SpatialQuery,
    TemporalQuery,
    QueryResult,
)


class TestUnifiedQueryClient:
    """Test suite for UnifiedQueryClient."""

    @pytest.mark.unit
    def test_unified_query_client_creation(self):
        """Test creating UnifiedQueryClient."""
        mock_mongo = Mock()

        client = UnifiedQueryClient(mongo_client=mock_mongo)

        assert client.mongo_client == mock_mongo
        assert client.stl_client is not None
        assert client.hatching_client is not None
        assert client.laser_client is not None

    @pytest.mark.unit
    def test_unified_query_client_query_all_sources(self, mock_mongodb_client):
        """Test querying all data sources."""
        client = UnifiedQueryClient(mongo_client=mock_mongodb_client)

        # Mock all client queries
        mock_result = QueryResult(points=[(0.0, 0.0, 0.0)], signals={"test": [1.0]}, metadata={})

        client.stl_client.query = Mock(return_value=mock_result)
        client.hatching_client.query = Mock(return_value=mock_result)
        client.laser_client.query = Mock(return_value=mock_result)
        client.ct_client.query = Mock(return_value=mock_result)
        client.ispm_client.query = Mock(return_value=mock_result)

        results = client.query_all_sources(model_id="test_model")

        assert isinstance(results, dict)
        assert "stl" in results
        assert "hatching" in results
        assert "laser" in results
        assert "ct" in results
        assert "ispm" in results

    @pytest.mark.unit
    def test_unified_query_client_query_with_spatial(self, mock_mongodb_client):
        """Test querying with spatial constraints."""
        client = UnifiedQueryClient(mongo_client=mock_mongodb_client)

        # Mock client queries
        mock_result = QueryResult(points=[], signals={}, metadata={})
        client.stl_client.query = Mock(return_value=mock_result)
        client.hatching_client.query = Mock(return_value=mock_result)
        client.laser_client.query = Mock(return_value=mock_result)

        spatial = SpatialQuery(
            component_id="test_model",
            bbox_min=(0.0, 0.0, 0.0),
            bbox_max=(10.0, 10.0, 10.0),
        )

        results = client.query_all_sources(model_id="test_model", spatial=spatial)

        assert isinstance(results, dict)
        # Verify spatial query was passed to clients
        client.hatching_client.query.assert_called_once()

    @pytest.mark.unit
    def test_unified_query_client_query_with_temporal(self, mock_mongodb_client):
        """Test querying with temporal constraints."""
        client = UnifiedQueryClient(mongo_client=mock_mongodb_client)

        # Mock client queries
        mock_result = QueryResult(points=[], signals={}, metadata={})
        client.stl_client.query = Mock(return_value=mock_result)
        client.hatching_client.query = Mock(return_value=mock_result)
        client.laser_client.query = Mock(return_value=mock_result)

        temporal = TemporalQuery(layer_start=0, layer_end=10)

        results = client.query_all_sources(model_id="test_model", temporal=temporal)

        assert isinstance(results, dict)

    @pytest.mark.unit
    def test_unified_query_client_get_coordinate_transformer(self, mock_mongodb_client):
        """Test getting coordinate system transformer."""
        client = UnifiedQueryClient(mongo_client=mock_mongodb_client)

        transformer = client.get_coordinate_transformer()

        # Transformer should be available or None
        assert transformer is None or hasattr(transformer, "transform_point")

    @pytest.mark.unit
    def test_unified_query_client_error_handling(self, mock_mongodb_client):
        """Test error handling in UnifiedQueryClient."""
        client = UnifiedQueryClient(mongo_client=mock_mongodb_client)

        # Mock client to raise error
        client.hatching_client.query = Mock(side_effect=Exception("Query error"))

        # query_all_sources catches errors and puts them in results dict
        results = client.query_all_sources(model_id="test_model")

        # Should have error in hatching results
        assert "hatching" in results
        assert "error" in results["hatching"]
