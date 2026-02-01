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

        # Mock all client queries (UnifiedQueryClient has stl, hatching, laser_client, ct_client; ispm_* clients are separate)
        mock_result = QueryResult(points=[(0.0, 0.0, 0.0)], signals={"test": [1.0]}, metadata={})

        client.stl_client.query = Mock(return_value=mock_result)
        client.hatching_client.query = Mock(return_value=mock_result)
        client.laser_client.query = Mock(return_value=mock_result)
        client.ct_client.query = Mock(return_value=mock_result)

        results = client.query_all_sources(model_id="test_model")

        assert isinstance(results, dict)
        assert "stl" in results
        assert "hatching" in results
        assert "laser_monitoring" in results
        assert "ct" in results

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

        # use_cache=False so we don't get a cached result from another test
        results = client.query_all_sources(model_id="test_model", use_cache=False)
        assert "hatching" in results
        assert isinstance(results["hatching"], dict), "expected error dict, got %s" % type(results["hatching"])
        assert "error" in results["hatching"]

    @pytest.mark.unit
    def test_query_source_hatching(self, mock_mongodb_client):
        """Test _query_source returns QueryResult for hatching when client available."""
        client = UnifiedQueryClient(mongo_client=mock_mongodb_client)
        mock_result = QueryResult(
            points=[(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)],
            signals={"power": [1.0, 2.0]},
            metadata={},
        )
        client.hatching_client.query = Mock(return_value=mock_result)
        out = client._query_source("model_1", "hatching")
        assert out is not None
        assert len(out.points) == 2
        assert out.signals.get("power") == [1.0, 2.0]
        client.hatching_client.query.assert_called_once()

    @pytest.mark.unit
    def test_query_source_unknown_returns_none(self, mock_mongodb_client):
        """Test _query_source returns None for unknown source type."""
        client = UnifiedQueryClient(mongo_client=mock_mongodb_client)
        assert client._query_source("model_1", "unknown_source") is None

    @pytest.mark.unit
    def test_query_and_transform_points_requires_native(self, mock_mongodb_client):
        """Test query_and_transform_points raises if am_qadf_native not available."""
        client = UnifiedQueryClient(mongo_client=mock_mongodb_client)
        mock_result = QueryResult(
            points=[(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)],
            signals={"temperature": [0.0, 1.0]},
            metadata={},
        )
        client.hatching_client.query = Mock(return_value=mock_result)
        client.ispm_thermal_client.query = Mock(return_value=mock_result) if client.ispm_thermal_client else None
        with patch.dict("sys.modules", {"am_qadf_native": None}):
            with pytest.raises(ImportError, match="am_qadf_native"):
                client.query_and_transform_points(
                    "model_1",
                    source_types=["hatching"],
                    reference_source="hatching",
                )

    @pytest.mark.unit
    def test_query_and_transform_points_single_source(self, mock_mongodb_client):
        """Test query_and_transform_points with single source (reference only); requires C++ (am_qadf_native)."""
        pytest.importorskip("am_qadf_native", reason="query_and_transform_points requires C++; build am_qadf_native")
        client = UnifiedQueryClient(mongo_client=mock_mongodb_client)
        pts = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        mock_result = QueryResult(
            points=pts,
            signals={"power": [0.0, 1.0, 2.0]},
            metadata={},
        )
        client.hatching_client.query = Mock(return_value=mock_result)
        out = client.query_and_transform_points(
            "model_1",
            source_types=["hatching"],
            reference_source="hatching",
        )
        assert "transformed_points" in out
        assert "hatching" in out["transformed_points"]
        assert out["transformed_points"]["hatching"].shape == (3, 3)
        assert "unified_bounds" in out
        assert "transformations" in out
        assert "validation_results" in out
        assert "raw_results" in out
        assert "signals" in out
        assert out["signals"]["hatching"]["power"].shape == (3,)