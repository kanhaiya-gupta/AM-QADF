"""
Unit tests for query_utils.

Tests for query utility functions.
"""

import pytest
import numpy as np
from datetime import datetime
from am_qadf.query.query_utils import (
    build_spatial_query,
    build_temporal_query,
    build_model_query,
    build_parameter_range_query,
    combine_queries,
    filter_points_in_bbox,
    extract_coordinate_system,
    get_bounding_box_from_doc,
    build_aggregation_pipeline,
    aggregate_by_layer,
)


class TestBuildSpatialQuery:
    """Test suite for build_spatial_query function."""

    @pytest.mark.unit
    def test_build_spatial_query_empty(self):
        """Test building empty spatial query."""
        query = build_spatial_query()

        assert query == {}

    @pytest.mark.unit
    def test_build_spatial_query_with_bbox(self):
        """Test building spatial query with bounding box."""
        bbox_min = (0.0, 0.0, 0.0)
        bbox_max = (10.0, 10.0, 10.0)

        query = build_spatial_query(bbox_min=bbox_min, bbox_max=bbox_max)

        assert "spatial_coordinates" in query
        assert "$elemMatch" in query["spatial_coordinates"]

    @pytest.mark.unit
    def test_build_spatial_query_custom_field_name(self):
        """Test building spatial query with custom field name."""
        bbox_min = (0.0, 0.0, 0.0)
        bbox_max = (10.0, 10.0, 10.0)

        query = build_spatial_query(bbox_min=bbox_min, bbox_max=bbox_max, field_name="custom_coords")

        assert "custom_coords" in query
        assert "spatial_coordinates" not in query


class TestBuildTemporalQuery:
    """Test suite for build_temporal_query function."""

    @pytest.mark.unit
    def test_build_temporal_query_empty(self):
        """Test building empty temporal query."""
        query = build_temporal_query()

        assert query == {}

    @pytest.mark.unit
    def test_build_temporal_query_with_time_range(self):
        """Test building temporal query with time range."""
        time_start = datetime(2024, 1, 1, 0, 0, 0)
        time_end = datetime(2024, 1, 2, 0, 0, 0)

        query = build_temporal_query(time_start=time_start, time_end=time_end)

        assert "timestamp" in query
        assert "$gte" in query["timestamp"]
        assert "$lte" in query["timestamp"]

    @pytest.mark.unit
    def test_build_temporal_query_with_layer_range(self):
        """Test building temporal query with layer range."""
        query = build_temporal_query(layer_start=0, layer_end=10)

        assert "layer_index" in query
        assert "$gte" in query["layer_index"]
        assert "$lte" in query["layer_index"]

    @pytest.mark.unit
    def test_build_temporal_query_with_both(self):
        """Test building temporal query with both time and layer ranges."""
        time_start = datetime(2024, 1, 1, 0, 0, 0)
        time_end = datetime(2024, 1, 2, 0, 0, 0)

        query = build_temporal_query(time_start=time_start, time_end=time_end, layer_start=0, layer_end=10)

        assert "timestamp" in query
        assert "layer_index" in query

    @pytest.mark.unit
    def test_build_temporal_query_custom_fields(self):
        """Test building temporal query with custom field names."""
        time_start = datetime(2024, 1, 1, 0, 0, 0)

        query = build_temporal_query(time_start=time_start, time_field="custom_time", layer_field="custom_layer")

        assert "custom_time" in query
        assert "custom_time" not in query or query.get("custom_time", {}).get("$gte") == time_start


class TestBuildModelQuery:
    """Test suite for build_model_query function."""

    @pytest.mark.unit
    def test_build_model_query_empty(self):
        """Test building empty model query."""
        query = build_model_query()

        assert query == {}

    @pytest.mark.unit
    def test_build_model_query_with_model_id(self):
        """Test building model query with model ID."""
        query = build_model_query(model_id="test_model_123")

        assert query["model_id"] == "test_model_123"

    @pytest.mark.unit
    def test_build_model_query_with_model_name(self):
        """Test building model query with model name."""
        query = build_model_query(model_name="Test Model")

        assert query["model_name"] == "Test Model"

    @pytest.mark.unit
    def test_build_model_query_with_filename(self):
        """Test building model query with filename."""
        query = build_model_query(filename="model.stl")

        assert query["filename"] == "model.stl"

    @pytest.mark.unit
    def test_build_model_query_with_all_fields(self):
        """Test building model query with all fields."""
        query = build_model_query(model_id="test_123", model_name="Test Model", filename="model.stl")

        assert query["model_id"] == "test_123"
        assert query["model_name"] == "Test Model"
        assert query["filename"] == "model.stl"


class TestBuildParameterRangeQuery:
    """Test suite for build_parameter_range_query function."""

    @pytest.mark.unit
    def test_build_parameter_range_query_empty(self):
        """Test building empty parameter range query."""
        query = build_parameter_range_query()

        assert query == {}

    @pytest.mark.unit
    def test_build_parameter_range_query_with_range(self):
        """Test building parameter range query."""
        query = build_parameter_range_query(field_name="laser_power", min_value=100.0, max_value=300.0)

        assert "laser_power" in query
        assert "$gte" in query["laser_power"]
        assert "$lte" in query["laser_power"]
        assert query["laser_power"]["$gte"] == 100.0
        assert query["laser_power"]["$lte"] == 300.0

    @pytest.mark.unit
    def test_build_parameter_range_query_min_only(self):
        """Test building parameter range query with min only."""
        query = build_parameter_range_query(field_name="laser_power", min_value=100.0)

        assert "laser_power" in query
        assert "$gte" in query["laser_power"]
        assert "$lte" not in query["laser_power"]

    @pytest.mark.unit
    def test_build_parameter_range_query_max_only(self):
        """Test building parameter range query with max only."""
        query = build_parameter_range_query(field_name="laser_power", max_value=300.0)

        assert "laser_power" in query
        assert "$lte" in query["laser_power"]
        assert "$gte" not in query["laser_power"]


class TestCombineQueries:
    """Test suite for combine_queries function."""

    @pytest.mark.unit
    def test_combine_queries_empty(self):
        """Test combining empty queries."""
        result = combine_queries()

        assert result == {}

    @pytest.mark.unit
    def test_combine_queries_single(self):
        """Test combining single query."""
        query1 = {"model_id": "test_123"}
        result = combine_queries(query1)

        assert result == query1

    @pytest.mark.unit
    def test_combine_queries_multiple(self):
        """Test combining multiple queries."""
        query1 = {"model_id": "test_123"}
        query2 = {"layer_index": {"$gte": 0, "$lte": 10}}
        query3 = {"timestamp": {"$gte": datetime(2024, 1, 1)}}

        result = combine_queries(query1, query2, query3)

        assert "model_id" in result
        assert "layer_index" in result
        assert "timestamp" in result

    @pytest.mark.unit
    def test_combine_queries_overlapping_keys(self):
        """Test combining queries with overlapping keys (later overwrites)."""
        query1 = {"field": "value1"}
        query2 = {"field": "value2"}

        result = combine_queries(query1, query2)

        assert result["field"] == "value2"  # Later query overwrites


class TestFilterPointsInBbox:
    """Test suite for filter_points_in_bbox function."""

    @pytest.mark.unit
    def test_filter_points_in_bbox_all_inside(self):
        """Test filtering points when all are inside bbox."""
        points = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
        bbox_min = (0.0, 0.0, 0.0)
        bbox_max = (10.0, 10.0, 10.0)

        mask = filter_points_in_bbox(points, bbox_min, bbox_max)

        assert len(mask) == 3
        assert mask.dtype == bool
        assert np.all(mask)  # All should be True

    @pytest.mark.unit
    def test_filter_points_in_bbox_some_inside(self):
        """Test filtering points when some are inside bbox."""
        points = np.array(
            [
                [1.0, 1.0, 1.0],  # Inside
                [15.0, 15.0, 15.0],  # Outside
                [2.0, 2.0, 2.0],  # Inside
            ]
        )
        bbox_min = (0.0, 0.0, 0.0)
        bbox_max = (10.0, 10.0, 10.0)

        mask = filter_points_in_bbox(points, bbox_min, bbox_max)

        assert len(mask) == 3
        assert bool(mask[0]) is True  # Inside
        assert bool(mask[1]) is False  # Outside
        assert bool(mask[2]) is True  # Inside

    @pytest.mark.unit
    def test_filter_points_in_bbox_all_outside(self):
        """Test filtering points when all are outside bbox."""
        points = np.array([[15.0, 15.0, 15.0], [20.0, 20.0, 20.0]])
        bbox_min = (0.0, 0.0, 0.0)
        bbox_max = (10.0, 10.0, 10.0)

        mask = filter_points_in_bbox(points, bbox_min, bbox_max)

        assert len(mask) == 2
        assert np.all(~mask)  # All should be False

    @pytest.mark.unit
    def test_filter_points_in_bbox_empty(self):
        """Test filtering empty points array."""
        points = np.array([]).reshape(0, 3)
        bbox_min = (0.0, 0.0, 0.0)
        bbox_max = (10.0, 10.0, 10.0)

        mask = filter_points_in_bbox(points, bbox_min, bbox_max)

        assert len(mask) == 0
        assert mask.dtype == bool


class TestExtractCoordinateSystem:
    """Test suite for extract_coordinate_system function."""

    @pytest.mark.unit
    def test_extract_coordinate_system_present(self):
        """Test extracting coordinate system when present."""
        doc = {
            "model_id": "test_123",
            "coordinate_system": {
                "type": "build_platform",
                "origin": [0.0, 0.0, 0.0],
                "axes": {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]},
            },
        }

        result = extract_coordinate_system(doc)

        assert result is not None
        assert result["type"] == "build_platform"

    @pytest.mark.unit
    def test_extract_coordinate_system_missing(self):
        """Test extracting coordinate system when missing."""
        doc = {"model_id": "test_123"}

        result = extract_coordinate_system(doc)

        assert result is None


class TestGetBoundingBoxFromDoc:
    """Test suite for get_bounding_box_from_doc function."""

    @pytest.mark.unit
    def test_get_bounding_box_from_doc_present(self):
        """Test getting bounding box when present."""
        doc = {
            "model_id": "test_123",
            "bbox_min": [0.0, 0.0, 0.0],
            "bbox_max": [10.0, 10.0, 10.0],
        }

        result = get_bounding_box_from_doc(doc)

        assert result is not None
        assert result[0] == (0.0, 0.0, 0.0)
        assert result[1] == (10.0, 10.0, 10.0)

    @pytest.mark.unit
    def test_get_bounding_box_from_doc_missing(self):
        """Test getting bounding box when missing."""
        doc = {"model_id": "test_123"}

        result = get_bounding_box_from_doc(doc)

        assert result is None


class TestBuildAggregationPipeline:
    """Test suite for build_aggregation_pipeline function."""

    @pytest.mark.unit
    def test_build_aggregation_pipeline_empty(self):
        """Test building empty aggregation pipeline."""
        pipeline = build_aggregation_pipeline()

        assert isinstance(pipeline, list)
        assert len(pipeline) == 0

    @pytest.mark.unit
    def test_build_aggregation_pipeline_with_match(self):
        """Test building aggregation pipeline with match stage."""
        match_stage = {"model_id": "test_123"}
        pipeline = build_aggregation_pipeline(match_stage=match_stage)

        assert len(pipeline) > 0
        assert any("$match" in stage for stage in pipeline)

    @pytest.mark.unit
    def test_build_aggregation_pipeline_with_group(self):
        """Test building aggregation pipeline with group stage."""
        group_stage = {"_id": "$layer_index", "count": {"$sum": 1}}
        pipeline = build_aggregation_pipeline(group_stage=group_stage)

        assert len(pipeline) > 0
        assert any("$group" in stage for stage in pipeline)


class TestAggregateByLayer:
    """Test suite for aggregate_by_layer function."""

    @pytest.mark.unit
    def test_aggregate_by_layer_basic(self, mock_mongodb_client):
        """Test basic layer aggregation."""
        # Mock collection
        mock_collection = mock_mongodb_client.db["test_collection"]
        mock_collection.aggregate.return_value = [
            {"_id": 0, "value": 200.0},
            {"_id": 1, "value": 250.0},
            {"_id": 2, "value": 300.0},
        ]

        result = aggregate_by_layer(
            collection=mock_collection,
            model_id="test_model",
            field_to_aggregate="laser_power",
            aggregation_type="avg",
        )

        assert isinstance(result, dict)
        assert 0 in result
        assert result[0] == 200.0
        assert result[1] == 250.0
        assert result[2] == 300.0

    @pytest.mark.unit
    def test_aggregate_by_layer_different_aggregation_types(self, mock_mongodb_client):
        """Test layer aggregation with different aggregation types."""
        mock_collection = mock_mongodb_client.db["test_collection"]
        mock_collection.aggregate.return_value = [{"_id": 0, "value": 100.0}]

        # Test min
        result_min = aggregate_by_layer(
            collection=mock_collection,
            model_id="test_model",
            field_to_aggregate="laser_power",
            aggregation_type="min",
        )
        assert isinstance(result_min, dict)

        # Test max
        result_max = aggregate_by_layer(
            collection=mock_collection,
            model_id="test_model",
            field_to_aggregate="laser_power",
            aggregation_type="max",
        )
        assert isinstance(result_max, dict)

        # Test sum
        result_sum = aggregate_by_layer(
            collection=mock_collection,
            model_id="test_model",
            field_to_aggregate="laser_power",
            aggregation_type="sum",
        )
        assert isinstance(result_sum, dict)
