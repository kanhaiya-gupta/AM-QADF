"""
Unit tests for BaseQueryClient and related classes.

Tests for SignalType, SpatialQuery, TemporalQuery, QueryResult, and BaseQueryClient.
"""

import pytest
from typing import Optional, List
from am_qadf.query.base_query_client import (
    SignalType,
    SpatialQuery,
    TemporalQuery,
    QueryResult,
    BaseQueryClient,
)
from am_qadf.core.exceptions import QueryError


class TestSignalType:
    """Test suite for SignalType enum."""

    @pytest.mark.unit
    def test_signal_type_values(self):
        """Test SignalType enum values."""
        assert SignalType.POWER.value == "power"
        assert SignalType.VELOCITY.value == "velocity"
        assert SignalType.ENERGY.value == "energy"
        assert SignalType.THERMAL.value == "thermal"
        assert SignalType.TEMPERATURE.value == "temperature"
        assert SignalType.DENSITY.value == "density"
        assert SignalType.STRESS.value == "stress"

    @pytest.mark.unit
    def test_signal_type_enumeration(self):
        """Test that SignalType can be enumerated."""
        signal_types = list(SignalType)
        assert len(signal_types) == 7
        assert SignalType.POWER in signal_types


class TestSpatialQuery:
    """Test suite for SpatialQuery dataclass."""

    @pytest.mark.unit
    def test_spatial_query_creation_empty(self):
        """Test creating empty SpatialQuery."""
        query = SpatialQuery()

        assert query.bbox_min is None
        assert query.bbox_max is None
        assert query.component_id is None
        assert query.layer_range is None
        assert query.is_empty() is True

    @pytest.mark.unit
    def test_spatial_query_creation_with_bbox(self):
        """Test creating SpatialQuery with bounding box."""
        bbox_min = (0.0, 0.0, 0.0)
        bbox_max = (10.0, 10.0, 10.0)
        query = SpatialQuery(bbox_min=bbox_min, bbox_max=bbox_max)

        assert query.bbox_min == bbox_min
        assert query.bbox_max == bbox_max
        assert query.is_empty() is False

    @pytest.mark.unit
    def test_spatial_query_creation_with_component_id(self):
        """Test creating SpatialQuery with component ID."""
        query = SpatialQuery(component_id="component_123")

        assert query.component_id == "component_123"
        assert query.is_empty() is False

    @pytest.mark.unit
    def test_spatial_query_creation_with_layer_range(self):
        """Test creating SpatialQuery with layer range."""
        query = SpatialQuery(layer_range=(0, 10))

        assert query.layer_range == (0, 10)
        assert query.is_empty() is False

    @pytest.mark.unit
    def test_spatial_query_creation_with_all_fields(self):
        """Test creating SpatialQuery with all fields."""
        query = SpatialQuery(
            bbox_min=(0.0, 0.0, 0.0),
            bbox_max=(10.0, 10.0, 10.0),
            component_id="component_123",
            layer_range=(0, 10),
        )

        assert query.bbox_min == (0.0, 0.0, 0.0)
        assert query.bbox_max == (10.0, 10.0, 10.0)
        assert query.component_id == "component_123"
        assert query.layer_range == (0, 10)
        assert query.is_empty() is False


class TestTemporalQuery:
    """Test suite for TemporalQuery dataclass."""

    @pytest.mark.unit
    def test_temporal_query_creation_empty(self):
        """Test creating empty TemporalQuery."""
        query = TemporalQuery()

        assert query.time_start is None
        assert query.time_end is None
        assert query.layer_start is None
        assert query.layer_end is None
        assert query.is_empty() is True

    @pytest.mark.unit
    def test_temporal_query_creation_with_time_range(self):
        """Test creating TemporalQuery with time range."""
        query = TemporalQuery(time_start=0.0, time_end=100.0)

        assert query.time_start == 0.0
        assert query.time_end == 100.0
        assert query.is_empty() is False

    @pytest.mark.unit
    def test_temporal_query_creation_with_layer_range(self):
        """Test creating TemporalQuery with layer range."""
        query = TemporalQuery(layer_start=0, layer_end=10)

        assert query.layer_start == 0
        assert query.layer_end == 10
        assert query.is_empty() is False

    @pytest.mark.unit
    def test_temporal_query_creation_with_all_fields(self):
        """Test creating TemporalQuery with all fields."""
        query = TemporalQuery(time_start=0.0, time_end=100.0, layer_start=0, layer_end=10)

        assert query.time_start == 0.0
        assert query.time_end == 100.0
        assert query.layer_start == 0
        assert query.layer_end == 10
        assert query.is_empty() is False


class TestQueryResult:
    """Test suite for QueryResult dataclass."""

    @pytest.mark.unit
    def test_query_result_creation_valid(self):
        """Test creating QueryResult with valid data."""
        points = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]
        signals = {"power": [200.0, 250.0], "speed": [100.0, 150.0]}
        metadata = {"layer": 1}

        result = QueryResult(points=points, signals=signals, metadata=metadata)

        assert result.points == points
        assert result.signals == signals
        assert result.metadata == metadata
        assert result.component_id is None

    @pytest.mark.unit
    def test_query_result_creation_with_component_id(self):
        """Test creating QueryResult with component ID."""
        points = [(0.0, 0.0, 0.0)]
        signals = {"power": [200.0]}
        metadata = {}

        result = QueryResult(
            points=points,
            signals=signals,
            metadata=metadata,
            component_id="component_123",
        )

        assert result.component_id == "component_123"

    @pytest.mark.unit
    def test_query_result_validation_matching_lengths(self):
        """Test QueryResult validation with matching signal lengths."""
        points = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]
        signals = {"power": [200.0, 250.0], "speed": [100.0, 150.0]}

        # Should not raise
        result = QueryResult(points=points, signals=signals, metadata={})
        assert result is not None

    @pytest.mark.unit
    def test_query_result_validation_mismatched_lengths(self):
        """Test QueryResult validation fails with mismatched signal lengths."""
        points = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]
        signals = {"power": [200.0, 250.0], "speed": [100.0]}  # Mismatched length

        with pytest.raises(ValueError, match="Signal 'speed' has 1 values"):
            QueryResult(points=points, signals=signals, metadata={})

    @pytest.mark.unit
    def test_query_result_validation_empty_points(self):
        """Test QueryResult validation with empty points (should not validate)."""
        points = []
        signals = {}
        metadata = {}

        # Should not raise (validation only runs if len(points) > 0)
        result = QueryResult(points=points, signals=signals, metadata={})
        assert result is not None


class TestBaseQueryClient:
    """Test suite for BaseQueryClient abstract class."""

    @pytest.mark.unit
    def test_base_query_client_creation(self):
        """Test creating BaseQueryClient instance (should fail - abstract)."""
        with pytest.raises(TypeError):
            BaseQueryClient()

    @pytest.mark.unit
    def test_base_query_client_abstract_methods(self):
        """Test that BaseQueryClient has abstract methods."""
        # Check that query method is abstract
        assert hasattr(BaseQueryClient, "query")
        assert getattr(BaseQueryClient.query, "__isabstractmethod__", False)

    @pytest.mark.unit
    def test_concrete_query_client_implementation(self):
        """Test that a concrete implementation can be created."""

        class ConcreteQueryClient(BaseQueryClient):
            def query(
                self,
                spatial: Optional[SpatialQuery] = None,
                temporal: Optional[TemporalQuery] = None,
                signal_types: Optional[List[SignalType]] = None,
            ) -> QueryResult:
                return QueryResult(points=[], signals={}, metadata={})

            def get_available_signals(self) -> List[SignalType]:
                return []

            def get_bounding_box(self, component_id: Optional[str] = None):
                return ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))

        client = ConcreteQueryClient(data_source="test_source")
        assert client.data_source == "test_source"
        assert isinstance(client, BaseQueryClient)

        # Test query method
        result = client.query()
        assert isinstance(result, QueryResult)

    @pytest.mark.unit
    def test_base_query_client_data_source(self):
        """Test BaseQueryClient data_source attribute."""

        class ConcreteQueryClient(BaseQueryClient):
            def query(self, spatial=None, temporal=None, signal_types=None):
                return QueryResult(points=[], signals={}, metadata={})

            def get_available_signals(self) -> List[SignalType]:
                return []

            def get_bounding_box(self, component_id: Optional[str] = None):
                return ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))

        client = ConcreteQueryClient(data_source="mongodb://localhost")
        assert client.data_source == "mongodb://localhost"

        client2 = ConcreteQueryClient()  # No data source
        assert client2.data_source is None

    @pytest.mark.unit
    def test_base_query_client_available_signals(self):
        """Test BaseQueryClient available_signals attribute."""

        class ConcreteQueryClient(BaseQueryClient):
            def query(self, spatial=None, temporal=None, signal_types=None):
                return QueryResult(points=[], signals={}, metadata={})

            def get_available_signals(self) -> List[SignalType]:
                return self._available_signals

            def get_bounding_box(self, component_id: Optional[str] = None):
                return ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))

        client = ConcreteQueryClient()
        assert client._available_signals == []

        # Test setting available signals
        client._available_signals = [SignalType.POWER, SignalType.VELOCITY]
        assert len(client._available_signals) == 2
        assert SignalType.POWER in client._available_signals
