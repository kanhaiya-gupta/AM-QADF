"""
Unit tests for temporal alignment.

Tests for TimePoint, LayerTimeMapper, and TemporalAligner.
"""

import pytest
import numpy as np
from am_qadf.synchronization.temporal_alignment import (
    TimePoint,
    LayerTimeMapper,
    TemporalAligner,
)


class TestTimePoint:
    """Test suite for TimePoint dataclass."""

    @pytest.mark.unit
    def test_time_point_creation(self):
        """Test creating TimePoint."""
        time_point = TimePoint(timestamp=100.0, layer_index=5, z_height=0.2, data={"power": 200.0})

        assert time_point.timestamp == 100.0
        assert time_point.layer_index == 5
        assert time_point.z_height == 0.2
        assert time_point.data["power"] == 200.0

    @pytest.mark.unit
    def test_time_point_minimal(self):
        """Test creating TimePoint with minimal parameters."""
        time_point = TimePoint(timestamp=50.0)

        assert time_point.timestamp == 50.0
        assert time_point.layer_index is None
        assert time_point.z_height is None
        assert time_point.data is None


class TestLayerTimeMapper:
    """Test suite for LayerTimeMapper class."""

    @pytest.fixture
    def mapper(self):
        """Create a LayerTimeMapper instance."""
        return LayerTimeMapper(layer_thickness=0.04, base_z=0.0, time_per_layer=10.0)

    @pytest.mark.unit
    def test_layer_time_mapper_creation_default(self):
        """Test creating LayerTimeMapper with default parameters."""
        mapper = LayerTimeMapper()

        assert mapper.layer_thickness == 0.04
        assert mapper.base_z == 0.0
        assert mapper.time_per_layer is None

    @pytest.mark.unit
    def test_layer_time_mapper_creation_custom(self):
        """Test creating LayerTimeMapper with custom parameters."""
        mapper = LayerTimeMapper(layer_thickness=0.05, base_z=1.0, time_per_layer=15.0)

        assert mapper.layer_thickness == 0.05
        assert mapper.base_z == 1.0
        assert mapper.time_per_layer == 15.0

    @pytest.mark.unit
    def test_add_layer_time(self, mapper):
        """Test adding layer-time mapping."""
        mapper.add_layer_time(layer_index=0, timestamp=0.0, z_height=0.0)
        mapper.add_layer_time(layer_index=10, timestamp=100.0, z_height=0.4)

        assert 0 in mapper._layer_times
        assert 10 in mapper._layer_times
        assert mapper._layer_times[0] == 0.0
        assert mapper._layer_times[10] == 100.0

    @pytest.mark.unit
    def test_add_layer_time_auto_z_height(self, mapper):
        """Test adding layer-time mapping with auto-computed z_height."""
        mapper.add_layer_time(layer_index=5, timestamp=50.0)

        assert 5 in mapper._z_heights
        expected_z = mapper.base_z + 5 * mapper.layer_thickness
        assert mapper._z_heights[5] == expected_z

    @pytest.mark.unit
    def test_layer_to_z(self, mapper):
        """Test converting layer index to Z height."""
        # With known mapping
        mapper.add_layer_time(layer_index=5, timestamp=50.0, z_height=0.25)
        z = mapper.layer_to_z(5)
        assert z == 0.25

        # Without known mapping (computed)
        z = mapper.layer_to_z(10)
        expected_z = mapper.base_z + 10 * mapper.layer_thickness
        assert z == expected_z

    @pytest.mark.unit
    def test_z_to_layer(self, mapper):
        """Test converting Z height to layer index."""
        layer = mapper.z_to_layer(0.2)

        expected_layer = int((0.2 - mapper.base_z) / mapper.layer_thickness)
        assert layer == expected_layer

    @pytest.mark.unit
    def test_z_to_layer_negative(self, mapper):
        """Test converting Z height below base_z."""
        layer = mapper.z_to_layer(-0.1)

        assert layer == 0  # Should clamp to 0

    @pytest.mark.unit
    def test_layer_to_time_known(self, mapper):
        """Test converting layer index to timestamp with known mapping."""
        mapper.add_layer_time(layer_index=5, timestamp=50.0)

        time = mapper.layer_to_time(5)
        assert time == 50.0

    @pytest.mark.unit
    def test_layer_to_time_interpolated(self, mapper):
        """Test converting layer index to timestamp with interpolation."""
        mapper.add_layer_time(layer_index=0, timestamp=0.0)
        mapper.add_layer_time(layer_index=10, timestamp=100.0)

        time = mapper.layer_to_time(5)

        # Should interpolate between 0 and 100
        assert 0.0 < time < 100.0

    @pytest.mark.unit
    def test_layer_to_time_extrapolated(self, mapper):
        """Test converting layer index to timestamp with extrapolation."""
        mapper.add_layer_time(layer_index=10, timestamp=100.0)

        # Before first layer
        time_before = mapper.layer_to_time(5)
        assert time_before is not None

        # After last layer
        time_after = mapper.layer_to_time(15)
        assert time_after is not None

    @pytest.mark.unit
    def test_layer_to_time_time_per_layer(self, mapper):
        """Test converting layer index to timestamp using time_per_layer."""
        mapper.add_layer_time(layer_index=0, timestamp=0.0)

        time = mapper.layer_to_time(5)

        # Should use time_per_layer
        assert time is not None
        assert time == 5 * mapper.time_per_layer

    @pytest.mark.unit
    def test_time_to_layer_known(self, mapper):
        """Test converting timestamp to layer index with known mapping."""
        mapper.add_layer_time(layer_index=5, timestamp=50.0)

        layer = mapper.time_to_layer(50.0)
        assert layer == 5

    @pytest.mark.unit
    def test_time_to_layer_interpolated(self, mapper):
        """Test converting timestamp to layer index with interpolation."""
        mapper.add_layer_time(layer_index=0, timestamp=0.0)
        mapper.add_layer_time(layer_index=10, timestamp=100.0)

        layer = mapper.time_to_layer(50.0)

        # Should interpolate between 0 and 10
        assert 0 <= layer <= 10

    @pytest.mark.unit
    def test_time_to_layer_extrapolated(self, mapper):
        """Test converting timestamp to layer index with extrapolation."""
        mapper.add_layer_time(layer_index=10, timestamp=100.0)

        # Before first layer
        layer_before = mapper.time_to_layer(50.0)
        assert layer_before is not None

        # After last layer
        layer_after = mapper.time_to_layer(150.0)
        assert layer_after is not None

    @pytest.mark.unit
    def test_time_to_layer_empty(self):
        """Test converting timestamp to layer index with no mappings."""
        mapper = LayerTimeMapper()

        layer = mapper.time_to_layer(50.0)
        assert layer is None


class TestTemporalAligner:
    """Test suite for TemporalAligner class."""

    @pytest.fixture
    def aligner(self):
        """Create a TemporalAligner instance."""
        mapper = LayerTimeMapper(layer_thickness=0.04, time_per_layer=10.0)
        mapper.add_layer_time(layer_index=0, timestamp=0.0)
        mapper.add_layer_time(layer_index=10, timestamp=100.0)
        return TemporalAligner(layer_mapper=mapper)

    @pytest.mark.unit
    def test_temporal_aligner_creation_default(self):
        """Test creating TemporalAligner with default mapper."""
        aligner = TemporalAligner()

        assert aligner.layer_mapper is not None
        assert len(aligner._time_points) == 0

    @pytest.mark.unit
    def test_temporal_aligner_creation_custom(self):
        """Test creating TemporalAligner with custom mapper."""
        mapper = LayerTimeMapper()
        aligner = TemporalAligner(layer_mapper=mapper)

        assert aligner.layer_mapper is mapper

    @pytest.mark.unit
    def test_add_time_point(self, aligner):
        """Test adding time point."""
        aligner.add_time_point(timestamp=50.0, layer_index=5, z_height=0.2, data={"power": 200.0})

        assert len(aligner._time_points) == 1
        assert aligner._time_points[0].timestamp == 50.0
        assert aligner._time_points[0].layer_index == 5

    @pytest.mark.unit
    def test_add_time_point_auto_layer(self, aligner):
        """Test adding time point with auto-computed layer index."""
        aligner.add_time_point(timestamp=50.0, data={"power": 200.0})

        assert len(aligner._time_points) == 1
        # Layer index should be computed from timestamp
        assert aligner._time_points[0].layer_index is not None

    @pytest.mark.unit
    def test_add_time_point_auto_z_height(self, aligner):
        """Test adding time point with auto-computed z_height."""
        aligner.add_time_point(timestamp=50.0, layer_index=5)

        assert len(aligner._time_points) == 1
        # Z height should be computed from layer index
        assert aligner._time_points[0].z_height is not None

    @pytest.mark.unit
    def test_align_to_layers(self, aligner):
        """Test aligning temporal data to specific layers."""
        aligner.add_time_point(timestamp=50.0, data={"power": 200.0})
        aligner.add_time_point(timestamp=100.0, data={"power": 250.0})

        target_layers = [0, 5, 10]
        aligned_data = aligner.align_to_layers(target_layers)

        assert len(aligned_data) == 3
        assert all(isinstance(d, dict) for d in aligned_data)

    @pytest.mark.unit
    def test_align_to_layers_empty(self):
        """Test aligning with no time points."""
        aligner = TemporalAligner()

        aligned_data = aligner.align_to_layers([0, 5, 10])

        assert len(aligned_data) == 3
        assert all(d == {} for d in aligned_data)

    @pytest.mark.unit
    def test_align_to_layers_nearest(self, aligner):
        """Test aligning with nearest interpolation method."""
        aligner.add_time_point(timestamp=50.0, data={"power": 200.0})
        aligner.add_time_point(timestamp=100.0, data={"power": 250.0})

        aligned_data = aligner.align_to_layers([5], interpolation_method="nearest")

        assert len(aligned_data) == 1

    @pytest.mark.unit
    def test_get_layer_data(self, aligner):
        """Test getting data for specific layer."""
        aligner.add_time_point(timestamp=50.0, layer_index=5, data={"power": 200.0})

        data = aligner.get_layer_data(5)

        assert data is not None
        assert data["power"] == 200.0

    @pytest.mark.unit
    def test_get_layer_data_nonexistent(self, aligner):
        """Test getting data for nonexistent layer."""
        data = aligner.get_layer_data(100)

        # Should return None or closest match
        assert data is None or isinstance(data, dict)

    @pytest.mark.unit
    def test_handle_missing_temporal_data(self, aligner):
        """Test handling missing temporal data."""
        aligner.add_time_point(timestamp=50.0, layer_index=5, data={"power": 200.0})

        required_layers = [0, 5, 10]
        result = aligner.handle_missing_temporal_data(required_layers)

        assert len(result) == 3
        assert 0 in result
        assert 5 in result
        assert 10 in result

    @pytest.mark.unit
    def test_handle_missing_temporal_data_with_default(self, aligner):
        """Test handling missing temporal data with default."""
        default_data = {"power": 0.0}
        required_layers = [0, 5, 10]

        result = aligner.handle_missing_temporal_data(required_layers, default_data=default_data)

        assert len(result) == 3
        # Missing layers should use default
        for layer_idx, data in result.items():
            assert isinstance(data, dict)
