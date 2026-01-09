"""
Unit tests for core value objects.

Tests for VoxelCoordinates and QualityMetric.
"""

import pytest
from datetime import datetime
from am_qadf.core.value_objects import VoxelCoordinates, QualityMetric
from am_qadf.core.exceptions import ValidationError


class TestVoxelCoordinates:
    """Test suite for VoxelCoordinates value object."""

    @pytest.mark.unit
    def test_voxel_coordinates_creation_basic(self):
        """Test creating VoxelCoordinates with basic coordinates."""
        coords = VoxelCoordinates(x=1.0, y=2.0, z=3.0)

        assert coords.x == 1.0
        assert coords.y == 2.0
        assert coords.z == 3.0
        assert coords.voxel_size == 0.1  # Default
        assert coords.voxel_volume == 0.1**3  # Auto-calculated

    @pytest.mark.unit
    def test_voxel_coordinates_immutable(self):
        """Test that VoxelCoordinates is immutable (frozen dataclass)."""
        coords = VoxelCoordinates(x=1.0, y=2.0, z=3.0)

        with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
            coords.x = 5.0

    @pytest.mark.unit
    def test_voxel_coordinates_volume_calculation(self):
        """Test automatic voxel volume calculation."""
        coords = VoxelCoordinates(x=0.0, y=0.0, z=0.0, voxel_size=0.5)

        expected_volume = 0.5**3
        assert coords.voxel_volume == expected_volume

    @pytest.mark.unit
    def test_voxel_coordinates_custom_volume(self):
        """Test setting custom voxel volume."""
        custom_volume = 0.125
        coords = VoxelCoordinates(x=0.0, y=0.0, z=0.0, voxel_size=0.5, voxel_volume=custom_volume)

        assert coords.voxel_volume == custom_volume

    @pytest.mark.unit
    def test_voxel_coordinates_get_coordinates(self):
        """Test get_coordinates method."""
        coords = VoxelCoordinates(x=1.0, y=2.0, z=3.0)

        result = coords.get_coordinates()

        assert result == (1.0, 2.0, 3.0)
        assert isinstance(result, tuple)

    @pytest.mark.unit
    def test_voxel_coordinates_get_rotations(self):
        """Test get_rotations method."""
        coords = VoxelCoordinates(x=0.0, y=0.0, z=0.0, rotation_x=10.0, rotation_y=20.0, rotation_z=30.0)

        result = coords.get_rotations()

        assert result == (10.0, 20.0, 30.0)
        assert isinstance(result, tuple)

    @pytest.mark.unit
    def test_voxel_coordinates_distance_to(self):
        """Test distance_to method."""
        coords1 = VoxelCoordinates(x=0.0, y=0.0, z=0.0)
        coords2 = VoxelCoordinates(x=3.0, y=4.0, z=0.0)

        distance = coords1.distance_to(coords2)

        # Distance should be 5.0 (3-4-5 right triangle)
        assert abs(distance - 5.0) < 1e-6

    @pytest.mark.unit
    def test_voxel_coordinates_validation_valid(self):
        """Test validation with valid coordinates."""
        # Should not raise
        coords = VoxelCoordinates(x=100.0, y=200.0, z=300.0)
        assert coords is not None

    @pytest.mark.unit
    def test_voxel_coordinates_validation_negative_size(self):
        """Test validation fails with negative voxel size."""
        with pytest.raises(ValueError, match="Voxel size must be positive"):
            VoxelCoordinates(x=0.0, y=0.0, z=0.0, voxel_size=-0.1)

    @pytest.mark.unit
    def test_voxel_coordinates_validation_zero_size(self):
        """Test validation fails with zero voxel size."""
        with pytest.raises(ValueError, match="Voxel size must be positive"):
            VoxelCoordinates(x=0.0, y=0.0, z=0.0, voxel_size=0.0)

    @pytest.mark.unit
    def test_voxel_coordinates_validation_rotation_out_of_range(self):
        """Test validation fails with rotation out of range."""
        with pytest.raises(ValueError, match="Rotation angles must be between"):
            VoxelCoordinates(x=0.0, y=0.0, z=0.0, rotation_x=200.0)

    @pytest.mark.unit
    def test_voxel_coordinates_validation_negative_density(self):
        """Test validation fails with negative material density."""
        with pytest.raises(ValueError, match="Material density must be positive"):
            VoxelCoordinates(x=0.0, y=0.0, z=0.0, material_density=-1.0)

    @pytest.mark.unit
    def test_voxel_coordinates_validation_quality_score_out_of_range(self):
        """Test validation fails with quality score out of range."""
        with pytest.raises(ValueError, match="Quality score must be between"):
            VoxelCoordinates(x=0.0, y=0.0, z=0.0, quality_score=150.0)

    @pytest.mark.unit
    def test_voxel_coordinates_validation_negative_temperature(self):
        """Test validation fails with negative temperature."""
        with pytest.raises(ValueError, match="Temperature cannot be negative"):
            VoxelCoordinates(x=0.0, y=0.0, z=0.0, temperature_peak=-10.0)

    @pytest.mark.unit
    def test_voxel_coordinates_validation_negative_cooling_rate(self):
        """Test validation fails with negative cooling rate."""
        with pytest.raises(ValueError, match="Cooling rate cannot be negative"):
            VoxelCoordinates(x=0.0, y=0.0, z=0.0, cooling_rate=-1.0)

    @pytest.mark.unit
    def test_voxel_coordinates_with_all_properties(self):
        """Test creating VoxelCoordinates with all optional properties."""
        timestamp = datetime.now()
        coords = VoxelCoordinates(
            x=10.0,
            y=20.0,
            z=30.0,
            voxel_size=0.5,
            rotation_x=15.0,
            rotation_y=30.0,
            rotation_z=45.0,
            is_solid=True,
            is_processed=True,
            is_defective=False,
            material_density=8.0,
            material_type="Ti6Al4V",
            layer_number=5,
            scan_vector_id="vec_123",
            processing_timestamp=timestamp,
            quality_score=95.0,
            temperature_peak=1200.0,
            cooling_rate=50.0,
        )

        assert coords.x == 10.0
        assert coords.y == 20.0
        assert coords.z == 30.0
        assert coords.is_solid is True
        assert coords.material_type == "Ti6Al4V"
        assert coords.layer_number == 5
        assert coords.quality_score == 95.0


class TestQualityMetric:
    """Test suite for QualityMetric value object."""

    @pytest.mark.unit
    def test_quality_metric_creation_basic(self):
        """Test creating QualityMetric with basic values."""
        metric = QualityMetric(value=0.95, metric_name="completeness")

        assert metric.value == 0.95
        assert metric.metric_name == "completeness"
        assert metric.unit is None
        assert metric.timestamp is None

    @pytest.mark.unit
    def test_quality_metric_creation_with_all_fields(self):
        """Test creating QualityMetric with all fields."""
        timestamp = datetime.now()
        metric = QualityMetric(value=0.95, metric_name="signal_quality", unit="SNR", timestamp=timestamp)

        assert metric.value == 0.95
        assert metric.metric_name == "signal_quality"
        assert metric.unit == "SNR"
        assert metric.timestamp == timestamp

    @pytest.mark.unit
    def test_quality_metric_immutable(self):
        """Test that QualityMetric is immutable (frozen dataclass)."""
        metric = QualityMetric(value=0.95, metric_name="test")

        with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
            metric.value = 0.99

    @pytest.mark.unit
    def test_quality_metric_validation_numeric_value(self):
        """Test validation accepts numeric values."""
        # Should not raise
        metric1 = QualityMetric(value=0.95, metric_name="test")
        metric2 = QualityMetric(value=95, metric_name="test")  # int
        metric3 = QualityMetric(value=95.5, metric_name="test")  # float

        assert metric1.value == 0.95
        assert metric2.value == 95
        assert metric3.value == 95.5

    @pytest.mark.unit
    def test_quality_metric_validation_non_numeric_value(self):
        """Test validation fails with non-numeric value."""
        with pytest.raises(ValueError, match="Quality metric value must be numeric"):
            QualityMetric(value="invalid", metric_name="test")

    @pytest.mark.unit
    def test_quality_metric_validation_empty_name(self):
        """Test validation fails with empty metric name."""
        with pytest.raises(ValueError, match="Quality metric name cannot be empty"):
            QualityMetric(value=0.95, metric_name="")

    @pytest.mark.unit
    def test_quality_metric_validation_whitespace_name(self):
        """Test validation fails with whitespace-only name."""
        with pytest.raises(ValueError, match="Quality metric name cannot be empty"):
            QualityMetric(value=0.95, metric_name="   ")
