"""
Unit tests for coordinate systems.

Tests for CoordinateSystemType, CoordinateSystem, and CoordinateSystemRegistry.
"""

import pytest
import numpy as np
from am_qadf.voxelization.coordinate_systems import (
    CoordinateSystemType,
    CoordinateSystem,
    CoordinateSystemRegistry,
)


class TestCoordinateSystemType:
    """Test suite for CoordinateSystemType enum."""

    @pytest.mark.unit
    def test_coordinate_system_type_values(self):
        """Test CoordinateSystemType enum values."""
        assert CoordinateSystemType.STL.value == "stl"
        assert CoordinateSystemType.BUILD_PLATFORM.value == "build_platform"
        assert CoordinateSystemType.GLOBAL.value == "global"
        assert CoordinateSystemType.COMPONENT_LOCAL.value == "component_local"

    @pytest.mark.unit
    def test_coordinate_system_type_enumeration(self):
        """Test that CoordinateSystemType can be enumerated."""
        types = list(CoordinateSystemType)
        assert len(types) == 4
        assert CoordinateSystemType.STL in types


class TestCoordinateSystem:
    """Test suite for CoordinateSystem class."""

    @pytest.mark.unit
    def test_coordinate_system_creation_basic(self):
        """Test creating CoordinateSystem with basic parameters."""
        system = CoordinateSystem(
            name="test_system",
            origin=(0.0, 0.0, 0.0),
            rotation=(0.0, 0.0, 0.0),
            scale=1.0,
        )

        assert system.name == "test_system"
        assert np.array_equal(system.origin, [0.0, 0.0, 0.0])
        assert np.array_equal(system.rotation, [0.0, 0.0, 0.0])
        assert system.scale == 1.0

    @pytest.mark.unit
    def test_coordinate_system_creation_with_translation(self):
        """Test creating CoordinateSystem with translation."""
        system = CoordinateSystem(name="translated_system", origin=(10.0, 20.0, 30.0))

        assert np.array_equal(system.origin, [10.0, 20.0, 30.0])

    @pytest.mark.unit
    def test_coordinate_system_creation_with_rotation(self):
        """Test creating CoordinateSystem with rotation."""
        system = CoordinateSystem(name="rotated_system", rotation=(90.0, 0.0, 0.0))  # 90 degrees around x-axis

        assert np.array_equal(system.rotation, [90.0, 0.0, 0.0])
        assert system.rotation_matrix is not None

    @pytest.mark.unit
    def test_coordinate_system_creation_with_scale(self):
        """Test creating CoordinateSystem with scale."""
        system = CoordinateSystem(name="scaled_system", scale=2.0)

        assert system.scale == 2.0

    @pytest.mark.unit
    def test_coordinate_system_transform_point_no_transformation(self):
        """Test transforming point with no transformation (identity)."""
        system = CoordinateSystem(name="identity", origin=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0), scale=1.0)

        point = np.array([1.0, 2.0, 3.0])
        transformed = system.transform_point(point)

        assert np.allclose(transformed, point)

    @pytest.mark.unit
    def test_coordinate_system_transform_point_translation_only(self):
        """Test transforming point with translation only."""
        system = CoordinateSystem(
            name="translated",
            origin=(10.0, 20.0, 30.0),
            rotation=(0.0, 0.0, 0.0),
            scale=1.0,
        )

        point = np.array([1.0, 2.0, 3.0])
        transformed = system.transform_point(point)

        expected = point + system.origin
        assert np.allclose(transformed, expected)

    @pytest.mark.unit
    def test_coordinate_system_transform_point_scale_only(self):
        """Test transforming point with scale only."""
        system = CoordinateSystem(name="scaled", origin=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0), scale=2.0)

        point = np.array([1.0, 2.0, 3.0])
        transformed = system.transform_point(point)

        expected = point * 2.0
        assert np.allclose(transformed, expected)

    @pytest.mark.unit
    def test_coordinate_system_inverse_transform_point(self):
        """Test inverse transformation."""
        system = CoordinateSystem(name="test", origin=(10.0, 20.0, 30.0), rotation=(0.0, 0.0, 0.0), scale=1.0)

        point = np.array([1.0, 2.0, 3.0])
        transformed = system.transform_point(point)
        back_transformed = system.inverse_transform_point(transformed)

        assert np.allclose(back_transformed, point, atol=1e-6)

    @pytest.mark.unit
    def test_coordinate_system_transform_point_invalid_shape(self):
        """Test transforming point with invalid shape."""
        system = CoordinateSystem(name="test")

        point = np.array([1.0, 2.0])  # Wrong shape

        with pytest.raises(ValueError, match="Point must be shape"):
            system.transform_point(point)

    @pytest.mark.unit
    def test_coordinate_system_get_bounding_box(self):
        """Test getting bounding box."""
        system = CoordinateSystem(name="test", origin=(10.0, 20.0, 30.0))

        bbox_min, bbox_max = system.get_bounding_box()

        assert np.array_equal(bbox_min, [10.0, 20.0, 30.0])
        assert np.array_equal(bbox_max, [10.0, 20.0, 30.0])


class TestCoordinateSystemRegistry:
    """Test suite for CoordinateSystemRegistry class."""

    @pytest.mark.unit
    def test_registry_creation(self):
        """Test creating CoordinateSystemRegistry."""
        registry = CoordinateSystemRegistry()

        assert len(registry.systems) == 0
        assert len(registry.parent_relationships) == 0

    @pytest.mark.unit
    def test_registry_register_system(self):
        """Test registering a coordinate system."""
        registry = CoordinateSystemRegistry()

        registry.register(
            name="test_system",
            origin=(0.0, 0.0, 0.0),
            rotation=(0.0, 0.0, 0.0),
            scale=1.0,
        )

        assert "test_system" in registry.systems
        assert isinstance(registry.systems["test_system"], CoordinateSystem)

    @pytest.mark.unit
    def test_registry_register_with_parent(self):
        """Test registering a coordinate system with parent."""
        registry = CoordinateSystemRegistry()

        registry.register(name="parent_system")
        registry.register(name="child_system", parent="parent_system")

        assert registry.parent_relationships["child_system"] == "parent_system"
        assert registry.parent_relationships["parent_system"] is None

    @pytest.mark.unit
    def test_registry_get_system(self):
        """Test getting a coordinate system."""
        registry = CoordinateSystemRegistry()

        registry.register(name="test_system")

        system = registry.get("test_system")

        assert system is not None
        assert system.name == "test_system"

    @pytest.mark.unit
    def test_registry_get_nonexistent_system(self):
        """Test getting nonexistent coordinate system."""
        registry = CoordinateSystemRegistry()

        system = registry.get("nonexistent")

        assert system is None

    @pytest.mark.unit
    def test_registry_transform_same_system(self):
        """Test transforming within the same system."""
        registry = CoordinateSystemRegistry()

        registry.register(name="system1")

        point = np.array([1.0, 2.0, 3.0])
        transformed = registry.transform(point, "system1", "system1")

        assert np.allclose(transformed, point)

    @pytest.mark.unit
    def test_registry_transform_direct(self):
        """Test transforming between two systems."""
        registry = CoordinateSystemRegistry()

        registry.register(name="system1", origin=(0.0, 0.0, 0.0))
        registry.register(name="system2", origin=(10.0, 20.0, 30.0))

        point = np.array([1.0, 2.0, 3.0])
        transformed = registry.transform(point, "system1", "system2")

        # Should transform through global (parent chain)
        expected = point + np.array([10.0, 20.0, 30.0])
        assert np.allclose(transformed, expected, atol=1e-6)

    @pytest.mark.unit
    def test_registry_transform_nonexistent_system(self):
        """Test transforming with nonexistent system."""
        registry = CoordinateSystemRegistry()

        point = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="Coordinate system 'nonexistent' not found"):
            registry.transform(point, "nonexistent", "system1")

    @pytest.mark.unit
    def test_registry_list_systems(self):
        """Test listing all coordinate systems."""
        registry = CoordinateSystemRegistry()

        registry.register(name="system1")
        registry.register(name="system2")
        registry.register(name="system3")

        systems = registry.list_systems()

        assert len(systems) == 3
        assert "system1" in systems
        assert "system2" in systems
        assert "system3" in systems
