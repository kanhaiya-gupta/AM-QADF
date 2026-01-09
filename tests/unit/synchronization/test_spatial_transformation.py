"""
Unit tests for spatial transformation.

Tests for TransformationMatrix, SpatialTransformer, and TransformationManager.
"""

import pytest
import numpy as np
from am_qadf.synchronization.spatial_transformation import (
    TransformationMatrix,
    SpatialTransformer,
    TransformationManager,
)


class TestTransformationMatrix:
    """Test suite for TransformationMatrix class."""

    @pytest.mark.unit
    def test_transformation_matrix_creation(self):
        """Test creating TransformationMatrix."""
        matrix = np.eye(4)
        trans = TransformationMatrix(matrix=matrix)

        assert np.array_equal(trans.matrix, matrix)

    @pytest.mark.unit
    def test_transformation_matrix_invalid_shape(self):
        """Test creating TransformationMatrix with invalid shape."""
        matrix = np.eye(3)  # Wrong shape

        with pytest.raises(ValueError, match="Transformation matrix must be 4x4"):
            TransformationMatrix(matrix=matrix)

    @pytest.mark.unit
    def test_apply_identity(self):
        """Test applying identity transformation."""
        trans = TransformationMatrix.identity()
        points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        result = trans.apply(points)

        assert np.allclose(result, points)

    @pytest.mark.unit
    def test_apply_single_point(self):
        """Test applying transformation to single point."""
        trans = TransformationMatrix.identity()
        point = np.array([1.0, 2.0, 3.0])

        result = trans.apply(point)

        assert result.shape == (1, 3)
        assert np.allclose(result[0], point)

    @pytest.mark.unit
    def test_apply_translation(self):
        """Test applying translation transformation."""
        trans = TransformationMatrix.translation(10.0, 20.0, 30.0)
        points = np.array([[1.0, 2.0, 3.0]])

        result = trans.apply(points)

        expected = np.array([[11.0, 22.0, 33.0]])
        assert np.allclose(result, expected)

    @pytest.mark.unit
    def test_apply_rotation(self):
        """Test applying rotation transformation."""
        # 90 degree rotation around z-axis
        trans = TransformationMatrix.rotation("z", np.pi / 2)
        points = np.array([[1.0, 0.0, 0.0]])

        result = trans.apply(points)

        # Should rotate to (0, 1, 0)
        assert np.allclose(result[0, 0], 0.0, atol=1e-6)
        assert np.allclose(result[0, 1], 1.0, atol=1e-6)
        assert np.allclose(result[0, 2], 0.0, atol=1e-6)

    @pytest.mark.unit
    def test_apply_scale(self):
        """Test applying scaling transformation."""
        trans = TransformationMatrix.scale(2.0, 3.0, 4.0)
        points = np.array([[1.0, 2.0, 3.0]])

        result = trans.apply(points)

        expected = np.array([[2.0, 6.0, 12.0]])
        assert np.allclose(result, expected)

    @pytest.mark.unit
    def test_inverse(self):
        """Test computing inverse transformation."""
        trans = TransformationMatrix.translation(10.0, 20.0, 30.0)
        inv_trans = trans.inverse()

        points = np.array([[1.0, 2.0, 3.0]])
        transformed = trans.apply(points)
        back_transformed = inv_trans.apply(transformed)

        assert np.allclose(back_transformed, points, atol=1e-6)

    @pytest.mark.unit
    def test_identity_classmethod(self):
        """Test creating identity transformation."""
        trans = TransformationMatrix.identity()

        assert np.allclose(trans.matrix, np.eye(4))

    @pytest.mark.unit
    def test_translation_classmethod(self):
        """Test creating translation transformation."""
        trans = TransformationMatrix.translation(10.0, 20.0, 30.0)

        assert trans.matrix[0, 3] == 10.0
        assert trans.matrix[1, 3] == 20.0
        assert trans.matrix[2, 3] == 30.0

    @pytest.mark.unit
    def test_rotation_classmethod(self):
        """Test creating rotation transformation."""
        # Test rotation around x-axis
        trans_x = TransformationMatrix.rotation("x", np.pi / 2)
        assert trans_x.matrix.shape == (4, 4)

        # Test rotation around y-axis
        trans_y = TransformationMatrix.rotation("y", np.pi / 2)
        assert trans_y.matrix.shape == (4, 4)

        # Test rotation around z-axis
        trans_z = TransformationMatrix.rotation("z", np.pi / 2)
        assert trans_z.matrix.shape == (4, 4)

    @pytest.mark.unit
    def test_scale_classmethod(self):
        """Test creating scaling transformation."""
        trans = TransformationMatrix.scale(2.0, 3.0, 4.0)

        assert trans.matrix[0, 0] == 2.0
        assert trans.matrix[1, 1] == 3.0
        assert trans.matrix[2, 2] == 4.0


class TestSpatialTransformer:
    """Test suite for SpatialTransformer class."""

    @pytest.fixture
    def transformer(self):
        """Create a SpatialTransformer instance."""
        return SpatialTransformer()

    @pytest.mark.unit
    def test_spatial_transformer_creation(self, transformer):
        """Test creating SpatialTransformer."""
        assert transformer is not None
        assert len(transformer._transformations) == 0

    @pytest.mark.unit
    def test_register_transformation(self, transformer):
        """Test registering a transformation."""
        trans = TransformationMatrix.identity()
        transformer.register_transformation("identity", trans)

        assert "identity" in transformer._transformations

    @pytest.mark.unit
    def test_get_transformation(self, transformer):
        """Test getting a registered transformation."""
        trans = TransformationMatrix.translation(10.0, 20.0, 30.0)
        transformer.register_transformation("translate", trans)

        retrieved = transformer.get_transformation("translate")

        assert retrieved is trans

    @pytest.mark.unit
    def test_get_transformation_nonexistent(self, transformer):
        """Test getting nonexistent transformation."""
        retrieved = transformer.get_transformation("nonexistent")

        assert retrieved is None

    @pytest.mark.unit
    def test_transform_points_with_name(self, transformer):
        """Test transforming points using transformation name."""
        trans = TransformationMatrix.translation(10.0, 20.0, 30.0)
        transformer.register_transformation("translate", trans)

        points = np.array([[1.0, 2.0, 3.0]])
        result = transformer.transform_points(points, transformation_name="translate")

        expected = np.array([[11.0, 22.0, 33.0]])
        assert np.allclose(result, expected)

    @pytest.mark.unit
    def test_transform_points_with_transformation(self, transformer):
        """Test transforming points using direct transformation."""
        trans = TransformationMatrix.translation(10.0, 20.0, 30.0)
        points = np.array([[1.0, 2.0, 3.0]])

        result = transformer.transform_points(points, transformation=trans)

        expected = np.array([[11.0, 22.0, 33.0]])
        assert np.allclose(result, expected)

    @pytest.mark.unit
    def test_transform_points_no_transformation(self, transformer):
        """Test transforming points without providing transformation."""
        points = np.array([[1.0, 2.0, 3.0]])

        with pytest.raises(ValueError, match="Must provide transformation_name or transformation"):
            transformer.transform_points(points)

    @pytest.mark.unit
    def test_align_coordinate_systems(self, transformer):
        """Test aligning coordinate systems."""
        # Create source and target point sets
        source_points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        # Target points are source points translated
        target_points = source_points + np.array([10.0, 20.0, 30.0])

        trans = transformer.align_coordinate_systems(source_points, target_points)

        # Apply transformation and verify alignment
        transformed = trans.apply(source_points)
        assert np.allclose(transformed, target_points, atol=1e-5)

    @pytest.mark.unit
    def test_align_coordinate_systems_shape_mismatch(self, transformer):
        """Test aligning coordinate systems with shape mismatch."""
        source_points = np.array([[1.0, 2.0, 3.0]])
        target_points = np.array([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

        with pytest.raises(ValueError, match="Source and target points must have same shape"):
            transformer.align_coordinate_systems(source_points, target_points)


class TestTransformationManager:
    """Test suite for TransformationManager class."""

    @pytest.fixture
    def manager(self):
        """Create a TransformationManager instance."""
        return TransformationManager()

    @pytest.mark.unit
    def test_transformation_manager_creation(self, manager):
        """Test creating TransformationManager."""
        assert manager is not None
        assert manager.transformer is not None
        assert len(manager._coordinate_systems) == 0

    @pytest.mark.unit
    def test_register_coordinate_system(self, manager):
        """Test registering a coordinate system."""
        manager.register_coordinate_system("system1", origin=(10.0, 20.0, 30.0))

        assert "system1" in manager._coordinate_systems
        assert manager._coordinate_systems["system1"]["origin"] == (10.0, 20.0, 30.0)

    @pytest.mark.unit
    def test_register_coordinate_system_default(self, manager):
        """Test registering coordinate system with defaults."""
        manager.register_coordinate_system("system1")

        assert "system1" in manager._coordinate_systems
        assert manager._coordinate_systems["system1"]["origin"] == (0.0, 0.0, 0.0)

    @pytest.mark.unit
    def test_set_transformation(self, manager):
        """Test setting transformation between systems."""
        trans = TransformationMatrix.translation(10.0, 20.0, 30.0)
        manager.set_transformation("system1", "system2", trans)

        # Should register both directions
        assert manager.transformer.get_transformation("system1_to_system2") is not None
        assert manager.transformer.get_transformation("system2_to_system1") is not None

    @pytest.mark.unit
    def test_get_transformation_identity(self, manager):
        """Test getting transformation for same system."""
        trans = manager.get_transformation("system1", "system1")

        assert trans is not None
        assert np.allclose(trans.matrix, np.eye(4))

    @pytest.mark.unit
    def test_get_transformation_direct(self, manager):
        """Test getting direct transformation."""
        trans = TransformationMatrix.translation(10.0, 20.0, 30.0)
        manager.set_transformation("system1", "system2", trans)

        retrieved = manager.get_transformation("system1", "system2")

        assert retrieved is not None
        assert np.allclose(retrieved.matrix, trans.matrix)

    @pytest.mark.unit
    def test_get_transformation_inverse(self, manager):
        """Test getting inverse transformation."""
        trans = TransformationMatrix.translation(10.0, 20.0, 30.0)
        manager.set_transformation("system1", "system2", trans)

        # Get inverse direction
        retrieved = manager.get_transformation("system2", "system1")

        assert retrieved is not None
        # Should be inverse
        expected_inv = trans.inverse()
        assert np.allclose(retrieved.matrix, expected_inv.matrix)

    @pytest.mark.unit
    def test_get_transformation_chained(self, manager):
        """Test getting chained transformation."""
        trans1 = TransformationMatrix.translation(10.0, 0.0, 0.0)
        trans2 = TransformationMatrix.translation(0.0, 20.0, 0.0)

        manager.set_transformation("system1", "system2", trans1)
        manager.set_transformation("system2", "system3", trans2)

        # Should chain through system2
        chained = manager.get_transformation("system1", "system3")

        assert chained is not None
        # Combined translation should be (10, 20, 0)
        test_point = np.array([[0.0, 0.0, 0.0]])
        result = chained.apply(test_point)
        assert np.allclose(result[0], [10.0, 20.0, 0.0])

    @pytest.mark.unit
    def test_get_transformation_nonexistent(self, manager):
        """Test getting transformation for nonexistent systems."""
        trans = manager.get_transformation("system1", "system2")

        assert trans is None

    @pytest.mark.unit
    def test_transform_points(self, manager):
        """Test transforming points between coordinate systems."""
        trans = TransformationMatrix.translation(10.0, 20.0, 30.0)
        manager.set_transformation("system1", "system2", trans)

        points = np.array([[1.0, 2.0, 3.0]])
        result = manager.transform_points(points, "system1", "system2")

        expected = np.array([[11.0, 22.0, 33.0]])
        assert np.allclose(result, expected)

    @pytest.mark.unit
    def test_transform_points_no_transformation(self, manager):
        """Test transforming points when no transformation exists."""
        points = np.array([[1.0, 2.0, 3.0]])

        with pytest.raises(ValueError, match="No transformation found"):
            manager.transform_points(points, "system1", "system2")
