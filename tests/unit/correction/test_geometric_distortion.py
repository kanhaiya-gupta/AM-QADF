"""
Unit tests for geometric distortion models.

Tests for DistortionModel, ScalingModel, RotationModel, WarpingModel, and CombinedDistortionModel.
"""

import pytest
import numpy as np
from am_qadf.correction.geometric_distortion import (
    DistortionModel,
    ScalingModel,
    RotationModel,
    WarpingModel,
    CombinedDistortionModel,
)


class TestDistortionModel:
    """Test suite for DistortionModel abstract base class."""

    @pytest.mark.unit
    def test_distortion_model_abstract(self):
        """Test that DistortionModel is abstract."""
        with pytest.raises(TypeError):
            DistortionModel()


class TestScalingModel:
    """Test suite for ScalingModel class."""

    @pytest.mark.unit
    def test_scaling_model_creation_default(self):
        """Test creating ScalingModel with default parameters."""
        model = ScalingModel()

        assert model.scale_x == 1.0
        assert model.scale_y == 1.0
        assert model.scale_z == 1.0
        assert model.center == (0.0, 0.0, 0.0)

    @pytest.mark.unit
    def test_scaling_model_creation_custom(self):
        """Test creating ScalingModel with custom parameters."""
        model = ScalingModel(scale_x=2.0, scale_y=3.0, scale_z=4.0, center=(10.0, 20.0, 30.0))

        assert model.scale_x == 2.0
        assert model.scale_y == 3.0
        assert model.scale_z == 4.0
        assert model.center == (10.0, 20.0, 30.0)

    @pytest.mark.unit
    def test_apply_scaling_identity(self):
        """Test applying identity scaling."""
        model = ScalingModel(scale_x=1.0, scale_y=1.0, scale_z=1.0)
        points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        result = model.apply(points)

        assert np.allclose(result, points)

    @pytest.mark.unit
    def test_apply_scaling_uniform(self):
        """Test applying uniform scaling."""
        model = ScalingModel(scale_x=2.0, scale_y=2.0, scale_z=2.0)
        points = np.array([[1.0, 2.0, 3.0]])

        result = model.apply(points)

        expected = np.array([[2.0, 4.0, 6.0]])
        assert np.allclose(result, expected)

    @pytest.mark.unit
    def test_apply_scaling_non_uniform(self):
        """Test applying non-uniform scaling."""
        model = ScalingModel(scale_x=2.0, scale_y=3.0, scale_z=4.0)
        points = np.array([[1.0, 2.0, 3.0]])

        result = model.apply(points)

        expected = np.array([[2.0, 6.0, 12.0]])
        assert np.allclose(result, expected)

    @pytest.mark.unit
    def test_apply_scaling_with_center(self):
        """Test applying scaling with custom center."""
        model = ScalingModel(scale_x=2.0, center=(10.0, 20.0, 30.0))
        points = np.array([[11.0, 21.0, 31.0]])  # 1 unit from center

        result = model.apply(points)

        # Should scale relative to center
        # Point is 1 unit from center in all directions
        # After 2x scaling in x only, should be 2 units from center in x, 1 unit in y and z
        expected = np.array([[12.0, 21.0, 31.0]])
        assert np.allclose(result, expected)

    @pytest.mark.unit
    def test_correct_scaling(self):
        """Test correcting scaling distortion."""
        model = ScalingModel(scale_x=2.0, scale_y=3.0, scale_z=4.0)
        points = np.array([[2.0, 6.0, 12.0]])  # Already scaled

        result = model.correct(points)

        expected = np.array([[1.0, 2.0, 3.0]])  # Should correct back
        assert np.allclose(result, expected)

    @pytest.mark.unit
    def test_correct_scaling_invertible(self):
        """Test that apply and correct are inverse operations."""
        model = ScalingModel(scale_x=2.0, scale_y=3.0, scale_z=4.0, center=(10.0, 20.0, 30.0))
        points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        distorted = model.apply(points)
        corrected = model.correct(distorted)

        assert np.allclose(corrected, points, atol=1e-6)

    @pytest.mark.unit
    def test_get_parameters(self):
        """Test getting scaling parameters."""
        model = ScalingModel(scale_x=2.0, scale_y=3.0, scale_z=4.0, center=(10.0, 20.0, 30.0))

        params = model.get_parameters()

        assert params["type"] == "scaling"
        assert params["scale_x"] == 2.0
        assert params["scale_y"] == 3.0
        assert params["scale_z"] == 4.0
        assert params["center"] == (10.0, 20.0, 30.0)


class TestRotationModel:
    """Test suite for RotationModel class."""

    @pytest.mark.unit
    def test_rotation_model_creation_default(self):
        """Test creating RotationModel with default parameters."""
        model = RotationModel()

        assert model.axis == "z"
        assert model.angle == 0.0
        assert model.center == (0.0, 0.0, 0.0)

    @pytest.mark.unit
    def test_rotation_model_creation_custom(self):
        """Test creating RotationModel with custom parameters."""
        model = RotationModel(axis="x", angle=np.pi / 2, center=(10.0, 20.0, 30.0))

        assert model.axis == "x"
        assert model.angle == np.pi / 2
        assert model.center == (10.0, 20.0, 30.0)

    @pytest.mark.unit
    def test_apply_rotation_identity(self):
        """Test applying identity rotation."""
        model = RotationModel(angle=0.0)
        points = np.array([[1.0, 2.0, 3.0]])

        result = model.apply(points)

        assert np.allclose(result, points)

    @pytest.mark.unit
    def test_apply_rotation_z_axis(self):
        """Test applying rotation around z-axis."""
        # 90 degree rotation around z-axis
        model = RotationModel(axis="z", angle=np.pi / 2)
        points = np.array([[1.0, 0.0, 0.0]])

        result = model.apply(points)

        # Should rotate to (0, 1, 0)
        assert np.allclose(result[0, 0], 0.0, atol=1e-6)
        assert np.allclose(result[0, 1], 1.0, atol=1e-6)
        assert np.allclose(result[0, 2], 0.0, atol=1e-6)

    @pytest.mark.unit
    def test_apply_rotation_x_axis(self):
        """Test applying rotation around x-axis."""
        # 90 degree rotation around x-axis
        model = RotationModel(axis="x", angle=np.pi / 2)
        points = np.array([[0.0, 1.0, 0.0]])

        result = model.apply(points)

        # Should rotate to (0, 0, 1)
        assert np.allclose(result[0, 0], 0.0, atol=1e-6)
        assert np.allclose(result[0, 1], 0.0, atol=1e-6)
        assert np.allclose(result[0, 2], 1.0, atol=1e-6)

    @pytest.mark.unit
    def test_apply_rotation_y_axis(self):
        """Test applying rotation around y-axis."""
        # 90 degree rotation around y-axis
        model = RotationModel(axis="y", angle=np.pi / 2)
        points = np.array([[1.0, 0.0, 0.0]])

        result = model.apply(points)

        # Should rotate to (0, 0, -1)
        assert np.allclose(result[0, 0], 0.0, atol=1e-6)
        assert np.allclose(result[0, 1], 0.0, atol=1e-6)
        assert np.allclose(result[0, 2], -1.0, atol=1e-6)

    @pytest.mark.unit
    def test_apply_rotation_with_center(self):
        """Test applying rotation with custom center."""
        model = RotationModel(axis="z", angle=np.pi / 2, center=(10.0, 20.0, 30.0))
        points = np.array([[11.0, 20.0, 30.0]])  # 1 unit in x from center

        result = model.apply(points)

        # Should rotate around center
        # Point is (11, 20, 30), center is (10, 20, 30)
        # After 90 deg rotation around z, should be (10, 21, 30)
        expected = np.array([[10.0, 21.0, 30.0]])
        assert np.allclose(result, expected, atol=1e-6)

    @pytest.mark.unit
    def test_correct_rotation(self):
        """Test correcting rotation distortion."""
        # 90 degree rotation
        model = RotationModel(axis="z", angle=np.pi / 2)
        points = np.array([[0.0, 1.0, 0.0]])  # Already rotated

        result = model.correct(points)

        # Should correct back to (1, 0, 0)
        expected = np.array([[1.0, 0.0, 0.0]])
        assert np.allclose(result, expected, atol=1e-6)

    @pytest.mark.unit
    def test_correct_rotation_invertible(self):
        """Test that apply and correct are inverse operations."""
        model = RotationModel(axis="z", angle=np.pi / 4, center=(10.0, 20.0, 30.0))
        points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        distorted = model.apply(points)
        corrected = model.correct(distorted)

        assert np.allclose(corrected, points, atol=1e-6)

    @pytest.mark.unit
    def test_get_parameters(self):
        """Test getting rotation parameters."""
        model = RotationModel(axis="x", angle=np.pi / 4, center=(10.0, 20.0, 30.0))

        params = model.get_parameters()

        assert params["type"] == "rotation"
        assert params["axis"] == "x"
        assert params["angle"] == np.pi / 4
        assert params["angle_degrees"] == 45.0
        assert params["center"] == (10.0, 20.0, 30.0)


class TestWarpingModel:
    """Test suite for WarpingModel class."""

    @pytest.mark.unit
    def test_warping_model_creation_default(self):
        """Test creating WarpingModel with default parameters."""
        model = WarpingModel()

        assert model.displacement_field is None
        assert model.reference_points is None
        assert model.displacement_vectors is None

    @pytest.mark.unit
    def test_warping_model_creation_with_data(self):
        """Test creating WarpingModel with displacement data."""
        reference_points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        displacement_vectors = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]])

        model = WarpingModel(reference_points=reference_points, displacement_vectors=displacement_vectors)

        assert model.reference_points is not None
        assert model.displacement_vectors is not None

    @pytest.mark.unit
    def test_apply_warping_no_displacement(self):
        """Test applying warping with no displacement."""
        model = WarpingModel()
        points = np.array([[1.0, 2.0, 3.0]])

        result = model.apply(points)

        # Should return points unchanged (zero displacement)
        assert np.allclose(result, points)

    @pytest.mark.unit
    def test_apply_warping_with_displacement(self):
        """Test applying warping with displacement vectors."""
        reference_points = np.array([[1.0, 2.0, 3.0]])
        displacement_vectors = np.array([[0.1, 0.2, 0.3]])

        model = WarpingModel(reference_points=reference_points, displacement_vectors=displacement_vectors)

        points = np.array([[1.0, 2.0, 3.0]])
        result = model.apply(points)

        # Should apply displacement (interpolated)
        # For exact match at reference point, should get exact displacement
        expected = points + displacement_vectors[0]
        # Note: griddata interpolation may not give exact match, so we check it's close
        assert np.allclose(result, expected, atol=0.1)

    @pytest.mark.unit
    def test_correct_warping(self):
        """Test correcting warping distortion."""
        reference_points = np.array([[1.0, 2.0, 3.0]])
        displacement_vectors = np.array([[0.1, 0.2, 0.3]])

        model = WarpingModel(reference_points=reference_points, displacement_vectors=displacement_vectors)

        # Points that have been warped
        warped_points = np.array([[1.1, 2.2, 3.3]])
        result = model.correct(warped_points)

        # Should correct by subtracting displacement
        expected = warped_points - displacement_vectors[0]
        assert np.allclose(result, expected, atol=0.1)

    @pytest.mark.unit
    def test_estimate_from_correspondences(self):
        """Test estimating warping model from correspondences."""
        model = WarpingModel()

        source_points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        target_points = np.array([[0.1, 0.1, 0.1], [1.2, 1.2, 1.2], [2.3, 2.3, 2.3]])

        model.estimate_from_correspondences(source_points, target_points)

        assert model.reference_points is not None
        assert model.displacement_vectors is not None
        assert len(model.reference_points) == 3
        assert len(model.displacement_vectors) == 3

    @pytest.mark.unit
    def test_estimate_from_correspondences_shape_mismatch(self):
        """Test estimating with shape mismatch."""
        model = WarpingModel()

        source_points = np.array([[0.0, 0.0, 0.0]])
        target_points = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]])

        with pytest.raises(ValueError, match="Source and target points must have same shape"):
            model.estimate_from_correspondences(source_points, target_points)

    @pytest.mark.unit
    def test_get_parameters(self):
        """Test getting warping parameters."""
        reference_points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        displacement_vectors = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]])

        model = WarpingModel(reference_points=reference_points, displacement_vectors=displacement_vectors)

        params = model.get_parameters()

        assert params["type"] == "warping"
        assert params["num_reference_points"] == 2


class TestCombinedDistortionModel:
    """Test suite for CombinedDistortionModel class."""

    @pytest.mark.unit
    def test_combined_model_creation(self):
        """Test creating CombinedDistortionModel."""
        scaling = ScalingModel(scale_x=2.0)
        rotation = RotationModel(axis="z", angle=np.pi / 2)

        model = CombinedDistortionModel([scaling, rotation])

        assert len(model.models) == 2

    @pytest.mark.unit
    def test_apply_combined_distortion(self):
        """Test applying combined distortions."""
        scaling = ScalingModel(scale_x=2.0, scale_y=2.0, scale_z=2.0)
        rotation = RotationModel(axis="z", angle=np.pi / 2)

        model = CombinedDistortionModel([scaling, rotation])
        points = np.array([[1.0, 0.0, 0.0]])

        result = model.apply(points)

        # First scale: (1, 0, 0) -> (2, 0, 0)
        # Then rotate 90 deg around z: (2, 0, 0) -> (0, 2, 0)
        expected = np.array([[0.0, 2.0, 0.0]])
        assert np.allclose(result, expected, atol=1e-6)

    @pytest.mark.unit
    def test_correct_combined_distortion(self):
        """Test correcting combined distortions."""
        scaling = ScalingModel(scale_x=2.0)
        rotation = RotationModel(axis="z", angle=np.pi / 2)

        model = CombinedDistortionModel([scaling, rotation])
        points = np.array([[1.0, 0.0, 0.0]])

        # Apply distortions
        distorted = model.apply(points)
        # Correct distortions (should reverse order)
        corrected = model.correct(distorted)

        assert np.allclose(corrected, points, atol=1e-5)

    @pytest.mark.unit
    def test_get_parameters(self):
        """Test getting parameters from combined model."""
        scaling = ScalingModel(scale_x=2.0)
        rotation = RotationModel(axis="z", angle=np.pi / 2)

        model = CombinedDistortionModel([scaling, rotation])

        params = model.get_parameters()

        assert params["type"] == "combined"
        assert params["num_models"] == 2
        assert len(params["models"]) == 2
        assert params["models"][0]["type"] == "scaling"
        assert params["models"][1]["type"] == "rotation"
