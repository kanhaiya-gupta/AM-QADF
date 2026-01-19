"""
Unit tests for CoordinateSystemTransformer.

Tests for coordinate system transformation functionality.
"""

import pytest
import numpy as np
from am_qadf.voxelization.transformer import CoordinateSystemTransformer


class TestCoordinateSystemTransformer:
    """Test suite for CoordinateSystemTransformer class."""

    @pytest.mark.unit
    def test_transformer_creation(self):
        """Test creating CoordinateSystemTransformer."""
        transformer = CoordinateSystemTransformer()

        assert transformer is not None

    @pytest.mark.unit
    def test_transform_point_no_transformation(self):
        """Test transforming point with no transformation."""
        transformer = CoordinateSystemTransformer()

        from_system = {"origin": [0.0, 0.0, 0.0]}
        to_system = {"origin": [0.0, 0.0, 0.0]}

        point = (1.0, 2.0, 3.0)
        transformed = transformer.transform_point(point, from_system, to_system)

        assert np.allclose(transformed, point)

    @pytest.mark.unit
    def test_transform_point_translation_only(self):
        """Test transforming point with translation only."""
        transformer = CoordinateSystemTransformer()

        from_system = {"origin": [0.0, 0.0, 0.0]}
        to_system = {"origin": [10.0, 20.0, 30.0]}

        point = (1.0, 2.0, 3.0)
        transformed = transformer.transform_point(point, from_system, to_system)

        expected = (11.0, 22.0, 33.0)
        assert np.allclose(transformed, expected)

    @pytest.mark.unit
    def test_transform_point_scale_only(self):
        """Test transforming point with scale only."""
        transformer = CoordinateSystemTransformer()

        from_system = {
            "origin": [0.0, 0.0, 0.0],
            "scale_factor": {"x": 1.0, "y": 1.0, "z": 1.0},
        }
        to_system = {
            "origin": [0.0, 0.0, 0.0],
            "scale_factor": {"x": 2.0, "y": 2.0, "z": 2.0},
        }

        point = (1.0, 2.0, 3.0)
        transformed = transformer.transform_point(point, from_system, to_system)

        expected = (2.0, 4.0, 6.0)
        assert np.allclose(transformed, expected)

    @pytest.mark.unit
    def test_transform_points_batch(self):
        """Test transforming multiple points."""
        transformer = CoordinateSystemTransformer()

        from_system = {"origin": [0.0, 0.0, 0.0]}
        to_system = {"origin": [10.0, 20.0, 30.0]}

        points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

        transformed = transformer.transform_points(points, from_system, to_system)

        assert transformed.shape == points.shape
        expected = points + np.array([10.0, 20.0, 30.0])
        assert np.allclose(transformed, expected)

    @pytest.mark.unit
    def test_transform_points_empty(self):
        """Test transforming empty points array."""
        transformer = CoordinateSystemTransformer()

        from_system = {"origin": [0.0, 0.0, 0.0]}
        to_system = {"origin": [10.0, 20.0, 30.0]}

        points = np.array([]).reshape(0, 3)
        transformed = transformer.transform_points(points, from_system, to_system)

        assert len(transformed) == 0

    @pytest.mark.unit
    def test_transform_points_single_point(self):
        """Test transforming single point as array."""
        transformer = CoordinateSystemTransformer()

        from_system = {"origin": [0.0, 0.0, 0.0]}
        to_system = {"origin": [10.0, 20.0, 30.0]}

        points = np.array([1.0, 2.0, 3.0])  # 1D array
        transformed = transformer.transform_points(points, from_system, to_system)

        assert transformed.shape == (1, 3)
        assert np.allclose(transformed[0], [11.0, 22.0, 33.0])

    @pytest.mark.unit
    def test_transform_point_invertible(self):
        """Test that transformation is invertible."""
        transformer = CoordinateSystemTransformer()

        from_system = {
            "origin": [10.0, 20.0, 30.0],
            "scale_factor": {"x": 2.0, "y": 2.0, "z": 2.0},
        }
        to_system = {
            "origin": [0.0, 0.0, 0.0],
            "scale_factor": {"x": 1.0, "y": 1.0, "z": 1.0},
        }

        point = (1.0, 2.0, 3.0)

        # Transform forward
        transformed = transformer.transform_point(point, from_system, to_system)

        # Transform back
        back_transformed = transformer.transform_point(transformed, to_system, from_system)

        assert np.allclose(back_transformed, point, atol=1e-5)

    @pytest.mark.unit
    def test_transform_point_with_rotation(self):
        """Test transforming point with rotation."""
        transformer = CoordinateSystemTransformer()

        # 90 degree rotation around z-axis
        from_system = {
            "origin": [0.0, 0.0, 0.0],
            "rotation": {"axis": "z", "angle": 90.0},
        }
        to_system = {"origin": [0.0, 0.0, 0.0]}

        point = (1.0, 0.0, 0.0)  # Point on x-axis
        transformed = transformer.transform_point(point, from_system, to_system)

        # When transforming FROM a coordinate system rotated 90° around z-axis TO an unrotated system,
        # we apply the inverse rotation. A point (1, 0, 0) in the rotated system becomes (0, -1, 0)
        # in the unrotated system (inverse of 90° rotation around z-axis).
        assert np.allclose(transformed[1], -1.0, atol=1e-5)
        assert np.allclose(transformed[0], 0.0, atol=1e-5)
        assert np.allclose(transformed[2], 0.0, atol=1e-5)