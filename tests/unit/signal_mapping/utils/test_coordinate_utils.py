"""
Unit tests for coordinate transformation utilities.

Tests for coordinate transformation helper functions.
"""

import pytest
import numpy as np
from am_qadf.signal_mapping.utils.coordinate_utils import (
    transform_coordinates,
    align_to_voxel_grid,
    get_voxel_centers,
)


class TestTransformCoordinates:
    """Test suite for transform_coordinates function."""

    @pytest.mark.unit
    def test_transform_coordinates_identity(self):
        """Test transform_coordinates with identity transformation."""
        points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        from_system = {}
        to_system = {}

        result = transform_coordinates(points, from_system, to_system)

        assert np.array_equal(result, points)
        assert result is not points  # Should be a copy

    @pytest.mark.unit
    def test_transform_coordinates_single_point(self):
        """Test transform_coordinates with single point."""
        points = np.array([[5.0, 6.0, 7.0]])
        from_system = {}
        to_system = {}

        result = transform_coordinates(points, from_system, to_system)

        assert result.shape == (1, 3)
        assert np.array_equal(result, points)

    @pytest.mark.unit
    def test_transform_coordinates_empty(self):
        """Test transform_coordinates with empty points."""
        points = np.array([]).reshape(0, 3)
        from_system = {}
        to_system = {}

        result = transform_coordinates(points, from_system, to_system)

        assert result.shape == (0, 3)
        assert len(result) == 0


class TestAlignToVoxelGrid:
    """Test suite for align_to_voxel_grid function."""

    @pytest.mark.unit
    def test_align_to_voxel_grid_basic(self):
        """Test aligning points to voxel grid origin."""
        points = np.array([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]])
        voxel_grid_origin = (2.0, 3.0, 4.0)
        voxel_resolution = 1.0

        result = align_to_voxel_grid(points, voxel_grid_origin, voxel_resolution)

        expected = points - np.array(voxel_grid_origin)
        assert np.allclose(result, expected)

    @pytest.mark.unit
    def test_align_to_voxel_grid_origin_zero(self):
        """Test aligning points when grid origin is zero."""
        points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        voxel_grid_origin = (0.0, 0.0, 0.0)
        voxel_resolution = 1.0

        result = align_to_voxel_grid(points, voxel_grid_origin, voxel_resolution)

        assert np.allclose(result, points)

    @pytest.mark.unit
    def test_align_to_voxel_grid_single_point(self):
        """Test aligning single point to voxel grid."""
        points = np.array([[5.0, 6.0, 7.0]])
        voxel_grid_origin = (2.0, 3.0, 4.0)
        voxel_resolution = 1.0

        result = align_to_voxel_grid(points, voxel_grid_origin, voxel_resolution)

        expected = np.array([[3.0, 3.0, 3.0]])
        assert np.allclose(result, expected)

    @pytest.mark.unit
    def test_align_to_voxel_grid_empty(self):
        """Test aligning empty points array."""
        points = np.array([]).reshape(0, 3)
        voxel_grid_origin = (0.0, 0.0, 0.0)
        voxel_resolution = 1.0

        result = align_to_voxel_grid(points, voxel_grid_origin, voxel_resolution)

        assert result.shape == (0, 3)
        assert len(result) == 0


class TestGetVoxelCenters:
    """Test suite for get_voxel_centers function."""

    @pytest.mark.unit
    def test_get_voxel_centers_basic(self):
        """Test getting voxel centers from indices."""
        voxel_indices = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        voxel_grid_origin = (0.0, 0.0, 0.0)
        voxel_resolution = 1.0

        result = get_voxel_centers(voxel_indices, voxel_grid_origin, voxel_resolution)

        # Voxel center = origin + (index + 0.5) * resolution
        expected = np.array([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5], [2.5, 2.5, 2.5]])
        assert np.allclose(result, expected)

    @pytest.mark.unit
    def test_get_voxel_centers_with_origin(self):
        """Test getting voxel centers with non-zero origin."""
        voxel_indices = np.array([[0, 0, 0], [1, 1, 1]])
        voxel_grid_origin = (10.0, 20.0, 30.0)
        voxel_resolution = 2.0

        result = get_voxel_centers(voxel_indices, voxel_grid_origin, voxel_resolution)

        # Voxel center = origin + (index + 0.5) * resolution
        expected = np.array(
            [
                [10.0 + 0.5 * 2.0, 20.0 + 0.5 * 2.0, 30.0 + 0.5 * 2.0],
                [10.0 + 1.5 * 2.0, 20.0 + 1.5 * 2.0, 30.0 + 1.5 * 2.0],
            ]
        )
        assert np.allclose(result, expected)

    @pytest.mark.unit
    def test_get_voxel_centers_single_index(self):
        """Test getting voxel center for single index."""
        voxel_indices = np.array([[5, 6, 7]])
        voxel_grid_origin = (0.0, 0.0, 0.0)
        voxel_resolution = 1.0

        result = get_voxel_centers(voxel_indices, voxel_grid_origin, voxel_resolution)

        expected = np.array([[5.5, 6.5, 7.5]])
        assert np.allclose(result, expected)

    @pytest.mark.unit
    def test_get_voxel_centers_empty(self):
        """Test getting voxel centers for empty indices."""
        voxel_indices = np.array([]).reshape(0, 3)
        voxel_grid_origin = (0.0, 0.0, 0.0)
        voxel_resolution = 1.0

        result = get_voxel_centers(voxel_indices, voxel_grid_origin, voxel_resolution)

        assert result.shape == (0, 3)
        assert len(result) == 0

    @pytest.mark.unit
    def test_get_voxel_centers_different_resolution(self):
        """Test getting voxel centers with different resolution."""
        voxel_indices = np.array([[0, 0, 0], [1, 1, 1]])
        voxel_grid_origin = (0.0, 0.0, 0.0)
        voxel_resolution = 0.5

        result = get_voxel_centers(voxel_indices, voxel_grid_origin, voxel_resolution)

        expected = np.array([[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]])
        assert np.allclose(result, expected)
