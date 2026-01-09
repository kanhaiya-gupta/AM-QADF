"""
Tests for point_clouds fixture module.

Tests the loading and generation functions in tests/fixtures/point_clouds/__init__.py
"""

import pytest
import json
from pathlib import Path

try:
    from tests.fixtures.point_clouds import (
        load_hatching_paths,
        load_laser_points,
        load_ct_points,
        generate_hatching_paths,
        generate_laser_points,
        generate_ct_points,
    )

    POINT_CLOUD_MODULE_AVAILABLE = True
except ImportError:
    POINT_CLOUD_MODULE_AVAILABLE = False


@pytest.mark.skipif(not POINT_CLOUD_MODULE_AVAILABLE, reason="Point cloud module not available")
class TestPointCloudLoading:
    """Tests for point cloud loading functions."""

    def test_load_hatching_paths(self):
        """Test loading hatching paths."""
        paths = load_hatching_paths()
        assert paths is not None
        assert isinstance(paths, list)
        assert len(paths) > 0

    def test_load_laser_points(self):
        """Test loading laser points."""
        points = load_laser_points()
        assert points is not None
        assert isinstance(points, list)
        assert len(points) > 0

    def test_load_ct_points(self):
        """Test loading CT scan points."""
        ct_data = load_ct_points()
        assert ct_data is not None
        assert isinstance(ct_data, dict)
        assert "points" in ct_data or "model_id" in ct_data


@pytest.mark.skipif(not POINT_CLOUD_MODULE_AVAILABLE, reason="Point cloud module not available")
class TestPointCloudGeneration:
    """Tests for point cloud generation functions."""

    def test_generate_hatching_paths(self):
        """Test generating hatching paths."""
        paths = generate_hatching_paths()
        assert paths is not None
        assert isinstance(paths, list)
        assert len(paths) == 5  # Should have 5 layers

        # Check structure of first layer
        if len(paths) > 0:
            layer = paths[0]
            assert "layer_index" in layer
            assert "hatches" in layer
            assert isinstance(layer["hatches"], list)

    def test_generate_laser_points(self):
        """Test generating laser points."""
        points = generate_laser_points()
        assert points is not None
        assert isinstance(points, list)
        assert len(points) == 1000  # Should have 1000 points

        # Check structure of first point
        if len(points) > 0:
            point = points[0]
            assert "spatial_coordinates" in point
            assert "laser_power" in point
            assert len(point["spatial_coordinates"]) == 3

    def test_generate_ct_points(self):
        """Test generating CT scan points."""
        ct_data = generate_ct_points()
        assert ct_data is not None
        assert isinstance(ct_data, dict)
        assert "model_id" in ct_data
        assert "points" in ct_data
        assert len(ct_data["points"]) == 500  # Should have 500 points

        # Check structure of first point
        if len(ct_data["points"]) > 0:
            point = ct_data["points"][0]
            assert "x" in point
            assert "y" in point
            assert "z" in point
            assert "density" in point


@pytest.mark.skipif(not POINT_CLOUD_MODULE_AVAILABLE, reason="Point cloud module not available")
class TestPointCloudDataStructure:
    """Tests for point cloud data structure validation."""

    def test_hatching_paths_structure(self):
        """Test that hatching paths have correct structure."""
        paths = generate_hatching_paths()
        for layer in paths:
            assert "layer_index" in layer
            assert "hatches" in layer
            for hatch in layer["hatches"]:
                assert "points" in hatch
                assert "laser_power" in hatch
                assert "scan_speed" in hatch

    def test_laser_points_structure(self):
        """Test that laser points have correct structure."""
        points = generate_laser_points()
        for point in points[:10]:  # Check first 10 points
            assert "spatial_coordinates" in point
            assert len(point["spatial_coordinates"]) == 3
            assert "laser_power" in point
            assert "scan_speed" in point

    def test_ct_points_structure(self):
        """Test that CT points have correct structure."""
        ct_data = generate_ct_points()
        assert "voxel_grid" in ct_data
        assert "points" in ct_data
        for point in ct_data["points"][:10]:  # Check first 10 points
            assert "x" in point
            assert "y" in point
            assert "z" in point
            assert "density" in point
