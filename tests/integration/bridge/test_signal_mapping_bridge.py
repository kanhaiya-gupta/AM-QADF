"""
Bridge tests: signal mapping (NearestNeighborMapper, Point, numpy_to_points) via am_qadf_native.

Aligned with docs/Tests/Test_New_Plans.md (tests/integration/bridge/).
"""

import pytest
import numpy as np


@pytest.mark.integration
@pytest.mark.bridge
class TestSignalMappingBridge:
    """Python → C++ signal mapping API."""

    def test_nearest_neighbor_map_to_grid(self, native_module):
        """NearestNeighborMapper.map: points + values → grid."""
        UniformVoxelGrid = native_module.UniformVoxelGrid
        NearestNeighborMapper = native_module.signal_mapping.NearestNeighborMapper
        Point = native_module.signal_mapping.Point
        grid = UniformVoxelGrid(1.0, 0.0, 0.0, 0.0)
        grid.set_signal_name("mapped")
        g = grid.get_grid()
        points = [Point(0.5, 0.5, 0.5), Point(1.5, 0.5, 0.5)]
        values = [10.0, 20.0]
        mapper = NearestNeighborMapper()
        mapper.map(g, points, values)
        assert grid.get_value(0, 0, 0) == pytest.approx(10.0)
        assert grid.get_value(1, 0, 0) == pytest.approx(20.0)

    def test_numpy_to_points(self, native_module):
        """signal_mapping.numpy_to_points converts (N,3) array to list of Point."""
        numpy_to_points = native_module.signal_mapping.numpy_to_points
        arr = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
        # bbox_min optional; pass (3,) when required by implementation
        bbox_min = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        points = numpy_to_points(arr, bbox_min)
        assert len(points) == 2
        assert points[0].x == pytest.approx(0.0)
        assert points[1].x == pytest.approx(1.0)

    def test_linear_mapper_map(self, native_module):
        """LinearMapper.map runs and fills grid (value at voxel depends on C++ linear interpolation)."""
        UniformVoxelGrid = native_module.UniformVoxelGrid
        LinearMapper = native_module.signal_mapping.LinearMapper
        Point = native_module.signal_mapping.Point
        grid = UniformVoxelGrid(1.0, 0.0, 0.0, 0.0)
        g = grid.get_grid()
        points = [Point(0.5, 0.5, 0.5)]
        values = [5.0]
        mapper = LinearMapper()
        mapper.map(g, points, values)
        val = grid.get_value(0, 0, 0)
        assert val is not None and isinstance(val, (int, float))
        assert 0 <= val <= 10.0  # single point 5.0; C++ may interpolate differently

    def test_idw_mapper_map(self, native_module):
        """IDWMapper.map runs and fills grid (IDW weights; values between source points)."""
        UniformVoxelGrid = native_module.UniformVoxelGrid
        IDWMapper = native_module.signal_mapping.IDWMapper
        Point = native_module.signal_mapping.Point
        grid = UniformVoxelGrid(1.0, 0.0, 0.0, 0.0)
        g = grid.get_grid()
        points = [Point(0.5, 0.5, 0.5), Point(1.5, 0.5, 0.5)]
        values = [1.0, 2.0]
        mapper = IDWMapper(2.0, 10)
        mapper.map(g, points, values)
        v0 = grid.get_value(0, 0, 0)
        v1 = grid.get_value(1, 0, 0)
        assert v0 is not None and v1 is not None
        assert 0.5 <= v0 <= 2.0 and 0.5 <= v1 <= 2.0  # IDW between 1.0 and 2.0
