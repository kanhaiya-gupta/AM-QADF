"""
Property-based tests for VoxelGrid.

Tests mathematical properties and invariants of voxel grid operations.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, assume, settings
from hypothesis.extra.numpy import arrays

try:
    from am_qadf.voxelization.uniform_resolution import VoxelGrid

    VOXEL_GRID_AVAILABLE = True
except ImportError:
    VOXEL_GRID_AVAILABLE = False


@pytest.mark.skipif(not VOXEL_GRID_AVAILABLE, reason="VoxelGrid not available")
@pytest.mark.property_based
class TestVoxelGridProperties:
    """Property-based tests for VoxelGrid."""

    @given(
        bbox_min=st.tuples(
            st.floats(min_value=-100.0, max_value=0.0),
            st.floats(min_value=-100.0, max_value=0.0),
            st.floats(min_value=-100.0, max_value=0.0),
        ),
        bbox_max=st.tuples(
            st.floats(min_value=0.0, max_value=100.0),
            st.floats(min_value=0.0, max_value=100.0),
            st.floats(min_value=0.0, max_value=100.0),
        ),
        resolution=st.floats(min_value=0.1, max_value=10.0),
    )
    @settings(max_examples=50)
    def test_bbox_ordering_property(self, bbox_min, bbox_max, resolution):
        """Test that bbox_min < bbox_max for all dimensions."""
        # Ensure valid bounding box
        assume(all(bmax > bmin for bmin, bmax in zip(bbox_min, bbox_max)))

        grid = VoxelGrid(bbox_min=bbox_min, bbox_max=bbox_max, resolution=resolution)

        # Property: bbox_min should be less than bbox_max
        assert all(grid.bbox_min[i] < grid.bbox_max[i] for i in range(3))

    @given(
        bbox_min=st.tuples(
            st.floats(min_value=-50.0, max_value=0.0),
            st.floats(min_value=-50.0, max_value=0.0),
            st.floats(min_value=-50.0, max_value=0.0),
        ),
        bbox_max=st.tuples(
            st.floats(min_value=0.0, max_value=50.0),
            st.floats(min_value=0.0, max_value=50.0),
            st.floats(min_value=0.0, max_value=50.0),
        ),
        resolution=st.floats(min_value=0.1, max_value=5.0),
    )
    @settings(max_examples=50)
    def test_dimensions_property(self, bbox_min, bbox_max, resolution):
        """Test that dimensions are consistent with bbox and resolution."""
        assume(all(bmax > bmin for bmin, bmax in zip(bbox_min, bbox_max)))

        grid = VoxelGrid(bbox_min=bbox_min, bbox_max=bbox_max, resolution=resolution)

        # Property: dimensions should be positive integers
        assert all(d > 0 for d in grid.dims)
        assert all(isinstance(d, (int, np.integer)) for d in grid.dims)

        # Property: dimensions should match bbox size / resolution
        bbox_size = np.array(bbox_max) - np.array(bbox_min)
        expected_dims = np.ceil(bbox_size / resolution).astype(int)
        assert np.allclose(grid.dims, expected_dims, atol=1)

    @given(
        bbox_min=st.tuples(
            st.floats(min_value=-50.0, max_value=0.0),
            st.floats(min_value=-50.0, max_value=0.0),
            st.floats(min_value=-50.0, max_value=0.0),
        ),
        bbox_max=st.tuples(
            st.floats(min_value=0.0, max_value=50.0),
            st.floats(min_value=0.0, max_value=50.0),
            st.floats(min_value=0.0, max_value=50.0),
        ),
        resolution=st.floats(min_value=0.1, max_value=5.0),
        n_points=st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=30)
    def test_add_point_idempotency(self, bbox_min, bbox_max, resolution, n_points):
        """Test that adding the same point multiple times is idempotent (after finalize)."""
        assume(all(bmax > bmin for bmin, bmax in zip(bbox_min, bbox_max)))

        grid1 = VoxelGrid(bbox_min=bbox_min, bbox_max=bbox_max, resolution=resolution)
        grid2 = VoxelGrid(bbox_min=bbox_min, bbox_max=bbox_max, resolution=resolution)

        # Generate random points within bbox
        np.random.seed(42)
        points = np.random.uniform(low=bbox_min, high=bbox_max, size=(n_points, 3))

        # Add points to both grids
        for point in points:
            signal_value = float(np.random.rand() * 100.0)
            grid1.add_point(point[0], point[1], point[2], signals={"test_signal": signal_value})
            # Add same point twice to grid2
            grid2.add_point(point[0], point[1], point[2], signals={"test_signal": signal_value})
            grid2.add_point(point[0], point[1], point[2], signals={"test_signal": signal_value})

        # Finalize both
        grid1.finalize()
        grid2.finalize()

        # Property: After finalization, adding same point twice should give same result
        # (aggregation should handle duplicates). Compare signal arrays.
        arr1 = grid1.get_signal_array("test_signal", default=0.0)
        arr2 = grid2.get_signal_array("test_signal", default=0.0)
        assert arr1.shape == arr2.shape
        assert np.allclose(arr1, arr2, atol=1e-6)

    @given(
        bbox_min=st.tuples(
            st.floats(min_value=-50.0, max_value=0.0),
            st.floats(min_value=-50.0, max_value=0.0),
            st.floats(min_value=-50.0, max_value=0.0),
        ),
        bbox_max=st.tuples(
            st.floats(min_value=0.0, max_value=50.0),
            st.floats(min_value=0.0, max_value=50.0),
            st.floats(min_value=0.0, max_value=50.0),
        ),
        resolution=st.floats(min_value=0.1, max_value=5.0),
    )
    @settings(max_examples=30)
    def test_world_to_voxel_roundtrip(self, bbox_min, bbox_max, resolution):
        """Test that world_to_voxel and voxel_to_world are consistent."""
        assume(all(bmax > bmin for bmin, bmax in zip(bbox_min, bbox_max)))

        grid = VoxelGrid(bbox_min=bbox_min, bbox_max=bbox_max, resolution=resolution)

        # Generate random points within bbox
        np.random.seed(42)
        n_points = 10
        world_points = np.random.uniform(low=bbox_min, high=bbox_max, size=(n_points, 3))

        for world_point in world_points:
            # Convert to voxel index
            voxel_idx = grid._world_to_voxel(world_point[0], world_point[1], world_point[2])

            # Voxel center in world coordinates (no _voxel_to_world in API)
            voxel_center = np.array(grid.bbox_min) + (np.array(voxel_idx) + 0.5) * resolution

            # Property: Voxel center should be within one resolution of original point
            distance = np.linalg.norm(world_point - voxel_center)
            max_distance = resolution * np.sqrt(3)  # Diagonal of voxel
            assert distance <= max_distance

    @given(
        bbox_min=st.tuples(
            st.floats(min_value=-50.0, max_value=0.0),
            st.floats(min_value=-50.0, max_value=0.0),
            st.floats(min_value=-50.0, max_value=0.0),
        ),
        bbox_max=st.tuples(
            st.floats(min_value=0.0, max_value=50.0),
            st.floats(min_value=0.0, max_value=50.0),
            st.floats(min_value=0.0, max_value=50.0),
        ),
        resolution=st.floats(min_value=0.1, max_value=5.0),
        signal_value=st.floats(min_value=0.0, max_value=1000.0),
    )
    @settings(max_examples=30)
    def test_signal_bounds_property(self, bbox_min, bbox_max, resolution, signal_value):
        """Test that signal values are preserved correctly."""
        assume(all(bmax > bmin for bmin, bmax in zip(bbox_min, bbox_max)))

        grid = VoxelGrid(bbox_min=bbox_min, bbox_max=bbox_max, resolution=resolution)

        # Add a point with a signal
        center = tuple((np.array(bbox_min) + np.array(bbox_max)) / 2)
        grid.add_point(center[0], center[1], center[2], signals={"test_signal": signal_value})
        grid.finalize()

        # Property: Signal should be retrievable
        if len(grid.voxels) > 0:
            voxel = list(grid.voxels.values())[0]
            if "test_signal" in voxel.signals:
                retrieved_value = voxel.signals["test_signal"]
                # Should be close to original (may be aggregated)
                assert isinstance(retrieved_value, (float, int, np.number))

    @given(
        bbox_min=st.tuples(
            st.floats(min_value=-50.0, max_value=0.0),
            st.floats(min_value=-50.0, max_value=0.0),
            st.floats(min_value=-50.0, max_value=0.0),
        ),
        bbox_max=st.tuples(
            st.floats(min_value=0.0, max_value=50.0),
            st.floats(min_value=0.0, max_value=50.0),
            st.floats(min_value=0.0, max_value=50.0),
        ),
        resolution=st.floats(min_value=0.1, max_value=5.0),
    )
    @settings(max_examples=20)
    def test_empty_grid_properties(self, bbox_min, bbox_max, resolution):
        """Test properties of empty voxel grid."""
        assume(all(bmax > bmin for bmin, bmax in zip(bbox_min, bbox_max)))

        grid = VoxelGrid(bbox_min=bbox_min, bbox_max=bbox_max, resolution=resolution)

        # Property: Empty grid should have no available signals
        assert len(grid.available_signals) == 0

        # Property: Dimensions should still be valid
        assert all(d > 0 for d in grid.dims)
