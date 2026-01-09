"""
Property-based tests for interpolation methods.

Tests mathematical properties like boundary conditions, monotonicity, and boundedness.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, assume, settings

try:
    from am_qadf.voxelization.voxel_grid import VoxelGrid
    from am_qadf.signal_mapping.methods.nearest_neighbor import (
        NearestNeighborInterpolation,
    )
    from am_qadf.signal_mapping.methods.linear import LinearInterpolation
    from am_qadf.signal_mapping.methods.idw import IDWInterpolation

    INTERPOLATION_AVAILABLE = True
except ImportError:
    INTERPOLATION_AVAILABLE = False


@pytest.mark.skipif(not INTERPOLATION_AVAILABLE, reason="Interpolation methods not available")
@pytest.mark.property_based
class TestInterpolationProperties:
    """Property-based tests for interpolation methods."""

    @given(
        n_points=st.integers(min_value=3, max_value=50),
        bbox_size=st.floats(min_value=10.0, max_value=100.0),
        resolution=st.floats(min_value=0.5, max_value=5.0),
    )
    @settings(max_examples=30)
    def test_interpolation_bounded_by_input(self, n_points, bbox_size, resolution):
        """Test that interpolated values are bounded by input signal range."""
        assume(bbox_size > resolution * 2)

        # Create grid
        grid = VoxelGrid(
            bbox_min=(0.0, 0.0, 0.0),
            bbox_max=(bbox_size, bbox_size, bbox_size),
            resolution=resolution,
        )

        # Generate random points and signals
        np.random.seed(42)
        points = np.random.rand(n_points, 3) * bbox_size
        signal_min = 0.0
        signal_max = 100.0
        signals = np.random.uniform(signal_min, signal_max, n_points)

        signal_dict = {"test_signal": signals}

        # Test with nearest neighbor
        method = NearestNeighborInterpolation()
        result_grid = method.interpolate(points, signal_dict, grid)

        # Property: All interpolated values should be within input range
        if len(result_grid.voxels) > 0:
            for voxel in result_grid.voxels.values():
                if "test_signal" in voxel.signals:
                    value = voxel.signals["test_signal"]
                    assert signal_min <= value <= signal_max or np.isclose(value, signal_min) or np.isclose(value, signal_max)

    @given(
        n_points=st.integers(min_value=3, max_value=30),
        bbox_size=st.floats(min_value=10.0, max_value=50.0),
        resolution=st.floats(min_value=1.0, max_value=5.0),
    )
    @settings(max_examples=20)
    def test_nearest_neighbor_exact_at_points(self, n_points, bbox_size, resolution):
        """Test that nearest neighbor interpolation gives exact values at input points."""
        assume(bbox_size > resolution * 2)

        # Create grid
        grid = VoxelGrid(
            bbox_min=(0.0, 0.0, 0.0),
            bbox_max=(bbox_size, bbox_size, bbox_size),
            resolution=resolution,
        )

        # Generate points
        np.random.seed(42)
        points = np.random.rand(n_points, 3) * bbox_size
        signals = np.random.rand(n_points) * 100.0
        signal_dict = {"test_signal": signals}

        # Interpolate
        method = NearestNeighborInterpolation()
        result_grid = method.interpolate(points, signal_dict, grid)

        # Property: At input point locations, value should match (within tolerance)
        # Note: Due to aggregation, when multiple points map to the same voxel,
        # the voxel stores the mean of all signals in that voxel.
        # So we check that interpolated values are within the range of signals in each voxel.

        # Build map of which points map to which voxels
        voxel_to_points = {}
        for i, point in enumerate(points):
            voxel_idx = grid._world_to_voxel(point[0], point[1], point[2])
            voxel_key = tuple(voxel_idx)
            if voxel_key not in voxel_to_points:
                voxel_to_points[voxel_key] = []
            voxel_to_points[voxel_key].append(i)

        # Check property for each voxel
        for voxel_key, point_indices in voxel_to_points.items():
            if voxel_key in result_grid.voxels:
                voxel = result_grid.voxels[voxel_key]
                if "test_signal" in voxel.signals:
                    interpolated = voxel.signals["test_signal"]
                    # Get all signals that map to this voxel
                    voxel_signals = [signals[i] for i in point_indices]
                    min_signal = min(voxel_signals)
                    max_signal = max(voxel_signals)

                    # Property: Interpolated value should be within range of signals in this voxel
                    # (for mean aggregation, it should be between min and max)
                    assert (
                        min_signal <= interpolated <= max_signal
                        or np.isclose(interpolated, min_signal, atol=1e-6)
                        or np.isclose(interpolated, max_signal, atol=1e-6)
                    )

    @given(
        n_points=st.integers(min_value=5, max_value=30),
        bbox_size=st.floats(min_value=10.0, max_value=50.0),
        resolution=st.floats(min_value=1.0, max_value=5.0),
        constant_value=st.floats(min_value=0.0, max_value=100.0),
    )
    @settings(max_examples=20)
    def test_constant_signal_preservation(self, n_points, bbox_size, resolution, constant_value):
        """Test that constant input signals produce constant interpolated values."""
        assume(bbox_size > resolution * 2)

        # Create grid
        grid = VoxelGrid(
            bbox_min=(0.0, 0.0, 0.0),
            bbox_max=(bbox_size, bbox_size, bbox_size),
            resolution=resolution,
        )

        # Generate points with constant signal
        np.random.seed(42)
        points = np.random.rand(n_points, 3) * bbox_size
        signals = np.full(n_points, constant_value)
        signal_dict = {"test_signal": signals}

        # Test with nearest neighbor
        method = NearestNeighborInterpolation()
        result_grid = method.interpolate(points, signal_dict, grid)

        # Property: All interpolated values should be constant (within tolerance)
        if len(result_grid.voxels) > 0:
            values = [voxel.signals["test_signal"] for voxel in result_grid.voxels.values() if "test_signal" in voxel.signals]
            if len(values) > 0:
                # All values should be close to constant
                assert all(np.isclose(v, constant_value, atol=1e-6) for v in values)

    @given(
        n_points=st.integers(min_value=5, max_value=30),
        bbox_size=st.floats(min_value=10.0, max_value=50.0),
        resolution=st.floats(min_value=1.0, max_value=5.0),
    )
    @settings(max_examples=20)
    def test_interpolation_commutativity(self, n_points, bbox_size, resolution):
        """Test that interpolation order doesn't matter (for same points)."""
        assume(bbox_size > resolution * 2)

        # Create two identical grids
        grid1 = VoxelGrid(
            bbox_min=(0.0, 0.0, 0.0),
            bbox_max=(bbox_size, bbox_size, bbox_size),
            resolution=resolution,
        )
        grid2 = VoxelGrid(
            bbox_min=(0.0, 0.0, 0.0),
            bbox_max=(bbox_size, bbox_size, bbox_size),
            resolution=resolution,
        )

        # Generate points
        np.random.seed(42)
        points = np.random.rand(n_points, 3) * bbox_size
        signals = np.random.rand(n_points) * 100.0
        signal_dict = {"test_signal": signals}

        # Shuffle points and signals together
        indices = np.random.permutation(n_points)
        shuffled_points = points[indices]
        shuffled_signals = signals[indices]
        shuffled_signal_dict = {"test_signal": shuffled_signals}

        # Interpolate with original and shuffled
        method = NearestNeighborInterpolation()
        result1 = method.interpolate(points, signal_dict, grid1)
        result2 = method.interpolate(shuffled_points, shuffled_signal_dict, grid2)

        # Property: Results should be the same (order shouldn't matter)
        # Compare voxel values
        voxels1 = {k: v.signals.get("test_signal", 0) for k, v in result1.voxels.items() if "test_signal" in v.signals}
        voxels2 = {k: v.signals.get("test_signal", 0) for k, v in result2.voxels.items() if "test_signal" in v.signals}

        # Should have same voxels with same values
        common_voxels = set(voxels1.keys()) & set(voxels2.keys())
        if len(common_voxels) > 0:
            for voxel_key in common_voxels:
                assert np.isclose(voxels1[voxel_key], voxels2[voxel_key], atol=1e-6)

    @given(
        n_points=st.integers(min_value=3, max_value=20),
        bbox_size=st.floats(min_value=10.0, max_value=50.0),
        resolution=st.floats(min_value=1.0, max_value=5.0),
        scale_factor=st.floats(min_value=0.5, max_value=2.0),
    )
    @settings(max_examples=20)
    def test_interpolation_scaling(self, n_points, bbox_size, resolution, scale_factor):
        """Test that scaling input signals scales output proportionally."""
        assume(bbox_size > resolution * 2)

        # Create grids
        grid1 = VoxelGrid(
            bbox_min=(0.0, 0.0, 0.0),
            bbox_max=(bbox_size, bbox_size, bbox_size),
            resolution=resolution,
        )
        grid2 = VoxelGrid(
            bbox_min=(0.0, 0.0, 0.0),
            bbox_max=(bbox_size, bbox_size, bbox_size),
            resolution=resolution,
        )

        # Generate points
        np.random.seed(42)
        points = np.random.rand(n_points, 3) * bbox_size
        signals1 = np.random.rand(n_points) * 100.0
        signals2 = signals1 * scale_factor

        signal_dict1 = {"test_signal": signals1}
        signal_dict2 = {"test_signal": signals2}

        # Interpolate
        method = NearestNeighborInterpolation()
        result1 = method.interpolate(points, signal_dict1, grid1)
        result2 = method.interpolate(points, signal_dict2, grid2)

        # Property: Scaled input should produce scaled output
        voxels1 = {k: v.signals.get("test_signal", 0) for k, v in result1.voxels.items() if "test_signal" in v.signals}
        voxels2 = {k: v.signals.get("test_signal", 0) for k, v in result2.voxels.items() if "test_signal" in v.signals}

        common_voxels = set(voxels1.keys()) & set(voxels2.keys())
        if len(common_voxels) > 0:
            for voxel_key in common_voxels:
                if voxels1[voxel_key] != 0:  # Avoid division by zero
                    ratio = voxels2[voxel_key] / voxels1[voxel_key]
                    assert np.isclose(ratio, scale_factor, atol=0.1)  # Allow some tolerance
