"""
Tests for test helper functions.

Ensures that test utilities work correctly.
"""

import pytest
import numpy as np
from tests.utils.test_helpers import (
    assert_array_close,
    assert_voxel_grid_valid,
    create_test_points,
    create_test_signal_array,
    assert_signal_valid,
)


class TestAssertArrayClose:
    """Tests for assert_array_close function."""

    def test_identical_arrays(self):
        """Test that identical arrays pass."""
        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([1.0, 2.0, 3.0])
        assert_array_close(arr1, arr2)

    def test_close_arrays_within_tolerance(self):
        """Test that arrays within tolerance pass."""
        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([1.0 + 1e-6, 2.0, 3.0])
        assert_array_close(arr1, arr2)

    def test_different_arrays_raises(self):
        """Test that different arrays raise assertion error."""
        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([1.0, 2.0, 4.0])
        with pytest.raises(AssertionError):
            assert_array_close(arr1, arr2)

    def test_custom_tolerance(self):
        """Test that custom tolerance works."""
        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([1.1, 2.0, 3.0])
        # Should pass with larger tolerance
        assert_array_close(arr1, arr2, rtol=0.2)
        # Should fail with smaller tolerance
        with pytest.raises(AssertionError):
            assert_array_close(arr1, arr2, rtol=0.05)

    def test_custom_message(self):
        """Test that custom error message is used."""
        arr1 = np.array([1.0, 2.0])
        arr2 = np.array([1.0, 3.0])
        with pytest.raises(AssertionError, match="Custom error"):
            assert_array_close(arr1, arr2, msg="Custom error")


class TestAssertVoxelGridValid:
    """Tests for assert_voxel_grid_valid function."""

    def test_valid_voxel_grid(self):
        """Test that valid voxel grid passes (dimensions attribute, e.g. mocks)."""

        class MockVoxelGrid:
            def __init__(self):
                self.dimensions = (10, 20, 30)
                self.resolution = 0.1

        grid = MockVoxelGrid()
        assert_voxel_grid_valid(grid)

    def test_valid_voxel_grid_with_dims(self):
        """Test that valid voxel grid with .dims passes (uniform_resolution.VoxelGrid API)."""

        class MockVoxelGridDims:
            def __init__(self):
                self.dims = (10, 20, 30)
                self.resolution = 0.1

        grid = MockVoxelGridDims()
        assert_voxel_grid_valid(grid)

    def test_none_voxel_grid_raises(self):
        """Test that None voxel grid raises error."""
        with pytest.raises(AssertionError, match="is None"):
            assert_voxel_grid_valid(None)

    def test_none_dimensions_raises(self):
        """Test that None dimensions raise error."""

        class MockVoxelGrid:
            def __init__(self):
                self.dimensions = None
                self.resolution = 0.1

        grid = MockVoxelGrid()
        with pytest.raises(AssertionError, match="dimensions are None"):
            assert_voxel_grid_valid(grid)

    def test_wrong_dimension_count_raises(self):
        """Test that wrong dimension count raises error."""

        class MockVoxelGrid:
            def __init__(self):
                self.dimensions = (10, 20)  # 2D instead of 3D
                self.resolution = 0.1

        grid = MockVoxelGrid()
        with pytest.raises(AssertionError, match="must be 3D"):
            assert_voxel_grid_valid(grid)

    def test_negative_dimensions_raises(self):
        """Test that negative dimensions raise error."""

        class MockVoxelGrid:
            def __init__(self):
                self.dimensions = (10, -5, 30)
                self.resolution = 0.1

        grid = MockVoxelGrid()
        with pytest.raises(AssertionError, match="must be positive"):
            assert_voxel_grid_valid(grid)

    def test_non_positive_resolution_raises(self):
        """Test that non-positive resolution raises error."""

        class MockVoxelGrid:
            def __init__(self):
                self.dimensions = (10, 20, 30)
                self.resolution = 0.0

        grid = MockVoxelGrid()
        with pytest.raises(AssertionError, match="Resolution must be positive"):
            assert_voxel_grid_valid(grid)


class TestCreateTestPoints:
    """Tests for create_test_points function."""

    def test_default_parameters(self):
        """Test with default parameters."""
        points = create_test_points()
        assert points.shape == (10, 3)
        assert np.all(points >= 0.0)
        assert np.all(points <= 10.0)

    def test_custom_number_of_points(self):
        """Test with custom number of points."""
        points = create_test_points(n_points=5)
        assert points.shape == (5, 3)

    def test_custom_bounds(self):
        """Test with custom bounds."""
        points = create_test_points(bounds=(5.0, 15.0))
        assert np.all(points >= 5.0)
        assert np.all(points <= 15.0)

    def test_all_points_are_3d(self):
        """Test that all points are 3D."""
        points = create_test_points(n_points=20)
        assert points.shape[1] == 3


class TestCreateTestSignalArray:
    """Tests for create_test_signal_array function."""

    def test_default_parameters(self):
        """Test with default parameters."""
        signal = create_test_signal_array((5, 10, 15))
        assert signal.shape == (5, 10, 15)
        assert np.all(signal >= 0.0)
        assert np.all(signal <= 100.0)

    def test_custom_value_range(self):
        """Test with custom value range."""
        signal = create_test_signal_array((3, 3, 3), value_range=(10.0, 20.0))
        assert np.all(signal >= 10.0)
        assert np.all(signal <= 20.0)

    def test_different_shapes(self):
        """Test with different shapes."""
        shapes = [(1, 1, 1), (10, 20, 30), (5, 5, 5)]
        for shape in shapes:
            signal = create_test_signal_array(shape)
            assert signal.shape == shape


class TestAssertSignalValid:
    """Tests for assert_signal_valid function."""

    def test_valid_signal(self):
        """Test that valid signal passes."""
        signal = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        assert_signal_valid(signal)

    def test_none_signal_raises(self):
        """Test that None signal raises error."""
        with pytest.raises(AssertionError, match="Signal is None"):
            assert_signal_valid(None)

    def test_non_array_raises(self):
        """Test that non-array raises error."""
        with pytest.raises(AssertionError, match="must be a numpy array"):
            assert_signal_valid([1, 2, 3])

    def test_correct_shape_passes(self):
        """Test that signal with correct shape passes."""
        signal = np.ones((5, 10, 15))
        assert_signal_valid(signal, expected_shape=(5, 10, 15))

    def test_wrong_shape_raises(self):
        """Test that wrong shape raises error."""
        signal = np.ones((5, 10, 15))
        with pytest.raises(AssertionError, match="shape"):
            assert_signal_valid(signal, expected_shape=(3, 3, 3))

    def test_nan_values_raise(self):
        """Test that NaN values raise error by default."""
        signal = np.array([[[1.0, np.nan], [3.0, 4.0]]])
        with pytest.raises(AssertionError, match="NaN"):
            assert_signal_valid(signal)

    def test_nan_allowed(self):
        """Test that NaN values are allowed when specified."""
        signal = np.array([[[1.0, np.nan], [3.0, 4.0]]])
        assert_signal_valid(signal, allow_nan=True)

    def test_inf_values_raise(self):
        """Test that Inf values raise error."""
        signal = np.array([[[1.0, np.inf], [3.0, 4.0]]])
        with pytest.raises(AssertionError, match="Inf"):
            assert_signal_valid(signal)

    def test_inf_values_raise_even_when_nan_allowed(self):
        """Test that Inf values raise error even when NaN is allowed."""
        signal = np.array([[[1.0, np.inf], [3.0, 4.0]]])
        with pytest.raises(AssertionError, match="Inf"):
            assert_signal_valid(signal, allow_nan=True)
