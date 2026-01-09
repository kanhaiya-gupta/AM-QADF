"""
Tests for custom assertion functions.

Ensures that assertion utilities work correctly.
"""

import pytest
import numpy as np
from tests.utils.assertions import (
    assert_voxel_coordinates_valid,
    assert_interpolation_result_valid,
    assert_fusion_result_valid,
    assert_quality_metric_valid,
)


class TestAssertVoxelCoordinatesValid:
    """Tests for assert_voxel_coordinates_valid function."""

    def test_valid_coordinates(self):
        """Test that valid coordinates pass."""
        coords = np.array([[0, 0, 0], [5, 10, 15], [9, 19, 29]])
        grid_dimensions = (10, 20, 30)
        assert_voxel_coordinates_valid(coords, grid_dimensions)

    def test_coordinates_not_3d_raises(self):
        """Test that non-3D coordinates raise error."""
        coords = np.array([[0, 0], [5, 10]])  # 2D
        grid_dimensions = (10, 20, 30)
        with pytest.raises(AssertionError, match="must be 3D"):
            assert_voxel_coordinates_valid(coords, grid_dimensions)

    def test_negative_coordinates_raise(self):
        """Test that negative coordinates raise error."""
        coords = np.array([[0, 0, 0], [-1, 10, 15]])
        grid_dimensions = (10, 20, 30)
        with pytest.raises(AssertionError, match="non-negative"):
            assert_voxel_coordinates_valid(coords, grid_dimensions)

    def test_x_out_of_bounds_raises(self):
        """Test that X coordinates out of bounds raise error."""
        coords = np.array([[0, 0, 0], [10, 10, 15]])  # x=10 >= 10
        grid_dimensions = (10, 20, 30)
        with pytest.raises(AssertionError, match="X coordinates out of bounds"):
            assert_voxel_coordinates_valid(coords, grid_dimensions)

    def test_y_out_of_bounds_raises(self):
        """Test that Y coordinates out of bounds raise error."""
        coords = np.array([[0, 0, 0], [5, 20, 15]])  # y=20 >= 20
        grid_dimensions = (10, 20, 30)
        with pytest.raises(AssertionError, match="Y coordinates out of bounds"):
            assert_voxel_coordinates_valid(coords, grid_dimensions)

    def test_z_out_of_bounds_raises(self):
        """Test that Z coordinates out of bounds raise error."""
        coords = np.array([[0, 0, 0], [5, 10, 30]])  # z=30 >= 30
        grid_dimensions = (10, 20, 30)
        with pytest.raises(AssertionError, match="Z coordinates out of bounds"):
            assert_voxel_coordinates_valid(coords, grid_dimensions)


class TestAssertInterpolationResultValid:
    """Tests for assert_interpolation_result_valid function."""

    def test_valid_result(self):
        """Test that valid interpolation result passes."""
        result = np.ones((10, 20, 30))
        expected_shape = (10, 20, 30)
        assert_interpolation_result_valid(result, expected_shape)

    def test_wrong_shape_raises(self):
        """Test that wrong shape raises error."""
        result = np.ones((10, 20, 30))
        expected_shape = (5, 10, 15)
        with pytest.raises(AssertionError, match="shape"):
            assert_interpolation_result_valid(result, expected_shape)

    def test_nan_values_raise(self):
        """Test that NaN values raise error by default."""
        result = np.array([[[1.0, np.nan], [3.0, 4.0]]])
        expected_shape = (1, 2, 2)
        with pytest.raises(AssertionError, match="NaN"):
            assert_interpolation_result_valid(result, expected_shape)

    def test_nan_allowed(self):
        """Test that NaN values are allowed when specified."""
        result = np.array([[[1.0, np.nan], [3.0, 4.0]]])
        expected_shape = (1, 2, 2)
        assert_interpolation_result_valid(result, expected_shape, allow_nan=True)

    def test_inf_values_raise(self):
        """Test that Inf values raise error."""
        result = np.array([[[1.0, np.inf], [3.0, 4.0]]])
        expected_shape = (1, 2, 2)
        with pytest.raises(AssertionError, match="Inf"):
            assert_interpolation_result_valid(result, expected_shape)

    def test_inf_values_raise_even_when_nan_allowed(self):
        """Test that Inf values raise error even when NaN is allowed."""
        result = np.array([[[1.0, np.inf], [3.0, 4.0]]])
        expected_shape = (1, 2, 2)
        with pytest.raises(AssertionError, match="Inf"):
            assert_interpolation_result_valid(result, expected_shape, allow_nan=True)


class TestAssertFusionResultValid:
    """Tests for assert_fusion_result_valid function."""

    def test_valid_fusion_result(self):
        """Test that valid fusion result passes."""
        source_signals = {
            "signal1": np.ones((10, 20, 30)),
            "signal2": np.ones((10, 20, 30)),
        }
        fused = np.ones((10, 20, 30))
        assert_fusion_result_valid(fused, source_signals)

    def test_shape_mismatch_raises(self):
        """Test that shape mismatch raises error."""
        source_signals = {
            "signal1": np.ones((10, 20, 30)),
            "signal2": np.ones((10, 20, 30)),
        }
        fused = np.ones((5, 10, 15))  # Different shape
        with pytest.raises(AssertionError, match="shape"):
            assert_fusion_result_valid(fused, source_signals)

    def test_nan_values_raise(self):
        """Test that NaN values raise error by default."""
        source_signals = {"signal1": np.ones((2, 2, 2)), "signal2": np.ones((2, 2, 2))}
        fused = np.array([[[1.0, np.nan], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        with pytest.raises(AssertionError, match="NaN"):
            assert_fusion_result_valid(fused, source_signals)

    def test_nan_allowed(self):
        """Test that NaN values are allowed when specified."""
        source_signals = {"signal1": np.ones((2, 2, 2)), "signal2": np.ones((2, 2, 2))}
        fused = np.array([[[1.0, np.nan], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        assert_fusion_result_valid(fused, source_signals, allow_nan=True)

    def test_inf_values_raise(self):
        """Test that Inf values raise error."""
        source_signals = {"signal1": np.ones((2, 2, 2)), "signal2": np.ones((2, 2, 2))}
        fused = np.array([[[1.0, np.inf], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        with pytest.raises(AssertionError, match="Inf"):
            assert_fusion_result_valid(fused, source_signals)

    def test_empty_source_signals_raises(self):
        """Test that empty source signals raise error."""
        source_signals = {}
        fused = np.ones((10, 20, 30))
        with pytest.raises(IndexError):
            assert_fusion_result_valid(fused, source_signals)


class TestAssertQualityMetricValid:
    """Tests for assert_quality_metric_valid function."""

    def test_valid_metric_in_range(self):
        """Test that valid metric in range passes."""
        assert_quality_metric_valid(0.5, "test_metric")
        assert_quality_metric_valid(0.0, "test_metric")
        assert_quality_metric_valid(1.0, "test_metric")

    def test_metric_below_range_raises(self):
        """Test that metric below range raises error."""
        with pytest.raises(AssertionError, match="not in range"):
            assert_quality_metric_valid(-0.1, "test_metric")

    def test_metric_above_range_raises(self):
        """Test that metric above range raises error."""
        with pytest.raises(AssertionError, match="not in range"):
            assert_quality_metric_valid(1.1, "test_metric")

    def test_custom_range(self):
        """Test that custom range works."""
        assert_quality_metric_valid(5.0, "test_metric", expected_range=(0.0, 10.0))
        assert_quality_metric_valid(0.0, "test_metric", expected_range=(0.0, 10.0))
        assert_quality_metric_valid(10.0, "test_metric", expected_range=(0.0, 10.0))

        with pytest.raises(AssertionError):
            assert_quality_metric_valid(11.0, "test_metric", expected_range=(0.0, 10.0))

    def test_nan_metric_raises(self):
        """Test that NaN metric raises error."""
        # Note: Range check happens first, so error message is about range, not NaN
        with pytest.raises(AssertionError, match="not in range"):
            assert_quality_metric_valid(np.nan, "test_metric")

    def test_inf_metric_raises(self):
        """Test that Inf metric raises error."""
        # Note: Range check happens first, so error message is about range, not Inf
        with pytest.raises(AssertionError, match="not in range"):
            assert_quality_metric_valid(np.inf, "test_metric")

    def test_negative_inf_metric_raises(self):
        """Test that negative Inf metric raises error."""
        # Note: Range check happens first, so error message is about range, not Inf
        with pytest.raises(AssertionError, match="not in range"):
            assert_quality_metric_valid(-np.inf, "test_metric")

    def test_metric_name_in_error_message(self):
        """Test that metric name appears in error message."""
        with pytest.raises(AssertionError, match="custom_metric"):
            assert_quality_metric_valid(np.nan, "custom_metric")
