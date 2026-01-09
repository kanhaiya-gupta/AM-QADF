"""
Unit tests for Spark execution.

Tests for Spark-based interpolation methods (with mocking).
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from am_qadf.signal_mapping.execution.spark import (
    SparkInterpolationMethod,
    SparkNearestNeighbor,
    SparkLinearInterpolation,
    SparkIDWInterpolation,
    SparkGaussianKDE,
    PYSPARK_AVAILABLE,
    interpolate_to_voxels_spark,
)
from am_qadf.voxelization.voxel_grid import VoxelGrid


class TestSparkInterpolationMethod:
    """Test suite for SparkInterpolationMethod base class."""

    @pytest.mark.unit
    def test_spark_interpolation_method_abstract(self):
        """Test that SparkInterpolationMethod is abstract."""
        with pytest.raises(TypeError):
            SparkInterpolationMethod()

    @pytest.mark.unit
    @patch("am_qadf.signal_mapping.execution.spark.PYSPARK_AVAILABLE", True)
    def test_calculate_voxel_indices_udf(self):
        """Test calculating voxel indices UDF."""

        class TestSparkMethod(SparkInterpolationMethod):
            def interpolate_spark(self, spark, points_df, voxel_grid_config):
                pass

        method = TestSparkMethod()
        voxel_grid_config = {
            "bbox_min": (0.0, 0.0, 0.0),
            "bbox_max": (10.0, 10.0, 10.0),
            "resolution": 1.0,
        }

        udf = method._calculate_voxel_indices_udf(voxel_grid_config)

        assert callable(udf)


class TestSparkNearestNeighbor:
    """Test suite for SparkNearestNeighbor class."""

    @pytest.mark.unit
    def test_spark_nearest_neighbor_creation(self):
        """Test creating SparkNearestNeighbor."""
        method = SparkNearestNeighbor()

        assert method is not None
        assert isinstance(method, SparkInterpolationMethod)

    @pytest.mark.unit
    @patch("am_qadf.signal_mapping.execution.spark.PYSPARK_AVAILABLE", True)
    def test_interpolate_spark_pyspark_not_available(self):
        """Test that interpolate_spark raises error if PySpark not available."""
        method = SparkNearestNeighbor()
        mock_spark = Mock()
        mock_df = Mock()
        voxel_grid_config = {
            "bbox_min": (0.0, 0.0, 0.0),
            "bbox_max": (10.0, 10.0, 10.0),
            "resolution": 1.0,
            "aggregation": "mean",
        }

        with patch("am_qadf.signal_mapping.execution.spark.PYSPARK_AVAILABLE", False):
            with pytest.raises(ImportError, match="PySpark is required"):
                method.interpolate_spark(mock_spark, mock_df, voxel_grid_config)


class TestSparkLinearInterpolation:
    """Test suite for SparkLinearInterpolation class."""

    @pytest.mark.unit
    def test_spark_linear_interpolation_creation(self):
        """Test creating SparkLinearInterpolation."""
        method = SparkLinearInterpolation(k_neighbors=4, radius=2.0)

        assert method.k_neighbors == 4
        assert method.radius == 2.0

    @pytest.mark.unit
    def test_spark_linear_interpolation_default(self):
        """Test creating SparkLinearInterpolation with defaults."""
        method = SparkLinearInterpolation()

        assert method.k_neighbors == 8
        assert method.radius is None


class TestSparkIDWInterpolation:
    """Test suite for SparkIDWInterpolation class."""

    @pytest.mark.unit
    def test_spark_idw_interpolation_creation(self):
        """Test creating SparkIDWInterpolation."""
        method = SparkIDWInterpolation(power=3.0, k_neighbors=6, radius=2.0)

        assert method.power == 3.0
        assert method.k_neighbors == 6
        assert method.radius == 2.0

    @pytest.mark.unit
    def test_spark_idw_interpolation_default(self):
        """Test creating SparkIDWInterpolation with defaults."""
        method = SparkIDWInterpolation()

        assert method.power == 2.0
        assert method.k_neighbors == 8
        assert method.radius is None


class TestSparkGaussianKDE:
    """Test suite for SparkGaussianKDE class."""

    @pytest.mark.unit
    def test_spark_gaussian_kde_creation(self):
        """Test creating SparkGaussianKDE."""
        method = SparkGaussianKDE(bandwidth=2.0, adaptive=True)

        assert method.bandwidth == 2.0
        assert method.adaptive is True

    @pytest.mark.unit
    def test_spark_gaussian_kde_default(self):
        """Test creating SparkGaussianKDE with defaults."""
        method = SparkGaussianKDE()

        assert method.bandwidth is None
        assert method.adaptive is False


class TestInterpolateToVoxelsSpark:
    """Test suite for interpolate_to_voxels_spark function."""

    @pytest.mark.unit
    @patch("am_qadf.signal_mapping.execution.spark.PYSPARK_AVAILABLE", False)
    def test_interpolate_to_voxels_spark_not_available(self):
        """Test that function handles PySpark not available."""
        mock_spark = Mock()
        points = np.array([[5.0, 5.0, 5.0]])
        signals = {"power": np.array([200.0])}
        voxel_grid_config = {
            "bbox_min": (0.0, 0.0, 0.0),
            "bbox_max": (10.0, 10.0, 10.0),
            "resolution": 1.0,
            "aggregation": "mean",
        }

        with pytest.raises(ImportError):
            interpolate_to_voxels_spark(mock_spark, points, signals, voxel_grid_config, method="nearest")

    @pytest.mark.unit
    def test_pyspark_available_flag(self):
        """Test that PYSPARK_AVAILABLE flag is set correctly."""
        # This flag depends on whether pyspark is installed
        # We just verify it's a boolean
        assert isinstance(PYSPARK_AVAILABLE, bool)


class TestSparkMethodRegistry:
    """Test suite for Spark method registry."""

    @pytest.mark.unit
    def test_spark_methods_importable(self):
        """Test that all Spark interpolation methods are importable."""
        from am_qadf.signal_mapping.execution.spark import SPARK_INTERPOLATION_METHODS

        assert "nearest" in SPARK_INTERPOLATION_METHODS
        assert "linear" in SPARK_INTERPOLATION_METHODS
        assert "idw" in SPARK_INTERPOLATION_METHODS
        assert "gaussian_kde" in SPARK_INTERPOLATION_METHODS
