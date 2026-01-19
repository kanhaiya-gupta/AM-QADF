"""
Integration tests for Spark-based RBF interpolation.

Tests RBF with Spark backend, validates distributed approach, and compares performance.
"""

import pytest
import numpy as np
from am_qadf.voxelization.voxel_grid import VoxelGrid
from am_qadf.signal_mapping.execution.sequential import interpolate_to_voxels

# Timeout for Spark tests (10 minutes max per test)
pytestmark = pytest.mark.timeout(600)


@pytest.fixture(scope="module")
def spark_session():
    """Create Spark session for testing."""
    try:
        from am_qadf.signal_mapping.utils.spark_utils import create_spark_session

        spark = create_spark_session(app_name="RBF_Spark_Test", master="local[2]")
        if spark is None:
            pytest.skip("PySpark not available or Spark session creation failed")
        yield spark
        spark.stop()
    except ImportError:
        pytest.skip("PySpark not available")


@pytest.fixture
def sample_voxel_grid():
    """Create a test voxel grid."""
    return VoxelGrid(
        bbox_min=(0.0, 0.0, 0.0),
        bbox_max=(10.0, 10.0, 10.0),
        resolution=0.5,
        aggregation="mean",
    )


def _create_voxel_grid_copy(grid):
    """Helper to create a copy of a voxel grid."""
    return VoxelGrid(
        bbox_min=tuple(grid.bbox_min),
        bbox_max=tuple(grid.bbox_max),
        resolution=grid.resolution,
        aggregation=grid.aggregation,
    )


def create_test_data(n_points, bbox_size=10.0, seed=42):
    """Create test data for Spark testing."""
    np.random.seed(seed)
    points = np.random.rand(n_points, 3) * bbox_size
    signals = {"power": 100.0 + 50.0 * np.sin(points[:, 0] / 2.0) + np.random.randn(n_points) * 2.0}
    return points, signals


@pytest.mark.integration
@pytest.mark.spark
class TestRBFSpark:
    """Integration tests for Spark-based RBF interpolation."""

    @pytest.mark.integration
    @pytest.mark.spark
    def test_rbf_with_spark_backend_small_dataset(self, spark_session, sample_voxel_grid):
        """Test RBF with Spark backend on small dataset (should use sequential)."""
        points, signals = create_test_data(n_points=500)  # Small dataset

        result = interpolate_to_voxels(
            points=points,
            signals=signals,
            voxel_grid=_create_voxel_grid_copy(sample_voxel_grid),
            method="rbf",
            use_spark=True,
            spark_session=spark_session,
            kernel="gaussian",
            epsilon=1.0,
            smoothing=0.0,
        )

        assert result is not None
        assert len(result.voxels) > 0

    @pytest.mark.integration
    @pytest.mark.spark
    def test_rbf_with_spark_backend_large_dataset(self, spark_session, sample_voxel_grid):
        """Test RBF with Spark backend on large dataset (should use distributed)."""
        points, signals = create_test_data(n_points=15000)  # Large dataset (> threshold)

        result = interpolate_to_voxels(
            points=points,
            signals=signals,
            voxel_grid=_create_voxel_grid_copy(sample_voxel_grid),
            method="rbf",
            use_spark=True,
            spark_session=spark_session,
            kernel="gaussian",
            epsilon=1.0,
            smoothing=0.0,
        )

        assert result is not None
        assert len(result.voxels) > 0

    @pytest.mark.integration
    @pytest.mark.spark
    def test_rbf_spark_vs_sequential_accuracy(self, spark_session, sample_voxel_grid):
        """Compare Spark RBF accuracy with sequential RBF."""
        points, signals = create_test_data(n_points=500)  # Small enough for sequential

        # Sequential RBF
        result_seq = interpolate_to_voxels(
            points=points,
            signals=signals,
            voxel_grid=_create_voxel_grid_copy(sample_voxel_grid),
            method="rbf",
            use_spark=False,
            kernel="gaussian",
            epsilon=1.0,
            smoothing=0.0,
        )

        # Spark RBF (will use sequential for small dataset)
        result_spark = interpolate_to_voxels(
            points=points,
            signals=signals,
            voxel_grid=_create_voxel_grid_copy(sample_voxel_grid),
            method="rbf",
            use_spark=True,
            spark_session=spark_session,
            kernel="gaussian",
            epsilon=1.0,
            smoothing=0.0,
        )

        # Both should produce results
        assert result_seq is not None
        assert result_spark is not None
        assert len(result_seq.voxels) > 0
        assert len(result_spark.voxels) > 0

        # Results should be similar (may not be identical due to numerical precision)
        # Check that both have similar number of filled voxels
        assert abs(len(result_seq.voxels) - len(result_spark.voxels)) < len(result_seq.voxels) * 0.1

    @pytest.mark.integration
    @pytest.mark.spark
    def test_rbf_spark_performance_comparison(self, spark_session, sample_voxel_grid):
        """Compare Spark RBF performance with sequential RBF."""
        import time

        points, signals = create_test_data(n_points=15000)  # Large dataset

        # Sequential RBF
        start_time = time.time()
        result_seq = interpolate_to_voxels(
            points=points,
            signals=signals,
            voxel_grid=_create_voxel_grid_copy(sample_voxel_grid),
            method="rbf",
            use_spark=False,
            kernel="gaussian",
            epsilon=1.0,
        )
        time_seq = time.time() - start_time

        # Spark RBF
        start_time = time.time()
        result_spark = interpolate_to_voxels(
            points=points,
            signals=signals,
            voxel_grid=_create_voxel_grid_copy(sample_voxel_grid),
            method="rbf",
            use_spark=True,
            spark_session=spark_session,
            kernel="gaussian",
            epsilon=1.0,
        )
        time_spark = time.time() - start_time

        print(f"\nSequential RBF: {time_seq:.3f}s")
        print(f"Spark RBF: {time_spark:.3f}s")
        print(f"Speedup: {time_seq / time_spark:.2f}x")

        assert result_seq is not None
        assert result_spark is not None

    @pytest.mark.integration
    @pytest.mark.spark
    def test_rbf_spark_different_kernels(self, spark_session, sample_voxel_grid):
        """Test Spark RBF with different kernel types."""
        points, signals = create_test_data(n_points=1000)

        kernels = ["gaussian", "multiquadric", "thin_plate_spline"]

        for kernel in kernels:
            result = interpolate_to_voxels(
                points=points,
                signals=signals,
                voxel_grid=_create_voxel_grid_copy(sample_voxel_grid),
                method="rbf",
                use_spark=True,
                spark_session=spark_session,
                kernel=kernel,
                epsilon=1.0,
            )

            assert result is not None
            assert len(result.voxels) > 0

    @pytest.mark.integration
    @pytest.mark.spark
    def test_rbf_spark_scalability(self, spark_session, sample_voxel_grid):
        """Test Spark RBF scalability with increasing dataset sizes."""
        import time

        sizes = [1000, 5000, 10000]
        times = []

        for n_points in sizes:
            points, signals = create_test_data(n_points)

            start_time = time.time()
            result = interpolate_to_voxels(
                points=points,
                signals=signals,
                voxel_grid=_create_voxel_grid_copy(sample_voxel_grid),
                method="rbf",
                use_spark=True,
                spark_session=spark_session,
                kernel="gaussian",
                epsilon=1.0,
            )
            elapsed = time.time() - start_time

            times.append(elapsed)
            print(f"Spark RBF with {n_points} points: {elapsed:.3f}s")

            assert result is not None
            assert len(result.voxels) > 0

        # Verify all completed
        assert len(times) == len(sizes)

    @pytest.mark.integration
    @pytest.mark.spark
    def test_rbf_spark_threshold_routing(self, spark_session, sample_voxel_grid):
        """Test that Spark RBF correctly routes to sequential for small datasets."""
        # Small dataset (< 10,000) should use sequential
        points, signals = create_test_data(n_points=500)

        result = interpolate_to_voxels(
            points=points,
            signals=signals,
            voxel_grid=_create_voxel_grid_copy(sample_voxel_grid),
            method="rbf",
            use_spark=True,
            spark_session=spark_session,
            kernel="gaussian",
            epsilon=1.0,
        )

        # Should complete successfully (using sequential fallback)
        assert result is not None
        assert len(result.voxels) > 0

    @pytest.mark.integration
    @pytest.mark.spark
    def test_rbf_spark_distributed_approximation(self, spark_session, sample_voxel_grid):
        """Test Spark RBF distributed approximation for large datasets."""
        # Large dataset (> 10,000) should use distributed approximation
        points, signals = create_test_data(n_points=20000)

        result = interpolate_to_voxels(
            points=points,
            signals=signals,
            voxel_grid=_create_voxel_grid_copy(sample_voxel_grid),
            method="rbf",
            use_spark=True,
            spark_session=spark_session,
            kernel="gaussian",
            epsilon=1.0,
        )

        # Should complete successfully (using distributed approximation)
        assert result is not None
        assert len(result.voxels) > 0

    @pytest.mark.integration
    @pytest.mark.spark
    def test_rbf_spark_error_handling(self, spark_session, sample_voxel_grid):
        """Test Spark RBF error handling."""
        # Test with empty points
        points = np.array([]).reshape(0, 3)
        signals = {}

        # Empty points should raise an error (Spark can't create DataFrame from empty data)
        # This is expected behavior - empty datasets should be handled before calling Spark
        with pytest.raises(ValueError, match="empty"):
            result = interpolate_to_voxels(
                points=points,
                signals=signals,
                voxel_grid=_create_voxel_grid_copy(sample_voxel_grid),
                method="rbf",
                use_spark=True,
                spark_session=spark_session,
            )

    @pytest.mark.integration
    @pytest.mark.spark
    def test_rbf_spark_multiple_signals(self, spark_session, sample_voxel_grid):
        """Test Spark RBF with multiple signal types."""
        # Use unique points to avoid singular matrix issues
        np.random.seed(42)
        n_points = 1000
        points = np.random.rand(n_points, 3) * 10.0
        signals = {
            "power": 100.0 + 50.0 * np.sin(points[:, 0] / 2.0) + np.random.randn(n_points) * 2.0,
            "speed": 500.0 + 100.0 * np.cos(points[:, 1] / 2.0) + np.random.randn(n_points) * 5.0,
        }

        result = interpolate_to_voxels(
            points=points,
            signals=signals,
            voxel_grid=_create_voxel_grid_copy(sample_voxel_grid),
            method="rbf",
            use_spark=True,
            spark_session=spark_session,
            kernel="gaussian",
            epsilon=1.0,
        )

        assert result is not None
        assert len(result.voxels) > 0
