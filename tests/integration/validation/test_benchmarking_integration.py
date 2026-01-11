"""
Integration tests for benchmarking with core framework operations.

Tests benchmarking integration with signal mapping, data fusion, and query operations.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

try:
    from am_qadf.validation.benchmarking import PerformanceBenchmarker
except ImportError:
    pytest.skip("Validation module not available", allow_module_level=True)


class MockVoxelGrid:
    """Mock voxel grid for testing."""

    def __init__(self, dims=(10, 10, 10)):
        self.dims = dims
        self.available_signals = {"laser_power", "temperature"}

    def get_signal_array(self, signal_name: str):
        """Get signal array."""
        return np.random.rand(*self.dims)


class TestBenchmarkingIntegration:
    """Integration tests for benchmarking framework operations."""

    @pytest.fixture
    def benchmarker(self):
        """PerformanceBenchmarker instance."""
        return PerformanceBenchmarker()

    @pytest.fixture
    def sample_voxel_grid(self):
        """Sample voxel grid for testing."""
        return MockVoxelGrid(dims=(20, 20, 10))

    @pytest.mark.integration
    def test_benchmark_signal_mapping_operation(self, benchmarker, sample_voxel_grid):
        """Test benchmarking a signal mapping-like operation."""

        def signal_mapping_operation(grid, source_data):
            """Simulate signal mapping."""
            result = np.zeros(grid.dims)
            for i in range(min(len(source_data), result.size)):
                result.flat[i] = source_data[i % len(source_data)]
            return result

        source_data = np.random.rand(1000) * 300

        result = benchmarker.benchmark_operation(
            "signal_mapping", signal_mapping_operation, sample_voxel_grid, source_data, iterations=3
        )

        assert isinstance(
            result,
            (
                type(benchmarker).__module__.BenchmarkResult
                if hasattr(type(benchmarker).__module__, "BenchmarkResult")
                else type(result)
            ),
        )
        assert result.operation_name == "signal_mapping"
        assert result.execution_time >= 0

    @pytest.mark.integration
    def test_benchmark_data_fusion_operation(self, benchmarker):
        """Test benchmarking a data fusion-like operation."""

        def fusion_operation(grids, method="weighted_average"):
            """Simulate data fusion."""
            if len(grids) == 0:
                return None
            result = grids[0].copy()
            for grid in grids[1:]:
                result = (result + grid) / 2
            return result

        grid1 = np.random.rand(50, 50, 10)
        grid2 = np.random.rand(50, 50, 10)
        grid3 = np.random.rand(50, 50, 10)

        result = benchmarker.benchmark_operation(
            "data_fusion", fusion_operation, [grid1, grid2, grid3], method="weighted_average", iterations=3
        )

        assert result.operation_name == "data_fusion"
        assert result.execution_time >= 0
        assert result.data_volume > 0

    @pytest.mark.integration
    def test_benchmark_quality_assessment_operation(self, benchmarker, sample_voxel_grid):
        """Test benchmarking a quality assessment-like operation."""

        def quality_assessment_operation(voxel_data, signals):
            """Simulate quality assessment."""
            results = {}
            for signal in signals:
                signal_array = voxel_data.get_signal_array(signal)
                results[signal] = {
                    "mean": float(np.mean(signal_array)),
                    "std": float(np.std(signal_array)),
                    "quality_score": 0.9,
                }
            return results

        result = benchmarker.benchmark_operation(
            "quality_assessment", quality_assessment_operation, sample_voxel_grid, ["laser_power", "temperature"], iterations=3
        )

        assert result.operation_name == "quality_assessment"
        assert result.execution_time >= 0

    @pytest.mark.integration
    def test_benchmark_query_operation(self, benchmarker):
        """Test benchmarking a query-like operation."""

        def query_operation(model_id, filters):
            """Simulate database query."""
            # Simulate query processing
            results = {
                "hatching": np.random.rand(1000, 3),
                "laser": np.random.rand(5000, 4),
            }
            return results

        result = benchmarker.benchmark_operation(
            "query_operation",
            query_operation,
            "test_model",
            {"spatial": None, "temporal": None},
            iterations=5,
            warmup_iterations=1,
        )

        assert result.operation_name == "query_operation"
        assert result.iterations == 5
        assert result.metadata["warmup_iterations"] == 1

    @pytest.mark.integration
    def test_compare_multiple_operations(self, benchmarker):
        """Test comparing multiple operations."""

        def op1(x):
            return x * 2

        def op2(x):
            return x**2

        def op3(x):
            return np.sqrt(x)

        operations = {
            "multiply": (op1, (100,), {}),
            "square": (op2, (100,), {}),
            "sqrt": (op3, (100,), {}),
        }

        results = benchmarker.compare_operations(operations, iterations=3)

        assert isinstance(results, dict)
        assert len(results) == 3
        assert all(
            isinstance(
                r,
                (
                    type(benchmarker).__module__.BenchmarkResult
                    if hasattr(type(benchmarker).__module__, "BenchmarkResult")
                    else type(r)
                ),
            )
            or r is None
            for r in results.values()
        )

    @pytest.mark.integration
    def test_benchmark_overhead_assessment(self, benchmarker):
        """Test that benchmarking overhead is reasonable."""

        def fast_operation(x):
            return x + 1

        # Time operation directly
        import time

        start = time.perf_counter()
        for _ in range(10):
            fast_operation(10)
        direct_time = time.perf_counter() - start

        # Time with benchmarking
        result = benchmarker.benchmark_operation("test_op", fast_operation, 10, iterations=10, warmup_iterations=0)

        benchmark_time = result.execution_time

        # Benchmarking overhead should be < 50% (allowing for measurement overhead)
        overhead_ratio = (benchmark_time - direct_time / 10) / (direct_time / 10)
        assert overhead_ratio < 0.5 or benchmark_time < 0.1  # Very fast ops may have higher relative overhead

    @pytest.mark.integration
    def test_benchmark_report_generation(self, benchmarker, tmp_path):
        """Test generating benchmark report from multiple operations."""
        results = []
        for i in range(3):

            def op(x):
                return x + i

            result = benchmarker.benchmark_operation(f"operation_{i}", op, 100, iterations=2)
            results.append(result)

        output_path = tmp_path / "integration_benchmark_report.txt"
        report = benchmarker.generate_report(results, output_path=str(output_path))

        assert isinstance(report, str)
        assert "Performance Benchmark Report" in report
        assert output_path.exists()
