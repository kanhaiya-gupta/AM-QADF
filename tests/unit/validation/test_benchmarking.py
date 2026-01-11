"""
Unit tests for benchmarking module.

Tests for BenchmarkResult, PerformanceBenchmarker, and benchmark decorator.
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

try:
    from am_qadf.validation.benchmarking import (
        BenchmarkResult,
        PerformanceBenchmarker,
        benchmark,
    )
except ImportError:
    pytest.skip("Validation module not available", allow_module_level=True)


class TestBenchmarkResult:
    """Test suite for BenchmarkResult dataclass."""

    @pytest.mark.unit
    def test_benchmark_result_creation(self):
        """Test creating BenchmarkResult with all fields."""
        result = BenchmarkResult(
            operation_name="test_operation",
            execution_time=1.5,
            memory_usage=100.5,
            data_volume=1000,
            throughput=0.67,
            iterations=5,
            metadata={"test": "value"},
        )

        assert result.operation_name == "test_operation"
        assert result.execution_time == 1.5
        assert result.memory_usage == 100.5
        assert result.data_volume == 1000
        assert result.throughput == 0.67
        assert result.iterations == 5
        assert result.metadata == {"test": "value"}
        assert isinstance(result.timestamp, datetime)

    @pytest.mark.unit
    def test_benchmark_result_defaults(self):
        """Test BenchmarkResult with default values."""
        result = BenchmarkResult(operation_name="test", execution_time=1.0, memory_usage=50.0, data_volume=500, throughput=1.0)

        assert result.iterations == 1
        assert result.metadata == {}
        assert isinstance(result.timestamp, datetime)

    @pytest.mark.unit
    def test_benchmark_result_to_dict(self):
        """Test converting BenchmarkResult to dictionary."""
        result = BenchmarkResult(
            operation_name="test_operation",
            execution_time=1.5,
            memory_usage=100.5,
            data_volume=1000,
            throughput=0.67,
            iterations=5,
            metadata={"key": "value"},
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["operation_name"] == "test_operation"
        assert result_dict["execution_time"] == 1.5
        assert result_dict["memory_usage"] == 100.5
        assert result_dict["data_volume"] == 1000
        assert result_dict["throughput"] == 0.67
        assert result_dict["iterations"] == 5
        assert result_dict["metadata"] == {"key": "value"}
        assert "timestamp" in result_dict


class TestPerformanceBenchmarker:
    """Test suite for PerformanceBenchmarker class."""

    @pytest.fixture
    def benchmarker(self):
        """Create a PerformanceBenchmarker instance."""
        return PerformanceBenchmarker()

    @pytest.fixture
    def sample_operation(self):
        """Sample operation for benchmarking."""

        def operation(data):
            time.sleep(0.01)  # Simulate work
            return data * 2

        return operation

    @pytest.fixture
    def fast_operation(self):
        """Fast operation for benchmarking."""

        def operation(x):
            return x + 1

        return operation

    @pytest.mark.unit
    def test_benchmarker_creation(self, benchmarker):
        """Test creating PerformanceBenchmarker."""
        assert benchmarker is not None

    @pytest.mark.unit
    def test_benchmark_operation_single_iteration(self, benchmarker, fast_operation):
        """Test benchmarking operation with single iteration."""
        result = benchmarker.benchmark_operation("test_operation", fast_operation, 10, iterations=1)

        assert isinstance(result, BenchmarkResult)
        assert result.operation_name == "test_operation"
        assert result.execution_time >= 0
        assert result.iterations == 1
        assert result.throughput >= 0

    @pytest.mark.unit
    def test_benchmark_operation_multiple_iterations(self, benchmarker, fast_operation):
        """Test benchmarking operation with multiple iterations."""
        result = benchmarker.benchmark_operation("test_operation", fast_operation, 10, iterations=5, warmup_iterations=2)

        assert isinstance(result, BenchmarkResult)
        assert result.iterations == 5
        assert result.metadata["warmup_iterations"] == 2
        assert "min_time" in result.metadata
        assert "max_time" in result.metadata
        assert "std_time" in result.metadata

    @pytest.mark.unit
    def test_benchmark_operation_with_exception(self, benchmarker):
        """Test benchmarking operation that raises exception."""

        def failing_operation():
            raise ValueError("Test error")

        result = benchmarker.benchmark_operation("failing_operation", failing_operation, iterations=3)

        assert isinstance(result, BenchmarkResult)
        assert result.execution_time == float("inf")  # Should handle exception

    @pytest.mark.unit
    def test_benchmark_operation_with_numpy_array(self, benchmarker):
        """Test benchmarking operation with numpy array."""

        def array_operation(arr):
            return arr * 2

        test_array = np.random.rand(100, 100)
        result = benchmarker.benchmark_operation("array_operation", array_operation, test_array, iterations=3)

        assert isinstance(result, BenchmarkResult)
        assert result.data_volume > 0  # Should estimate array size

    @pytest.mark.unit
    def test_benchmark_operation_estimate_data_volume(self, benchmarker):
        """Test data volume estimation."""
        # Test with numpy array
        arr = np.random.rand(50, 50)

        def op1(x):
            return x * 2

        result1 = benchmarker.benchmark_operation("op1", op1, arr, iterations=1)
        assert result1.data_volume > 0

        # Test with list
        def op2(x):
            return x * 2

        result2 = benchmarker.benchmark_operation("op2", op2, [1, 2, 3], iterations=1)
        assert result2.data_volume >= 3

        # Test with dict
        def op3(x):
            return x

        test_dict = {"a": np.array([1, 2, 3]), "b": [4, 5, 6]}
        result3 = benchmarker.benchmark_operation("op3", op3, test_dict, iterations=1)
        assert result3.data_volume > 0

    @pytest.mark.unit
    def test_compare_operations(self, benchmarker):
        """Test comparing multiple operations."""

        def op1(x):
            return x + 1

        def op2(x):
            return x * 2

        operations = {
            "operation_1": (op1, (10,), {}),
            "operation_2": (op2, (10,), {}),
        }

        results = benchmarker.compare_operations(operations, iterations=3)

        assert isinstance(results, dict)
        assert "operation_1" in results
        assert "operation_2" in results
        assert isinstance(results["operation_1"], BenchmarkResult)
        assert isinstance(results["operation_2"], BenchmarkResult)

    @pytest.mark.unit
    def test_compare_operations_with_failure(self, benchmarker):
        """Test comparing operations when one fails."""

        def op1(x):
            return x + 1

        def op2(x):
            raise ValueError("Error")

        operations = {
            "operation_1": (op1, (10,), {}),
            "operation_2": (op2, (10,), {}),
        }

        results = benchmarker.compare_operations(operations, iterations=1)

        assert "operation_1" in results
        assert results["operation_1"] is not None
        assert "operation_2" in results
        # operation_2 may be None or have inf execution_time

    @pytest.mark.unit
    def test_generate_report(self, benchmarker, tmp_path):
        """Test generating benchmark report."""
        results = [
            BenchmarkResult(
                operation_name="op1",
                execution_time=1.0,
                memory_usage=50.0,
                data_volume=1000,
                throughput=1.0,
                iterations=5,
                metadata={"min_time": 0.9, "max_time": 1.1, "std_time": 0.05},
            ),
            BenchmarkResult(
                operation_name="op2",
                execution_time=2.0,
                memory_usage=100.0,
                data_volume=2000,
                throughput=0.5,
                iterations=5,
                metadata={"min_time": 1.9, "max_time": 2.1, "std_time": 0.05},
            ),
        ]

        output_path = tmp_path / "benchmark_report.txt"
        report = benchmarker.generate_report(results, output_path=str(output_path))

        assert isinstance(report, str)
        assert "Performance Benchmark Report" in report
        assert "op1" in report
        assert "op2" in report
        assert output_path.exists()

    @pytest.mark.unit
    def test_generate_report_empty_results(self, benchmarker):
        """Test generating report with empty results."""
        report = benchmarker.generate_report([])

        assert isinstance(report, str)
        assert "Performance Benchmark Report" in report

    @pytest.mark.unit
    @patch("am_qadf.validation.benchmarking.PSUTIL_AVAILABLE", True)
    def test_benchmark_with_psutil(self, benchmarker, fast_operation):
        """Test benchmarking with psutil available."""
        with patch("am_qadf.validation.benchmarking.psutil") as mock_psutil:
            mock_process = MagicMock()
            mock_process.memory_info.return_value.rss = 100 * 1024 * 1024  # 100 MB
            mock_psutil.Process.return_value = mock_process

            benchmarker_with_psutil = PerformanceBenchmarker()
            result = benchmarker_with_psutil.benchmark_operation("test", fast_operation, 10, iterations=1)

            assert isinstance(result, BenchmarkResult)
            assert result.memory_usage >= 0

    @pytest.mark.unit
    @patch("am_qadf.validation.benchmarking.PSUTIL_AVAILABLE", False)
    def test_benchmark_without_psutil(self, benchmarker, fast_operation):
        """Test benchmarking without psutil (fallback to tracemalloc)."""
        result = benchmarker.benchmark_operation("test", fast_operation, 10, iterations=1)

        assert isinstance(result, BenchmarkResult)
        assert result.memory_usage >= 0  # Should still work with fallback


class TestBenchmarkDecorator:
    """Test suite for benchmark decorator."""

    @pytest.mark.unit
    def test_benchmark_decorator_default(self):
        """Test @benchmark decorator with default parameters."""

        @benchmark()
        def test_function(x):
            return x * 2

        result = test_function(10)

        assert result == 20
        assert hasattr(test_function, "_benchmark_results")
        assert len(test_function._benchmark_results) > 0
        assert isinstance(test_function._benchmark_results[0], BenchmarkResult)

    @pytest.mark.unit
    def test_benchmark_decorator_custom_params(self):
        """Test @benchmark decorator with custom parameters."""

        @benchmark(iterations=5, warmup_iterations=2, operation_name="custom_op")
        def test_function(x):
            return x + 1

        result = test_function(10)

        assert result == 11
        assert len(test_function._benchmark_results) == 1
        benchmark_result = test_function._benchmark_results[0]
        assert benchmark_result.iterations == 5
        assert benchmark_result.metadata["warmup_iterations"] == 2

    @pytest.mark.unit
    def test_benchmark_decorator_preserves_metadata(self):
        """Test that decorator preserves function metadata."""

        @benchmark()
        def test_function(x):
            """Test function docstring."""
            return x

        assert test_function.__name__ == "test_function"
        assert "Test function docstring" in test_function.__doc__

    @pytest.mark.unit
    def test_benchmark_decorator_with_exception(self):
        """Test @benchmark decorator with function that raises exception."""

        @benchmark()
        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            failing_function()

        # Benchmark result should still be recorded
        assert hasattr(failing_function, "_benchmark_results")
