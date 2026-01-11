"""
Performance Benchmarking

Benchmarking utilities for measuring framework operation performance.
Provides timing, memory usage, and throughput metrics.
"""

import time
import tracemalloc
import functools
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
import logging
from datetime import datetime
import os

# Optional dependencies
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("psutil not available, memory tracking will be limited")

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a benchmarking operation."""

    operation_name: str
    execution_time: float  # seconds
    memory_usage: float  # MB
    data_volume: int  # bytes or number of elements
    throughput: float  # operations per second
    iterations: int = 1
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "operation_name": self.operation_name,
            "execution_time": self.execution_time,
            "memory_usage": self.memory_usage,
            "data_volume": self.data_volume,
            "throughput": self.throughput,
            "iterations": self.iterations,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class PerformanceBenchmarker:
    """
    Performance benchmarking utility for framework operations.

    Provides:
    - Execution time measurement
    - Memory usage tracking
    - Throughput calculation
    - Comparison between operations
    """

    def __init__(self):
        """Initialize the performance benchmarker."""
        if PSUTIL_AVAILABLE:
            self.process = psutil.Process(os.getpid())
        else:
            self.process = None

    def benchmark_operation(
        self, operation_name: str, operation: Callable, *args, iterations: int = 1, warmup_iterations: int = 0, **kwargs
    ) -> BenchmarkResult:
        """
        Benchmark a single operation.

        Args:
            operation_name: Name of the operation being benchmarked
            operation: Function or method to benchmark
            *args: Positional arguments for the operation
            iterations: Number of iterations to run (for averaging)
            warmup_iterations: Number of warmup iterations (excluded from timing)
            **kwargs: Keyword arguments for the operation

        Returns:
            BenchmarkResult with performance metrics
        """
        # Warmup iterations
        for _ in range(warmup_iterations):
            try:
                operation(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Warmup iteration failed: {e}")

        # Measure memory before
        tracemalloc.start()
        if self.process:
            memory_before = self.process.memory_info().rss / (1024 * 1024)  # MB
        else:
            memory_before = 0.0

        # Time the operation
        execution_times = []
        result_data = None

        for i in range(iterations):
            start_time = time.perf_counter()
            try:
                result_data = operation(*args, **kwargs)
                end_time = time.perf_counter()
                execution_times.append(end_time - start_time)
            except Exception as e:
                logger.error(f"Benchmark iteration {i} failed: {e}")
                execution_times.append(float("inf"))

        # Measure memory after
        if self.process:
            memory_after = self.process.memory_info().rss / (1024 * 1024)  # MB
        else:
            memory_after = 0.0
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Calculate metrics
        avg_execution_time = np.mean(execution_times) if execution_times else float("inf")
        memory_usage = memory_after - memory_before
        peak_memory = peak / (1024 * 1024)  # MB

        # Estimate data volume
        data_volume = self._estimate_data_volume(result_data, args, kwargs)

        # Calculate throughput
        throughput = 1.0 / avg_execution_time if avg_execution_time > 0 else 0.0

        metadata = {
            "min_time": float(np.min(execution_times)) if execution_times else None,
            "max_time": float(np.max(execution_times)) if execution_times else None,
            "std_time": float(np.std(execution_times)) if len(execution_times) > 1 else 0.0,
            "peak_memory_mb": peak_memory,
            "warmup_iterations": warmup_iterations,
        }

        return BenchmarkResult(
            operation_name=operation_name,
            execution_time=avg_execution_time,
            memory_usage=max(memory_usage, peak_memory),
            data_volume=data_volume,
            throughput=throughput,
            iterations=iterations,
            metadata=metadata,
        )

    def _estimate_data_volume(self, result: Any, args: Tuple, kwargs: Dict) -> int:
        """
        Estimate data volume processed.

        Args:
            result: Operation result
            args: Operation arguments
            kwargs: Operation keyword arguments

        Returns:
            Estimated data volume in bytes or number of elements
        """
        volume = 0

        # Estimate from result
        if result is not None:
            if isinstance(result, np.ndarray):
                volume = result.nbytes
            elif isinstance(result, (list, tuple)):
                volume = len(result)
                if len(result) > 0 and isinstance(result[0], np.ndarray):
                    volume = sum(arr.nbytes for arr in result)
            elif isinstance(result, dict):
                volume = sum(
                    v.nbytes if isinstance(v, np.ndarray) else len(v) if isinstance(v, (list, tuple)) else 1
                    for v in result.values()
                )

        # Estimate from arguments
        for arg in args:
            if isinstance(arg, np.ndarray):
                volume += arg.nbytes
            elif isinstance(arg, (list, tuple)):
                volume += len(arg)

        for value in kwargs.values():
            if isinstance(value, np.ndarray):
                volume += value.nbytes
            elif isinstance(value, (list, tuple)):
                volume += len(value)

        return volume

    def compare_operations(
        self, operations: Dict[str, Tuple[Callable, Tuple, Dict]], iterations: int = 1, warmup_iterations: int = 0
    ) -> Dict[str, BenchmarkResult]:
        """
        Compare multiple operations.

        Args:
            operations: Dictionary mapping operation names to (function, args, kwargs) tuples
            iterations: Number of iterations per operation
            warmup_iterations: Number of warmup iterations

        Returns:
            Dictionary mapping operation names to BenchmarkResult
        """
        results = {}
        for op_name, (op_func, op_args, op_kwargs) in operations.items():
            try:
                results[op_name] = self.benchmark_operation(
                    op_name, op_func, *op_args, iterations=iterations, warmup_iterations=warmup_iterations, **op_kwargs
                )
            except Exception as e:
                logger.error(f"Failed to benchmark {op_name}: {e}")
                results[op_name] = None

        return results

    def generate_report(self, results: List[BenchmarkResult], output_path: Optional[str] = None) -> str:
        """
        Generate a benchmark report.

        Args:
            results: List of benchmark results
            output_path: Optional path to save report

        Returns:
            Report as formatted string
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("Performance Benchmark Report")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        # Summary table
        report_lines.append("Summary")
        report_lines.append("-" * 80)
        report_lines.append(f"{'Operation':<30} {'Time (s)':<15} {'Memory (MB)':<15} {'Throughput (ops/s)':<20}")
        report_lines.append("-" * 80)

        for result in results:
            if result:
                report_lines.append(
                    f"{result.operation_name:<30} "
                    f"{result.execution_time:<15.6f} "
                    f"{result.memory_usage:<15.2f} "
                    f"{result.throughput:<20.2f}"
                )
        report_lines.append("")

        # Detailed results
        report_lines.append("Detailed Results")
        report_lines.append("-" * 80)
        for result in results:
            if result:
                report_lines.append(f"\nOperation: {result.operation_name}")
                report_lines.append(f"  Execution Time: {result.execution_time:.6f} seconds")
                if result.metadata.get("min_time") is not None:
                    report_lines.append(f"    Min: {result.metadata['min_time']:.6f} s")
                    report_lines.append(f"    Max: {result.metadata['max_time']:.6f} s")
                    report_lines.append(f"    Std: {result.metadata['std_time']:.6f} s")
                report_lines.append(f"  Memory Usage: {result.memory_usage:.2f} MB")
                if "peak_memory_mb" in result.metadata:
                    report_lines.append(f"    Peak: {result.metadata['peak_memory_mb']:.2f} MB")
                report_lines.append(f"  Data Volume: {result.data_volume:,} bytes")
                report_lines.append(f"  Throughput: {result.throughput:.2f} operations/second")
                report_lines.append(f"  Iterations: {result.iterations}")

        report_lines.append("")
        report_lines.append("=" * 80)

        report_text = "\n".join(report_lines)

        if output_path:
            with open(output_path, "w") as f:
                f.write(report_text)
            logger.info(f"Benchmark report saved to {output_path}")

        return report_text


def benchmark(iterations: int = 1, warmup_iterations: int = 0, operation_name: Optional[str] = None):
    """
    Decorator for automatic benchmarking of functions.

    Args:
        iterations: Number of iterations to run
        warmup_iterations: Number of warmup iterations
        operation_name: Name for the operation (defaults to function name)

    Usage:
        @benchmark(iterations=10)
        def my_function(arg1, arg2):
            # Function implementation
            pass
    """

    def decorator(func: Callable) -> Callable:
        name = operation_name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            benchmarker = PerformanceBenchmarker()
            result = benchmarker.benchmark_operation(
                name, func, *args, iterations=iterations, warmup_iterations=warmup_iterations, **kwargs
            )

            # Store benchmark result in wrapper function metadata
            if not hasattr(wrapper, "_benchmark_results"):
                wrapper._benchmark_results = []
            wrapper._benchmark_results.append(result)

            # Execute function and return result
            return func(*args, **kwargs)

        return wrapper

    return decorator
