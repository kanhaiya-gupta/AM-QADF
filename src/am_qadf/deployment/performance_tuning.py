"""
Performance tuning and optimization utilities.

This module provides production performance optimization utilities,
profiling, and tuning recommendations.
"""

import time
import cProfile
import pstats
import io
import tracemalloc
from dataclasses import dataclass
from typing import Callable, Any, Dict, List, Optional, Tuple
from datetime import datetime


@dataclass
class PerformanceMetrics:
    """Performance metrics for tuning."""

    throughput: float  # Requests per second
    latency_p50: float  # 50th percentile latency in seconds
    latency_p95: float  # 95th percentile latency in seconds
    latency_p99: float  # 99th percentile latency in seconds
    error_rate: float  # Error rate percentage (0.0-1.0)
    cpu_efficiency: float  # CPU utilization / throughput
    memory_efficiency: float  # Memory usage / throughput

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "throughput": self.throughput,
            "latency_p50": self.latency_p50,
            "latency_p95": self.latency_p95,
            "latency_p99": self.latency_p99,
            "error_rate": self.error_rate,
            "cpu_efficiency": self.cpu_efficiency,
            "memory_efficiency": self.memory_efficiency,
        }


class PerformanceProfiler:
    """Performance profiling utilities."""

    def __init__(self):
        """Initialize performance profiler."""
        self.profiler = None
        self.memory_trace_active = False

    def profile_function(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """
        Profile function execution.

        Args:
            func: Function to profile
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Dictionary with profiling results
        """
        profiler = cProfile.Profile()
        profiler.enable()

        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            success = True
        except Exception as e:
            result = None
            success = False
            exception = str(e)
        finally:
            end_time = time.time()
            profiler.disable()

        # Get profiling statistics
        stats_stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stats_stream)
        stats.sort_stats("cumulative")
        stats.print_stats(20)  # Top 20 functions

        profile_output = stats_stream.getvalue()

        # Extract key metrics
        total_time = end_time - start_time
        total_calls = stats.total_calls

        return {
            "function_name": func.__name__,
            "success": success,
            "total_time": total_time,
            "total_calls": total_calls,
            "calls_per_second": total_calls / total_time if total_time > 0 else 0,
            "profile_output": profile_output,
            "result": result if success else None,
            "exception": exception if not success else None,
            "timestamp": datetime.now().isoformat(),
        }

    def profile_memory(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """
        Profile memory usage of function.

        Args:
            func: Function to profile
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Dictionary with memory profiling results
        """
        tracemalloc.start()

        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            success = True
        except Exception as e:
            result = None
            success = False
            exception = str(e)
        finally:
            end_time = time.time()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

        return {
            "function_name": func.__name__,
            "success": success,
            "execution_time": end_time - start_time,
            "current_memory_mb": current / (1024 * 1024),
            "peak_memory_mb": peak / (1024 * 1024),
            "result": result if success else None,
            "exception": exception if not success else None,
            "timestamp": datetime.now().isoformat(),
        }

    def generate_report(self, profile_data: Dict[str, Any]) -> str:
        """
        Generate performance profiling report.

        Args:
            profile_data: Profiling data from profile_function or profile_memory

        Returns:
            Formatted report string
        """
        report_lines = [
            f"Performance Profile Report",
            f"=" * 50,
            f"Function: {profile_data.get('function_name', 'unknown')}",
            f"Timestamp: {profile_data.get('timestamp', 'unknown')}",
            f"Success: {profile_data.get('success', False)}",
            "",
        ]

        if "total_time" in profile_data:
            # Time-based profiling
            report_lines.extend(
                [
                    f"Execution Time: {profile_data['total_time']:.4f} seconds",
                    f"Total Calls: {profile_data['total_calls']}",
                    f"Calls per Second: {profile_data['calls_per_second']:.2f}",
                    "",
                    "Top Functions by Cumulative Time:",
                    "-" * 50,
                    profile_data.get("profile_output", "No profiling data available"),
                ]
            )
        elif "execution_time" in profile_data:
            # Memory-based profiling
            report_lines.extend(
                [
                    f"Execution Time: {profile_data['execution_time']:.4f} seconds",
                    f"Current Memory: {profile_data['current_memory_mb']:.2f} MB",
                    f"Peak Memory: {profile_data['peak_memory_mb']:.2f} MB",
                ]
            )

        if not profile_data.get("success"):
            report_lines.extend(
                [
                    "",
                    f"Error: {profile_data.get('exception', 'Unknown error')}",
                ]
            )

        return "\n".join(report_lines)


class PerformanceTuner:
    """Automatic performance tuning utilities."""

    def __init__(self):
        """Initialize performance tuner."""
        pass

    def optimize_database_queries(self, query_logs: List[Dict]) -> List[Dict]:
        """
        Analyze and suggest database query optimizations.

        Args:
            query_logs: List of query log dictionaries with keys:
                - query: Query string
                - execution_time: Execution time in seconds
                - rows_returned: Number of rows returned
                - timestamp: Query timestamp

        Returns:
            List of optimization suggestions
        """
        suggestions = []

        # Sort by execution time (slowest first)
        slow_queries = sorted(query_logs, key=lambda x: x.get("execution_time", 0), reverse=True)[
            :10
        ]  # Top 10 slowest queries

        for query_log in slow_queries:
            query = query_log.get("query", "")
            exec_time = query_log.get("execution_time", 0)
            rows = query_log.get("rows_returned", 0)

            suggestion = {
                "query": query[:100] + "..." if len(query) > 100 else query,
                "current_execution_time": exec_time,
                "rows_returned": rows,
                "recommendations": [],
            }

            # Check for common issues
            query_lower = query.lower()

            if "select *" in query_lower:
                suggestion["recommendations"].append("Avoid SELECT * - specify only needed columns")

            if "like %" in query_lower or query_lower.count("like") > 0:
                suggestion["recommendations"].append(
                    "Consider using full-text search or indexed columns instead of LIKE with wildcards"
                )

            if exec_time > 1.0:
                suggestion["recommendations"].append(
                    "Query is very slow - consider adding indexes or optimizing query structure"
                )

            if rows > 10000:
                suggestion["recommendations"].append("Large result set - consider pagination or filtering")

            if not suggestion["recommendations"]:
                suggestion["recommendations"].append("Query performance is acceptable")

            suggestions.append(suggestion)

        return suggestions

    def optimize_cache_settings(self, cache_stats: Dict) -> Dict:
        """
        Suggest cache optimization settings.

        Args:
            cache_stats: Dictionary with cache statistics:
                - hit_rate: Cache hit rate (0.0-1.0)
                - miss_rate: Cache miss rate (0.0-1.0)
                - eviction_count: Number of evictions
                - total_size_mb: Total cache size in MB
                - max_size_mb: Maximum cache size in MB

        Returns:
            Dictionary with optimization suggestions
        """
        hit_rate = cache_stats.get("hit_rate", 0.0)
        miss_rate = cache_stats.get("miss_rate", 0.0)
        eviction_count = cache_stats.get("eviction_count", 0)
        total_size = cache_stats.get("total_size_mb", 0)
        max_size = cache_stats.get("max_size_mb", 0)

        suggestions = {
            "current_hit_rate": hit_rate,
            "recommendations": [],
        }

        if hit_rate < 0.5:
            suggestions["recommendations"].append("Low cache hit rate - consider increasing cache size or TTL")

        if eviction_count > 1000:
            suggestions["recommendations"].append("High eviction count - consider increasing cache size")

        if total_size / max_size > 0.9:
            suggestions["recommendations"].append("Cache is nearly full - consider increasing max_size_mb")

        if hit_rate > 0.8 and eviction_count < 100:
            suggestions["recommendations"].append("Cache performance is good - current settings are optimal")

        return suggestions

    def optimize_worker_threads(self, metrics: Dict) -> int:
        """
        Suggest optimal number of worker threads.

        Args:
            metrics: Dictionary with performance metrics:
                - cpu_utilization: CPU utilization (0.0-1.0)
                - current_threads: Current number of threads
                - throughput: Requests per second
                - avg_latency: Average latency in seconds

        Returns:
            Suggested number of worker threads
        """
        cpu_utilization = metrics.get("cpu_utilization", 0.5)
        current_threads = metrics.get("current_threads", 4)
        throughput = metrics.get("throughput", 0)
        avg_latency = metrics.get("avg_latency", 0.1)

        # Base calculation: threads = CPU cores * (1 + wait_time / compute_time)
        # Simplified: use CPU utilization and throughput
        if cpu_utilization < 0.5:
            # CPU underutilized, can increase threads
            suggested = int(current_threads * 1.5)
        elif cpu_utilization > 0.9:
            # CPU overutilized, reduce threads
            suggested = int(current_threads * 0.8)
        else:
            # CPU well utilized, keep similar
            suggested = current_threads

        # Consider latency
        if avg_latency > 1.0:
            # High latency, might need more threads for I/O-bound work
            suggested = max(suggested, current_threads + 2)

        # Clamp to reasonable range
        suggested = max(1, min(32, suggested))

        return suggested

    def generate_tuning_recommendations(self, current_config: Any, metrics: Dict) -> List[str]:  # ProductionConfig
        """
        Generate performance tuning recommendations.

        Args:
            current_config: Current production configuration
            metrics: Performance metrics dictionary

        Returns:
            List of tuning recommendations
        """
        recommendations = []

        # Check worker threads
        if hasattr(current_config, "worker_threads"):
            optimal_threads = self.optimize_worker_threads(
                {
                    "cpu_utilization": metrics.get("cpu_utilization", 0.5),
                    "current_threads": current_config.worker_threads,
                    "throughput": metrics.get("throughput", 0),
                    "avg_latency": metrics.get("avg_latency", 0.1),
                }
            )
            if optimal_threads != current_config.worker_threads:
                recommendations.append(
                    f"Consider adjusting worker_threads from {current_config.worker_threads} to {optimal_threads}"
                )

        # Check database pool size
        if hasattr(current_config, "database_pool_size"):
            pool_size = current_config.database_pool_size
            if metrics.get("db_connection_wait_time", 0) > 0.1:
                recommendations.append(
                    f"Database connection pool may be too small (current: {pool_size}). "
                    f"Consider increasing database_pool_size"
                )

        # Check request timeout
        if hasattr(current_config, "request_timeout"):
            timeout = current_config.request_timeout
            if metrics.get("p99_latency", 0) > timeout * 0.9:
                recommendations.append(
                    f"Request timeout ({timeout}s) may be too low. " f"P99 latency is {metrics.get('p99_latency', 0):.2f}s"
                )

        # Check max concurrent requests
        if hasattr(current_config, "max_concurrent_requests"):
            max_requests = current_config.max_concurrent_requests
            if metrics.get("queue_depth", 0) > max_requests * 0.8:
                recommendations.append(
                    f"Queue depth is high. Consider increasing max_concurrent_requests " f"from {max_requests}"
                )

        if not recommendations:
            recommendations.append("Current configuration appears optimal based on metrics")

        return recommendations
