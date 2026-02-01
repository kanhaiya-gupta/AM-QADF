"""
Unit tests for performance tuning utilities.

Tests for PerformanceProfiler and PerformanceTuner.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock

from am_qadf.deployment.performance_tuning import (
    PerformanceMetrics,
    PerformanceProfiler,
    PerformanceTuner,
)
from am_qadf.deployment.production_config import ProductionConfig


class TestPerformanceMetrics:
    """Test suite for PerformanceMetrics dataclass."""

    @pytest.mark.unit
    def test_performance_metrics_creation(self):
        """Test creating PerformanceMetrics."""
        metrics = PerformanceMetrics(
            throughput=100.0,
            latency_p50=0.1,
            latency_p95=0.5,
            latency_p99=1.0,
            error_rate=0.05,
            cpu_efficiency=0.5,
            memory_efficiency=0.3,
        )

        assert metrics.throughput == 100.0
        assert metrics.latency_p50 == 0.1
        assert metrics.latency_p95 == 0.5
        assert metrics.latency_p99 == 1.0
        assert metrics.error_rate == 0.05
        assert metrics.cpu_efficiency == 0.5
        assert metrics.memory_efficiency == 0.3

    @pytest.mark.unit
    def test_performance_metrics_to_dict(self):
        """Test converting PerformanceMetrics to dictionary."""
        metrics = PerformanceMetrics(
            throughput=100.0,
            latency_p50=0.1,
            latency_p95=0.5,
            latency_p99=1.0,
            error_rate=0.05,
            cpu_efficiency=0.5,
            memory_efficiency=0.3,
        )

        metrics_dict = metrics.to_dict()

        assert isinstance(metrics_dict, dict)
        assert metrics_dict["throughput"] == 100.0
        assert metrics_dict["latency_p50"] == 0.1
        assert metrics_dict["error_rate"] == 0.05


class TestPerformanceProfiler:
    """Test suite for PerformanceProfiler class."""

    @pytest.fixture
    def profiler(self):
        """Create a PerformanceProfiler instance."""
        return PerformanceProfiler()

    @pytest.mark.unit
    def test_profiler_creation(self, profiler):
        """Test creating PerformanceProfiler."""
        assert profiler.profiler is None
        assert profiler.memory_trace_active is False

    @pytest.mark.unit
    def test_profile_function_success(self, profiler):
        """Test profiling a successful function."""

        def test_func(x, y):
            return x + y

        result = profiler.profile_function(test_func, 2, 3)

        assert result["function_name"] == "test_func"
        assert result["success"] is True
        assert result["result"] == 5
        assert result["total_time"] > 0
        assert result["total_calls"] > 0
        assert "profile_output" in result

    @pytest.mark.unit
    def test_profile_function_failure(self, profiler):
        """Test profiling a failing function."""

        def failing_func():
            raise ValueError("test error")

        result = profiler.profile_function(failing_func)

        assert result["function_name"] == "failing_func"
        assert result["success"] is False
        assert result["result"] is None
        assert "test error" in result["exception"]

    @pytest.mark.unit
    def test_profile_function_with_args(self, profiler):
        """Test profiling function with arguments."""

        def multiply(x, y, z=1):
            return x * y * z

        result = profiler.profile_function(multiply, 2, 3, z=4)

        assert result["success"] is True
        assert result["result"] == 24

    @pytest.mark.unit
    def test_profile_memory_success(self, profiler):
        """Test memory profiling of successful function."""

        def test_func():
            data = [i for i in range(1000)]
            return len(data)

        result = profiler.profile_memory(test_func)

        assert result["function_name"] == "test_func"
        assert result["success"] is True
        assert result["result"] == 1000
        assert result["execution_time"] > 0
        assert result["current_memory_mb"] >= 0
        assert result["peak_memory_mb"] >= 0

    @pytest.mark.unit
    def test_profile_memory_failure(self, profiler):
        """Test memory profiling of failing function."""

        def failing_func():
            raise ValueError("error")

        result = profiler.profile_memory(failing_func)

        assert result["success"] is False
        assert "error" in result["exception"]

    @pytest.mark.unit
    def test_generate_report_function(self, profiler):
        """Test generating report for function profile."""

        def test_func():
            time.sleep(0.01)
            return "result"

        profile_data = profiler.profile_function(test_func)
        report = profiler.generate_report(profile_data)

        assert isinstance(report, str)
        assert "Performance Profile Report" in report
        assert "test_func" in report
        assert "Execution Time" in report
        assert "Total Calls" in report

    @pytest.mark.unit
    def test_generate_report_memory(self, profiler):
        """Test generating report for memory profile."""

        def test_func():
            data = list(range(100))
            return len(data)

        profile_data = profiler.profile_memory(test_func)
        report = profiler.generate_report(profile_data)

        assert isinstance(report, str)
        assert "Performance Profile Report" in report
        assert "Execution Time" in report
        assert "Memory" in report or "memory" in report

    @pytest.mark.unit
    def test_generate_report_with_error(self, profiler):
        """Test generating report with error."""
        profile_data = {
            "function_name": "test_func",
            "success": False,
            "exception": "Test error message",
            "timestamp": "2024-01-01T00:00:00",
        }

        report = profiler.generate_report(profile_data)

        assert "Test error message" in report
        assert "Error:" in report


class TestPerformanceTuner:
    """Test suite for PerformanceTuner class."""

    @pytest.fixture
    def tuner(self):
        """Create a PerformanceTuner instance."""
        return PerformanceTuner()

    @pytest.mark.unit
    def test_tuner_creation(self, tuner):
        """Test creating PerformanceTuner."""
        assert tuner is not None

    @pytest.mark.unit
    def test_optimize_database_queries(self, tuner):
        """Test database query optimization suggestions."""
        query_logs = [
            {
                "query": 'SELECT * FROM users WHERE name LIKE "%test%"',
                "execution_time": 2.5,
                "rows_returned": 5000,
            },
            {
                "query": "SELECT id, name FROM users WHERE id = 123",
                "execution_time": 0.01,
                "rows_returned": 1,
            },
            {
                "query": 'SELECT * FROM orders WHERE date > "2024-01-01"',
                "execution_time": 1.8,
                "rows_returned": 15000,
            },
        ]

        suggestions = tuner.optimize_database_queries(query_logs)

        assert len(suggestions) <= 10  # Top 10 slowest
        assert len(suggestions) > 0

        # Check first suggestion (slowest query)
        first_suggestion = suggestions[0]
        assert "query" in first_suggestion
        assert "recommendations" in first_suggestion
        assert len(first_suggestion["recommendations"]) > 0

    @pytest.mark.unit
    def test_optimize_database_queries_empty(self, tuner):
        """Test database query optimization with empty logs."""
        suggestions = tuner.optimize_database_queries([])

        assert len(suggestions) == 0

    @pytest.mark.unit
    def test_optimize_cache_settings_low_hit_rate(self, tuner):
        """Test cache optimization with low hit rate."""
        cache_stats = {
            "hit_rate": 0.3,
            "miss_rate": 0.7,
            "eviction_count": 5000,
            "total_size_mb": 900,
            "max_size_mb": 1000,
        }

        suggestions = tuner.optimize_cache_settings(cache_stats)

        assert suggestions["current_hit_rate"] == 0.3
        assert len(suggestions["recommendations"]) > 0
        assert any("hit rate" in r.lower() for r in suggestions["recommendations"])

    @pytest.mark.unit
    def test_optimize_cache_settings_high_eviction(self, tuner):
        """Test cache optimization with high eviction count."""
        cache_stats = {
            "hit_rate": 0.6,
            "miss_rate": 0.4,
            "eviction_count": 2000,
            "total_size_mb": 950,
            "max_size_mb": 1000,
        }

        suggestions = tuner.optimize_cache_settings(cache_stats)

        assert len(suggestions["recommendations"]) > 0
        assert any("eviction" in r.lower() or "size" in r.lower() for r in suggestions["recommendations"])

    @pytest.mark.unit
    def test_optimize_cache_settings_good_performance(self, tuner):
        """Test cache optimization with good performance."""
        cache_stats = {
            "hit_rate": 0.9,
            "miss_rate": 0.1,
            "eviction_count": 50,
            "total_size_mb": 500,
            "max_size_mb": 1000,
        }

        suggestions = tuner.optimize_cache_settings(cache_stats)

        assert len(suggestions["recommendations"]) > 0
        assert any("good" in r.lower() or "optimal" in r.lower() for r in suggestions["recommendations"])

    @pytest.mark.unit
    def test_optimize_worker_threads_low_cpu(self, tuner):
        """Test worker thread optimization with low CPU."""
        metrics = {
            "cpu_utilization": 0.3,
            "current_threads": 4,
            "throughput": 50.0,
            "avg_latency": 0.1,
        }

        suggested = tuner.optimize_worker_threads(metrics)

        # Should increase threads for low CPU
        assert suggested >= 4
        assert suggested <= 32

    @pytest.mark.unit
    def test_optimize_worker_threads_high_cpu(self, tuner):
        """Test worker thread optimization with high CPU."""
        metrics = {
            "cpu_utilization": 0.95,
            "current_threads": 10,
            "throughput": 200.0,
            "avg_latency": 0.1,
        }

        suggested = tuner.optimize_worker_threads(metrics)

        # Should decrease threads for high CPU
        assert suggested <= 10
        assert suggested >= 1

    @pytest.mark.unit
    def test_optimize_worker_threads_high_latency(self, tuner):
        """Test worker thread optimization with high latency."""
        metrics = {
            "cpu_utilization": 0.5,
            "current_threads": 4,
            "throughput": 50.0,
            "avg_latency": 2.0,  # High latency
        }

        suggested = tuner.optimize_worker_threads(metrics)

        # Should increase threads for I/O-bound work
        assert suggested >= 4
        assert suggested <= 32

    @pytest.mark.unit
    def test_optimize_worker_threads_clamped(self, tuner):
        """Test worker thread optimization is clamped."""
        metrics = {
            "cpu_utilization": 0.1,
            "current_threads": 1,
            "throughput": 10.0,
            "avg_latency": 0.05,
        }

        suggested = tuner.optimize_worker_threads(metrics)

        # Should be clamped between 1 and 32
        assert 1 <= suggested <= 32

    @pytest.mark.unit
    def test_generate_tuning_recommendations(self, tuner):
        """Test generating tuning recommendations."""
        config = ProductionConfig(
            worker_threads=4,
            database_pool_size=20,
            request_timeout=60.0,
            max_concurrent_requests=100,
        )

        metrics = {
            "cpu_utilization": 0.9,
            "throughput": 50.0,
            "avg_latency": 0.2,
            "p99_latency": 2.0,
            "db_connection_wait_time": 0.2,
            "queue_depth": 90,
        }

        recommendations = tuner.generate_tuning_recommendations(config, metrics)

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

    @pytest.mark.unit
    def test_generate_tuning_recommendations_optimal(self, tuner):
        """Test tuning recommendations with optimal configuration."""
        config = ProductionConfig(
            worker_threads=4,
            database_pool_size=20,
            request_timeout=60.0,
            max_concurrent_requests=100,
        )

        metrics = {
            "cpu_utilization": 0.7,
            "throughput": 100.0,
            "avg_latency": 0.1,
            "p99_latency": 0.5,
            "db_connection_wait_time": 0.01,
            "queue_depth": 20,
        }

        recommendations = tuner.generate_tuning_recommendations(config, metrics)

        # Should have at least one recommendation (even if it says optimal)
        assert len(recommendations) > 0

    @pytest.mark.unit
    def test_generate_tuning_recommendations_no_config_attributes(self, tuner):
        """Test tuning recommendations with config missing attributes."""

        class SimpleConfig:
            pass

        config = SimpleConfig()
        metrics = {
            "cpu_utilization": 0.7,
            "throughput": 100.0,
        }

        recommendations = tuner.generate_tuning_recommendations(config, metrics)

        # Should still return recommendations
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert "optimal" in recommendations[0].lower()
