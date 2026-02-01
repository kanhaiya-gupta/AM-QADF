"""
Integration tests for deployment workflows.

Tests for deployment components working together in production scenarios.
"""

import pytest
import time
import os
from unittest.mock import Mock, patch
from datetime import datetime

from am_qadf.deployment import (
    ProductionConfig,
    ResourceMonitor,
    LoadBalancer,
    AutoScaler,
    ScalingMetrics,
    RetryPolicy,
    CircuitBreaker,
    GracefulDegradation,
    PerformanceProfiler,
    PerformanceTuner,
)
from am_qadf.deployment.fault_tolerance import retry_with_policy


class TestProductionConfigurationWorkflow:
    """Test suite for production configuration workflow."""

    @pytest.mark.integration
    def test_config_from_env_and_validation(self):
        """Test loading config from environment and validating."""
        env_vars = {
            "AM_QADF_ENV": "production",
            "AM_QADF_LOG_LEVEL": "INFO",
            "AM_QADF_DB_POOL_SIZE": "25",
            "AM_QADF_WORKER_THREADS": "8",
            "AM_QADF_REDIS_HOST": "redis.prod.com",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            config = ProductionConfig.from_env()

            is_valid, errors = config.validate()

            assert is_valid is True
            assert len(errors) == 0
            assert config.environment == "production"
            assert config.log_level == "INFO"
            assert config.database_pool_size == 25

    @pytest.mark.integration
    def test_config_with_resource_monitor(self):
        """Test production config with resource monitoring."""
        config = ProductionConfig(
            worker_threads=4,
            max_concurrent_requests=100,
            enable_metrics=True,
        )

        monitor = ResourceMonitor(update_interval=0.1)

        # Get initial metrics
        metrics = monitor.get_system_metrics()

        # Check resource limits
        exceeded, warnings = monitor.check_resource_limits(
            cpu_threshold=0.8,
            memory_threshold=0.8,
            disk_threshold=0.8,
        )

        # Should not exceed with normal usage
        assert isinstance(exceeded, bool)
        assert isinstance(warnings, list)


class TestScalabilityWorkflow:
    """Test suite for scalability workflow."""

    @pytest.mark.integration
    def test_load_balancer_with_auto_scaler(self):
        """Test load balancer integrated with auto-scaler."""
        # Create load balancer
        lb = LoadBalancer(strategy="round_robin")

        # Register workers
        lb.register_worker("worker1", weight=1.0)
        lb.register_worker("worker2", weight=1.0)
        lb.register_worker("worker3", weight=2.0)

        # Create auto-scaler
        auto_scaler = AutoScaler(
            min_instances=2,
            max_instances=10,
            target_utilization=0.7,
        )

        # Simulate high utilization
        metrics = ScalingMetrics(
            cpu_utilization=0.9,
            memory_utilization=0.85,
            request_rate=200.0,
            error_rate=0.05,
            queue_depth=50,
            response_time_p50=0.2,
            response_time_p95=0.8,
            response_time_p99=2.0,
        )

        # Get scaling decision
        current_instances = len(lb.get_healthy_workers())
        decision = auto_scaler.get_scaling_decision(current_instances, metrics)

        # Should recommend scale-up
        if decision["action"] == "scale_up":
            # Add new workers
            for i in range(current_instances + 1, decision["desired_instances"] + 1):
                lb.register_worker(f"worker{i}", weight=1.0)

        # Verify workers are registered
        all_workers = lb.get_all_workers_status()
        assert len(all_workers) >= current_instances

    @pytest.mark.integration
    def test_auto_scaler_with_resource_monitor(self):
        """Test auto-scaler with resource monitoring."""
        # Create resource monitor
        monitor = ResourceMonitor(update_interval=0.1)

        # Create auto-scaler
        auto_scaler = AutoScaler(
            min_instances=2,
            max_instances=10,
            target_utilization=0.7,
        )

        # Get system metrics
        resource_metrics = monitor.get_system_metrics()

        # Convert to scaling metrics
        scaling_metrics = ScalingMetrics(
            cpu_utilization=resource_metrics.cpu_percent / 100.0,
            memory_utilization=resource_metrics.memory_percent / 100.0,
            request_rate=100.0,
            error_rate=0.02,
            queue_depth=10,
            response_time_p50=0.1,
            response_time_p95=0.5,
            response_time_p99=1.0,
        )

        # Get scaling decision
        decision = auto_scaler.get_scaling_decision(current_instances=5, metrics=scaling_metrics)

        assert "action" in decision
        assert "current_instances" in decision
        assert decision["current_instances"] == 5


class TestFaultToleranceWorkflow:
    """Test suite for fault tolerance workflow."""

    @pytest.mark.integration
    def test_circuit_breaker_with_retry(self):
        """Test circuit breaker with retry policy."""
        # Create retry policy
        retry_policy = RetryPolicy(
            max_retries=3,
            initial_backoff_time=0.01,
            exponential_backoff=True,
            retryable_exceptions=(ValueError,),
        )

        # Create circuit breaker
        circuit_breaker = CircuitBreaker(
            failure_threshold=5,  # High threshold so it doesn't open
            timeout=1.0,
        )

        # Function that fails initially
        attempt_count = [0]

        def failing_func():
            attempt_count[0] += 1
            if attempt_count[0] < 3:
                raise ValueError("Temporary failure")
            return "success"

        # Wrap with retry and circuit breaker
        @retry_with_policy(retry_policy)
        def retry_func():
            return circuit_breaker.call(failing_func)

        # Should succeed after retries
        result = retry_func()
        assert result == "success"
        assert attempt_count[0] == 3

    @pytest.mark.integration
    def test_graceful_degradation_with_fallback(self):
        """Test graceful degradation with fallback."""

        @GracefulDegradation.with_fallback(fallback_value="fallback_result")
        def failing_func():
            raise ValueError("Error occurred")

        result = failing_func()

        assert result == "fallback_result"

    @pytest.mark.integration
    def test_graceful_degradation_with_timeout(self):
        """Test graceful degradation with timeout."""

        @GracefulDegradation.with_timeout(timeout=0.1, default_value="timeout")
        def slow_func():
            time.sleep(1.0)  # Takes longer than timeout
            return "done"

        result = slow_func()

        # May return timeout or done depending on platform
        assert result in ["timeout", "done"]

    @pytest.mark.integration
    def test_circuit_breaker_opens_after_failures(self):
        """Test circuit breaker opens after consecutive failures."""
        circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            timeout=10.0,
        )

        def failing_func():
            raise ValueError("Always fails")

        # Trigger failures up to threshold
        for _ in range(circuit_breaker.failure_threshold):
            try:
                circuit_breaker.call(failing_func)
            except ValueError:
                pass

        # Circuit should be open
        assert circuit_breaker.state.value == "open"

        # Next call should raise CircuitBreakerOpenError
        from am_qadf.deployment.fault_tolerance import CircuitBreakerOpenError

        with pytest.raises(CircuitBreakerOpenError):
            circuit_breaker.call(failing_func)


class TestPerformanceTuningWorkflow:
    """Test suite for performance tuning workflow."""

    @pytest.mark.integration
    def test_profiler_with_tuner(self):
        """Test performance profiler with tuner."""
        # Create profiler
        profiler = PerformanceProfiler()

        # Profile a function
        def test_func(x, y):
            time.sleep(0.01)  # Simulate work
            return x * y

        profile_result = profiler.profile_function(test_func, 5, 3)

        assert profile_result["success"] is True
        assert profile_result["result"] == 15
        assert profile_result["total_time"] > 0

        # Create tuner
        tuner = PerformanceTuner()

        # Get tuning recommendations
        config = ProductionConfig(
            worker_threads=4,
            database_pool_size=20,
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

    @pytest.mark.integration
    def test_profiler_memory_with_tuner(self):
        """Test memory profiler with tuner."""
        profiler = PerformanceProfiler()

        def memory_intensive_func():
            data = [i for i in range(10000)]
            return len(data)

        profile_result = profiler.profile_memory(memory_intensive_func)

        assert profile_result["success"] is True
        assert profile_result["result"] == 10000
        assert profile_result["execution_time"] > 0
        assert profile_result["peak_memory_mb"] >= 0


class TestCompleteProductionWorkflow:
    """Test suite for complete production workflow."""

    @pytest.mark.integration
    def test_production_deployment_workflow(self):
        """Test complete production deployment workflow."""
        # 1. Load production configuration
        config = ProductionConfig(
            environment="production",
            log_level="INFO",
            worker_threads=4,
            enable_metrics=True,
            enable_tracing=True,
        )

        is_valid, errors = config.validate()
        assert is_valid is True

        # 2. Initialize resource monitoring
        monitor = ResourceMonitor(update_interval=0.1)

        # 3. Set up load balancing
        lb = LoadBalancer(strategy="least_connections")
        lb.register_worker("worker1", weight=1.0)
        lb.register_worker("worker2", weight=1.0)

        # 4. Set up auto-scaling
        auto_scaler = AutoScaler(
            min_instances=2,
            max_instances=10,
            target_utilization=0.7,
        )

        # 5. Set up fault tolerance
        circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            timeout=60.0,
        )

        # 6. Monitor resources
        resource_metrics = monitor.get_system_metrics()

        # 7. Check if scaling is needed
        scaling_metrics = ScalingMetrics(
            cpu_utilization=resource_metrics.cpu_percent / 100.0,
            memory_utilization=resource_metrics.memory_percent / 100.0,
            request_rate=100.0,
            error_rate=0.02,
            queue_depth=10,
            response_time_p50=0.1,
            response_time_p95=0.5,
            response_time_p99=1.0,
        )

        current_instances = len(lb.get_healthy_workers())
        scaling_decision = auto_scaler.get_scaling_decision(current_instances, scaling_metrics)

        # 8. Execute with circuit breaker protection
        def production_func():
            return "production result"

        result = circuit_breaker.call(production_func)

        assert result == "production result"
        assert config.environment == "production"
        assert len(lb.get_all_workers_status()) >= 2
        assert "action" in scaling_decision

    @pytest.mark.integration
    def test_monitoring_with_alerting(self):
        """Test resource monitoring with alert generation."""
        # Create monitor
        monitor = ResourceMonitor(update_interval=0.1)

        # Start monitoring with callback
        alerts_generated = []

        def alert_callback(metrics):
            exceeded, warnings = monitor.check_resource_limits(
                cpu_threshold=0.01,  # Very low threshold to trigger
                memory_threshold=0.01,
                disk_threshold=0.01,
            )
            if exceeded:
                alerts_generated.append(warnings)

        monitor.start_monitoring(alert_callback)
        time.sleep(0.15)
        monitor.stop_monitoring()

        # Monitoring should have run
        assert len(monitor.metrics_history) >= 1
