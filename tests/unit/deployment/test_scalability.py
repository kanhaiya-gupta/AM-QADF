"""
Unit tests for scalability utilities.

Tests for LoadBalancer and AutoScaler.
"""

import pytest
from unittest.mock import Mock, patch
import time

from am_qadf.deployment.scalability import (
    LoadBalancingStrategy,
    ScalingMetrics,
    WorkerStatus,
    LoadBalancer,
    AutoScaler,
)


class TestLoadBalancingStrategy:
    """Test suite for LoadBalancingStrategy enum."""

    @pytest.mark.unit
    def test_load_balancing_strategy_values(self):
        """Test LoadBalancingStrategy enum values."""
        assert LoadBalancingStrategy.ROUND_ROBIN.value == "round_robin"
        assert LoadBalancingStrategy.LEAST_CONNECTIONS.value == "least_connections"
        assert LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN.value == "weighted_round_robin"
        assert LoadBalancingStrategy.IP_HASH.value == "ip_hash"


class TestScalingMetrics:
    """Test suite for ScalingMetrics dataclass."""

    @pytest.mark.unit
    def test_scaling_metrics_creation(self):
        """Test creating ScalingMetrics."""
        metrics = ScalingMetrics(
            cpu_utilization=0.7,
            memory_utilization=0.6,
            request_rate=100.0,
            error_rate=0.05,
            queue_depth=10,
            response_time_p50=0.1,
            response_time_p95=0.5,
            response_time_p99=1.0,
        )

        assert metrics.cpu_utilization == 0.7
        assert metrics.memory_utilization == 0.6
        assert metrics.request_rate == 100.0
        assert metrics.error_rate == 0.05
        assert metrics.queue_depth == 10
        assert metrics.response_time_p50 == 0.1
        assert metrics.response_time_p95 == 0.5
        assert metrics.response_time_p99 == 1.0

    @pytest.mark.unit
    def test_scaling_metrics_to_dict(self):
        """Test converting ScalingMetrics to dictionary."""
        metrics = ScalingMetrics(
            cpu_utilization=0.7,
            memory_utilization=0.6,
            request_rate=100.0,
            error_rate=0.05,
            queue_depth=10,
            response_time_p50=0.1,
            response_time_p95=0.5,
            response_time_p99=1.0,
        )

        metrics_dict = metrics.to_dict()

        assert isinstance(metrics_dict, dict)
        assert metrics_dict["cpu_utilization"] == 0.7
        assert metrics_dict["memory_utilization"] == 0.6


class TestWorkerStatus:
    """Test suite for WorkerStatus dataclass."""

    @pytest.mark.unit
    def test_worker_status_creation(self):
        """Test creating WorkerStatus."""
        status = WorkerStatus(
            worker_id="worker1",
            status="healthy",
            active_connections=5,
            total_requests=100,
            error_count=2,
            last_health_check=time.time(),
            weight=1.0,
        )

        assert status.worker_id == "worker1"
        assert status.status == "healthy"
        assert status.active_connections == 5
        assert status.total_requests == 100
        assert status.error_count == 2
        assert status.weight == 1.0

    @pytest.mark.unit
    def test_worker_status_is_available(self):
        """Test worker availability check."""
        healthy_status = WorkerStatus(
            worker_id="worker1",
            status="healthy",
            active_connections=0,
            total_requests=0,
            error_count=0,
            last_health_check=time.time(),
        )
        assert healthy_status.is_available() is True

        draining_status = WorkerStatus(
            worker_id="worker2",
            status="draining",
            active_connections=0,
            total_requests=0,
            error_count=0,
            last_health_check=time.time(),
        )
        assert draining_status.is_available() is True

        down_status = WorkerStatus(
            worker_id="worker3",
            status="down",
            active_connections=0,
            total_requests=0,
            error_count=0,
            last_health_check=time.time(),
        )
        assert down_status.is_available() is False

    @pytest.mark.unit
    def test_worker_status_get_health_score(self):
        """Test worker health score calculation."""
        healthy_status = WorkerStatus(
            worker_id="worker1",
            status="healthy",
            active_connections=0,
            total_requests=100,
            error_count=0,
            last_health_check=time.time(),
        )
        assert healthy_status.get_health_score() > 0.9

        unhealthy_status = WorkerStatus(
            worker_id="worker2",
            status="unhealthy",
            active_connections=0,
            total_requests=0,
            error_count=0,
            last_health_check=time.time(),
        )
        assert unhealthy_status.get_health_score() == 0.3

        down_status = WorkerStatus(
            worker_id="worker3",
            status="down",
            active_connections=0,
            total_requests=0,
            error_count=0,
            last_health_check=time.time(),
        )
        assert down_status.get_health_score() == 0.0

        high_error_status = WorkerStatus(
            worker_id="worker4",
            status="healthy",
            active_connections=0,
            total_requests=100,
            error_count=50,  # 50% error rate
            last_health_check=time.time(),
        )
        score = high_error_status.get_health_score()
        assert 0.4 <= score <= 0.6  # Should be around 0.5


class TestLoadBalancer:
    """Test suite for LoadBalancer class."""

    @pytest.fixture
    def load_balancer(self):
        """Create a LoadBalancer instance."""
        return LoadBalancer(strategy="round_robin")

    @pytest.mark.unit
    def test_load_balancer_creation_default(self):
        """Test creating LoadBalancer with default strategy."""
        lb = LoadBalancer()

        assert lb.strategy == LoadBalancingStrategy.ROUND_ROBIN
        assert len(lb.workers) == 0
        assert lb.round_robin_index == 0

    @pytest.mark.unit
    def test_load_balancer_creation_invalid_strategy(self):
        """Test creating LoadBalancer with invalid strategy."""
        with pytest.raises(ValueError, match="Invalid load balancing strategy"):
            LoadBalancer(strategy="invalid_strategy")

    @pytest.mark.unit
    def test_register_worker(self, load_balancer):
        """Test registering a worker."""
        load_balancer.register_worker("worker1", weight=1.0)

        assert "worker1" in load_balancer.workers
        assert load_balancer.workers["worker1"].worker_id == "worker1"
        assert load_balancer.workers["worker1"].status == "healthy"
        assert load_balancer.workers["worker1"].weight == 1.0

    @pytest.mark.unit
    def test_register_worker_update_weight(self, load_balancer):
        """Test updating worker weight."""
        load_balancer.register_worker("worker1", weight=1.0)
        load_balancer.register_worker("worker1", weight=2.0)

        assert load_balancer.workers["worker1"].weight == 2.0
        assert load_balancer.worker_weights["worker1"] == 2.0

    @pytest.mark.unit
    def test_unregister_worker(self, load_balancer):
        """Test unregistering a worker."""
        load_balancer.register_worker("worker1")
        assert "worker1" in load_balancer.workers

        load_balancer.unregister_worker("worker1")
        assert "worker1" not in load_balancer.workers

    @pytest.mark.unit
    def test_update_worker_status(self, load_balancer):
        """Test updating worker status."""
        load_balancer.register_worker("worker1")

        load_balancer.update_worker_status("worker1", "unhealthy")
        assert load_balancer.workers["worker1"].status == "unhealthy"

    @pytest.mark.unit
    def test_record_request(self, load_balancer):
        """Test recording a request."""
        load_balancer.register_worker("worker1")

        load_balancer.record_request("worker1", success=True)
        assert load_balancer.workers["worker1"].total_requests == 1
        assert load_balancer.workers["worker1"].error_count == 0

        load_balancer.record_request("worker1", success=False)
        assert load_balancer.workers["worker1"].total_requests == 2
        assert load_balancer.workers["worker1"].error_count == 1

    @pytest.mark.unit
    def test_record_connection(self, load_balancer):
        """Test recording a connection."""
        load_balancer.register_worker("worker1")

        load_balancer.record_connection("worker1", connected=True)
        assert load_balancer.workers["worker1"].active_connections == 1

        load_balancer.record_connection("worker1", connected=False)
        assert load_balancer.workers["worker1"].active_connections == 0

    @pytest.mark.unit
    def test_select_worker_round_robin(self, load_balancer):
        """Test round-robin worker selection."""
        load_balancer.register_worker("worker1")
        load_balancer.register_worker("worker2")
        load_balancer.register_worker("worker3")

        # Should cycle through workers
        worker1 = load_balancer.select_worker()
        worker2 = load_balancer.select_worker()
        worker3 = load_balancer.select_worker()
        worker4 = load_balancer.select_worker()

        assert worker1 == "worker1"
        assert worker2 == "worker2"
        assert worker3 == "worker3"
        assert worker4 == "worker1"  # Back to first

    @pytest.mark.unit
    def test_select_worker_no_workers(self, load_balancer):
        """Test worker selection with no workers."""
        worker = load_balancer.select_worker()
        assert worker is None

    @pytest.mark.unit
    def test_select_worker_only_unhealthy(self, load_balancer):
        """Test worker selection with only unhealthy workers."""
        load_balancer.register_worker("worker1")
        load_balancer.update_worker_status("worker1", "down")

        worker = load_balancer.select_worker()
        assert worker is None

    @pytest.mark.unit
    def test_select_worker_least_connections(self):
        """Test least connections worker selection."""
        lb = LoadBalancer(strategy="least_connections")
        lb.register_worker("worker1")
        lb.register_worker("worker2")
        lb.register_worker("worker3")

        # Set different connection counts
        lb.workers["worker1"].active_connections = 10
        lb.workers["worker2"].active_connections = 5
        lb.workers["worker3"].active_connections = 15

        worker = lb.select_worker()
        assert worker == "worker2"  # Least connections

    @pytest.mark.unit
    def test_select_worker_ip_hash(self):
        """Test IP hash worker selection."""
        lb = LoadBalancer(strategy="ip_hash")
        lb.register_worker("worker1")
        lb.register_worker("worker2")
        lb.register_worker("worker3")

        # Same IP should select same worker
        worker1 = lb.select_worker(client_ip="192.168.1.100")
        worker2 = lb.select_worker(client_ip="192.168.1.100")
        assert worker1 == worker2

        # Different IP might select different worker
        worker3 = lb.select_worker(client_ip="192.168.1.200")
        # Worker3 might be different (depends on hash)

    @pytest.mark.unit
    def test_select_worker_ip_hash_no_ip(self):
        """Test IP hash fallback when no IP provided."""
        lb = LoadBalancer(strategy="ip_hash")
        lb.register_worker("worker1")
        lb.register_worker("worker2")

        # Should fallback to round-robin
        worker1 = lb.select_worker()
        worker2 = lb.select_worker()

        assert worker1 is not None
        assert worker2 is not None

    @pytest.mark.unit
    def test_get_worker_status(self, load_balancer):
        """Test getting worker status."""
        load_balancer.register_worker("worker1")

        status = load_balancer.get_worker_status("worker1")
        assert status is not None
        assert status.worker_id == "worker1"

        status = load_balancer.get_worker_status("nonexistent")
        assert status is None

    @pytest.mark.unit
    def test_get_all_workers_status(self, load_balancer):
        """Test getting all workers status."""
        load_balancer.register_worker("worker1")
        load_balancer.register_worker("worker2")

        all_status = load_balancer.get_all_workers_status()
        assert len(all_status) == 2
        assert "worker1" in all_status
        assert "worker2" in all_status

    @pytest.mark.unit
    def test_get_healthy_workers(self, load_balancer):
        """Test getting healthy workers."""
        load_balancer.register_worker("worker1")
        load_balancer.register_worker("worker2")
        load_balancer.register_worker("worker3")

        load_balancer.update_worker_status("worker2", "unhealthy")
        load_balancer.update_worker_status("worker3", "down")

        healthy = load_balancer.get_healthy_workers()
        assert len(healthy) == 1
        assert "worker1" in healthy


class TestAutoScaler:
    """Test suite for AutoScaler class."""

    @pytest.fixture
    def auto_scaler(self):
        """Create an AutoScaler instance."""
        return AutoScaler(min_instances=2, max_instances=10, target_utilization=0.7)

    @pytest.mark.unit
    def test_auto_scaler_creation(self):
        """Test creating AutoScaler."""
        scaler = AutoScaler(min_instances=1, max_instances=10)

        assert scaler.min_instances == 1
        assert scaler.max_instances == 10
        assert scaler.target_utilization == 0.7
        assert scaler.scale_up_threshold == 0.8
        assert scaler.scale_down_threshold == 0.5

    @pytest.mark.unit
    def test_auto_scaler_creation_invalid_min(self):
        """Test creating AutoScaler with invalid min_instances."""
        with pytest.raises(ValueError, match="min_instances must be at least 1"):
            AutoScaler(min_instances=0, max_instances=10)

    @pytest.mark.unit
    def test_auto_scaler_creation_invalid_max(self):
        """Test creating AutoScaler with max < min."""
        with pytest.raises(ValueError, match="max_instances must be >= min_instances"):
            AutoScaler(min_instances=10, max_instances=5)

    @pytest.mark.unit
    def test_should_scale_up(self, auto_scaler):
        """Test scale-up decision."""
        # High utilization should trigger scale-up
        assert auto_scaler.should_scale_up(0.9) is True

        # Medium utilization should not trigger scale-up
        assert auto_scaler.should_scale_up(0.75) is False

        # Low utilization should not trigger scale-up
        assert auto_scaler.should_scale_up(0.5) is False

    @pytest.mark.unit
    def test_should_scale_up_cooldown(self, auto_scaler):
        """Test scale-up with cooldown period."""
        auto_scaler.last_scale_action_time = time.time()

        # Should not scale up immediately after previous action
        assert auto_scaler.should_scale_up(0.9) is False

    @pytest.mark.unit
    def test_should_scale_down(self, auto_scaler):
        """Test scale-down decision."""
        # Low utilization should trigger scale-down
        assert auto_scaler.should_scale_down(0.4) is True

        # Medium utilization should not trigger scale-down
        assert auto_scaler.should_scale_down(0.6) is False

        # High utilization should not trigger scale-down
        assert auto_scaler.should_scale_down(0.9) is False

    @pytest.mark.unit
    def test_calculate_desired_instances(self, auto_scaler):
        """Test calculating desired instances."""
        metrics = ScalingMetrics(
            cpu_utilization=0.8,
            memory_utilization=0.7,
            request_rate=100.0,
            error_rate=0.05,
            queue_depth=10,
            response_time_p50=0.1,
            response_time_p95=0.5,
            response_time_p99=1.0,
        )

        desired = auto_scaler.calculate_desired_instances(current_instances=5, metrics=metrics)

        assert auto_scaler.min_instances <= desired <= auto_scaler.max_instances
        assert isinstance(desired, int)

    @pytest.mark.unit
    def test_calculate_desired_instances_high_queue(self, auto_scaler):
        """Test calculating desired instances with high queue depth."""
        metrics = ScalingMetrics(
            cpu_utilization=0.5,
            memory_utilization=0.5,
            request_rate=50.0,
            error_rate=0.02,
            queue_depth=200,  # High queue
            response_time_p50=0.1,
            response_time_p95=0.5,
            response_time_p99=1.0,
        )

        desired = auto_scaler.calculate_desired_instances(current_instances=5, metrics=metrics)

        # Should increase instances due to high queue
        assert desired >= 5

    @pytest.mark.unit
    def test_calculate_desired_instances_high_error_rate(self, auto_scaler):
        """Test calculating desired instances with high error rate."""
        metrics = ScalingMetrics(
            cpu_utilization=0.5,
            memory_utilization=0.5,
            request_rate=50.0,
            error_rate=0.2,  # High error rate
            queue_depth=10,
            response_time_p50=0.1,
            response_time_p95=0.5,
            response_time_p99=1.0,
        )

        desired = auto_scaler.calculate_desired_instances(current_instances=5, metrics=metrics)

        # Should increase instances due to high error rate
        assert desired >= 5

    @pytest.mark.unit
    def test_calculate_desired_instances_clamped(self, auto_scaler):
        """Test desired instances are clamped to min/max."""
        metrics = ScalingMetrics(
            cpu_utilization=0.1,
            memory_utilization=0.1,
            request_rate=10.0,
            error_rate=0.01,
            queue_depth=0,
            response_time_p50=0.1,
            response_time_p95=0.5,
            response_time_p99=1.0,
        )

        # Very low utilization, should scale down
        desired = auto_scaler.calculate_desired_instances(current_instances=5, metrics=metrics)
        assert desired >= auto_scaler.min_instances

        # Very high utilization, should scale up
        metrics.cpu_utilization = 1.0
        metrics.memory_utilization = 1.0
        desired = auto_scaler.calculate_desired_instances(current_instances=100, metrics=metrics)
        assert desired <= auto_scaler.max_instances

    @pytest.mark.unit
    def test_get_scaling_decision_no_action(self, auto_scaler):
        """Test scaling decision with no action needed."""
        metrics = ScalingMetrics(
            cpu_utilization=0.7,
            memory_utilization=0.7,
            request_rate=100.0,
            error_rate=0.05,
            queue_depth=10,
            response_time_p50=0.1,
            response_time_p95=0.5,
            response_time_p99=1.0,
        )

        decision = auto_scaler.get_scaling_decision(current_instances=5, metrics=metrics)

        assert decision["action"] == "no_action"
        assert decision["current_instances"] == 5

    @pytest.mark.unit
    def test_get_scaling_decision_scale_up(self, auto_scaler):
        """Test scaling decision for scale-up."""
        auto_scaler.last_scale_action_time = 0  # Reset cooldown
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

        decision = auto_scaler.get_scaling_decision(current_instances=5, metrics=metrics)

        assert decision["action"] == "scale_up"
        assert decision["desired_instances"] > decision["current_instances"]
        assert "scale_by" in decision

    @pytest.mark.unit
    def test_get_scaling_decision_scale_down(self, auto_scaler):
        """Test scaling decision for scale-down."""
        auto_scaler.last_scale_action_time = 0  # Reset cooldown
        metrics = ScalingMetrics(
            cpu_utilization=0.3,
            memory_utilization=0.4,
            request_rate=20.0,
            error_rate=0.01,
            queue_depth=0,
            response_time_p50=0.05,
            response_time_p95=0.2,
            response_time_p99=0.5,
        )

        decision = auto_scaler.get_scaling_decision(current_instances=10, metrics=metrics)

        assert decision["action"] == "scale_down"
        assert decision["desired_instances"] < decision["current_instances"]
        assert "scale_by" in decision

    @pytest.mark.unit
    def test_get_scaling_history(self, auto_scaler):
        """Test getting scaling history."""
        metrics = ScalingMetrics(
            cpu_utilization=0.7,
            memory_utilization=0.7,
            request_rate=100.0,
            error_rate=0.05,
            queue_depth=10,
            response_time_p50=0.1,
            response_time_p95=0.5,
            response_time_p99=1.0,
        )

        # Generate some decisions
        for _ in range(5):
            auto_scaler.get_scaling_decision(current_instances=5, metrics=metrics)

        history = auto_scaler.get_scaling_history(limit=3)
        assert len(history) <= 3

        full_history = auto_scaler.get_scaling_history()
        assert len(full_history) <= 10  # Default limit
