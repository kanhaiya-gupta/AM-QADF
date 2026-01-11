"""
Scalability and load balancing utilities.

This module provides utilities for horizontal and vertical scaling,
load balancing, and resource distribution.
"""

import time
import random
from dataclasses import dataclass
from typing import List, Dict, Optional, Callable, Any
from collections import defaultdict
from enum import Enum


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""

    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    IP_HASH = "ip_hash"


@dataclass
class ScalingMetrics:
    """Metrics for scaling decisions."""

    cpu_utilization: float
    memory_utilization: float
    request_rate: float
    error_rate: float
    queue_depth: int
    response_time_p50: float
    response_time_p95: float
    response_time_p99: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "cpu_utilization": self.cpu_utilization,
            "memory_utilization": self.memory_utilization,
            "request_rate": self.request_rate,
            "error_rate": self.error_rate,
            "queue_depth": self.queue_depth,
            "response_time_p50": self.response_time_p50,
            "response_time_p95": self.response_time_p95,
            "response_time_p99": self.response_time_p99,
        }


@dataclass
class WorkerStatus:
    """Worker status information."""

    worker_id: str
    status: str  # 'healthy', 'unhealthy', 'draining', 'down'
    active_connections: int
    total_requests: int
    error_count: int
    last_health_check: float
    weight: float = 1.0  # For weighted round-robin

    def is_available(self) -> bool:
        """Check if worker is available for requests."""
        return self.status in ["healthy", "draining"]

    def get_health_score(self) -> float:
        """Calculate health score (0.0 to 1.0)."""
        if self.status == "down":
            return 0.0
        elif self.status == "unhealthy":
            return 0.3
        elif self.status == "draining":
            return 0.5
        else:  # healthy
            # Factor in error rate
            if self.total_requests > 0:
                error_rate = self.error_count / self.total_requests
                return max(0.5, 1.0 - error_rate)
            return 1.0


class LoadBalancer:
    """Load balancer for distributing workloads."""

    def __init__(self, strategy: str = "round_robin"):
        """
        Initialize load balancer.

        Args:
            strategy: Load balancing strategy ('round_robin', 'least_connections',
                    'weighted_round_robin', 'ip_hash')
        """
        try:
            self.strategy = LoadBalancingStrategy(strategy)
        except ValueError:
            raise ValueError(
                f"Invalid load balancing strategy: {strategy}. " f"Must be one of: {[s.value for s in LoadBalancingStrategy]}"
            )

        self.workers: Dict[str, WorkerStatus] = {}
        self.round_robin_index = 0
        self.worker_weights: Dict[str, float] = {}

    def register_worker(self, worker_id: str, weight: float = 1.0):
        """Register a worker with the load balancer."""
        if worker_id not in self.workers:
            self.workers[worker_id] = WorkerStatus(
                worker_id=worker_id,
                status="healthy",
                active_connections=0,
                total_requests=0,
                error_count=0,
                last_health_check=time.time(),
                weight=weight,
            )
            self.worker_weights[worker_id] = weight
        else:
            # Update weight if worker already exists
            self.workers[worker_id].weight = weight
            self.worker_weights[worker_id] = weight

    def unregister_worker(self, worker_id: str):
        """Unregister a worker from the load balancer."""
        if worker_id in self.workers:
            del self.workers[worker_id]
            if worker_id in self.worker_weights:
                del self.worker_weights[worker_id]

    def update_worker_status(self, worker_id: str, status: str):
        """Update worker health status."""
        if worker_id in self.workers:
            self.workers[worker_id].status = status
            self.workers[worker_id].last_health_check = time.time()

    def record_request(self, worker_id: str, success: bool = True):
        """Record a request to a worker."""
        if worker_id in self.workers:
            worker = self.workers[worker_id]
            worker.total_requests += 1
            if not success:
                worker.error_count += 1

    def record_connection(self, worker_id: str, connected: bool = True):
        """Record a connection to a worker."""
        if worker_id in self.workers:
            if connected:
                self.workers[worker_id].active_connections += 1
            else:
                self.workers[worker_id].active_connections = max(0, self.workers[worker_id].active_connections - 1)

    def select_worker(self, workers: Optional[List[str]] = None, client_ip: Optional[str] = None) -> Optional[str]:
        """
        Select worker for next request.

        Args:
            workers: Optional list of worker IDs to choose from. If None, uses all registered workers.
            client_ip: Optional client IP for IP hash strategy.

        Returns:
            Selected worker ID or None if no workers available
        """
        if workers is None:
            workers = list(self.workers.keys())

        # Filter to only available workers
        available_workers = [w for w in workers if w in self.workers and self.workers[w].is_available()]

        if not available_workers:
            return None

        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            worker = available_workers[self.round_robin_index % len(available_workers)]
            self.round_robin_index += 1
            return worker

        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            # Select worker with fewest active connections
            worker = min(available_workers, key=lambda w: self.workers[w].active_connections)
            return worker

        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            # Weighted round-robin based on worker weights
            total_weight = sum(self.worker_weights.get(w, 1.0) for w in available_workers)
            if total_weight == 0:
                return available_workers[0]

            # Simple weighted selection (can be improved with more sophisticated algorithm)
            r = random.uniform(0, total_weight)
            cumulative = 0.0
            for worker in available_workers:
                cumulative += self.worker_weights.get(worker, 1.0)
                if r <= cumulative:
                    return worker
            return available_workers[-1]

        elif self.strategy == LoadBalancingStrategy.IP_HASH:
            # Hash client IP to select worker
            if client_ip is None:
                # Fallback to round-robin if no IP provided
                worker = available_workers[self.round_robin_index % len(available_workers)]
                self.round_robin_index += 1
                return worker

            # Simple hash-based selection
            hash_value = hash(client_ip)
            index = abs(hash_value) % len(available_workers)
            return available_workers[index]

        else:
            # Default to round-robin
            worker = available_workers[self.round_robin_index % len(available_workers)]
            self.round_robin_index += 1
            return worker

    def get_worker_status(self, worker_id: str) -> Optional[WorkerStatus]:
        """Get status of a specific worker."""
        return self.workers.get(worker_id)

    def get_all_workers_status(self) -> Dict[str, WorkerStatus]:
        """Get status of all workers."""
        return self.workers.copy()

    def get_healthy_workers(self) -> List[str]:
        """Get list of healthy worker IDs."""
        return [worker_id for worker_id, worker in self.workers.items() if worker.status == "healthy"]


class AutoScaler:
    """Auto-scaling utility based on metrics."""

    def __init__(
        self,
        min_instances: int,
        max_instances: int,
        target_utilization: float = 0.7,
        scale_up_threshold: float = 0.8,
        scale_down_threshold: float = 0.5,
        cooldown_period: float = 300.0,
    ):
        """
        Initialize auto-scaler.

        Args:
            min_instances: Minimum number of instances
            max_instances: Maximum number of instances
            target_utilization: Target utilization (0.0-1.0)
            scale_up_threshold: Utilization threshold for scaling up (0.0-1.0)
            scale_down_threshold: Utilization threshold for scaling down (0.0-1.0)
            cooldown_period: Cooldown period in seconds between scaling actions
        """
        if min_instances < 1:
            raise ValueError("min_instances must be at least 1")
        if max_instances < min_instances:
            raise ValueError("max_instances must be >= min_instances")
        if not 0.0 <= target_utilization <= 1.0:
            raise ValueError("target_utilization must be between 0.0 and 1.0")
        if not 0.0 <= scale_up_threshold <= 1.0:
            raise ValueError("scale_up_threshold must be between 0.0 and 1.0")
        if not 0.0 <= scale_down_threshold <= 1.0:
            raise ValueError("scale_down_threshold must be between 0.0 and 1.0")

        self.min_instances = min_instances
        self.max_instances = max_instances
        self.target_utilization = target_utilization
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.cooldown_period = cooldown_period

        self.last_scale_action_time = 0.0
        self.scaling_history: List[Dict[str, Any]] = []

    def should_scale_up(self, current_utilization: float) -> bool:
        """
        Determine if scaling up is needed.

        Args:
            current_utilization: Current utilization (0.0-1.0)

        Returns:
            True if scaling up is recommended
        """
        if current_utilization > self.scale_up_threshold:
            # Check cooldown period
            if time.time() - self.last_scale_action_time < self.cooldown_period:
                return False
            return True
        return False

    def should_scale_down(self, current_utilization: float) -> bool:
        """
        Determine if scaling down is needed.

        Args:
            current_utilization: Current utilization (0.0-1.0)

        Returns:
            True if scaling down is recommended
        """
        if current_utilization < self.scale_down_threshold:
            # Check cooldown period
            if time.time() - self.last_scale_action_time < self.cooldown_period:
                return False
            return True
        return False

    def calculate_desired_instances(self, current_instances: int, metrics: ScalingMetrics) -> int:
        """
        Calculate desired number of instances based on metrics.

        Args:
            current_instances: Current number of instances
            metrics: Scaling metrics

        Returns:
            Desired number of instances
        """
        # Use CPU and memory utilization (weighted average)
        avg_utilization = metrics.cpu_utilization * 0.6 + metrics.memory_utilization * 0.4

        # Calculate desired instances based on target utilization
        if avg_utilization > 0:
            desired_instances = int(current_instances * (avg_utilization / self.target_utilization))
        else:
            desired_instances = current_instances

        # Also consider request rate and queue depth
        if metrics.queue_depth > 100:  # High queue depth
            desired_instances = max(desired_instances, current_instances + 1)

        # Also consider error rate
        if metrics.error_rate > 0.1:  # High error rate might indicate overload
            desired_instances = max(desired_instances, current_instances + 1)

        # Clamp to min/max
        desired_instances = max(self.min_instances, min(self.max_instances, desired_instances))

        return desired_instances

    def get_scaling_decision(self, current_instances: int, metrics: ScalingMetrics) -> Dict[str, Any]:
        """
        Get scaling decision based on current state and metrics.

        Args:
            current_instances: Current number of instances
            metrics: Scaling metrics

        Returns:
            Dictionary with scaling decision information
        """
        avg_utilization = metrics.cpu_utilization * 0.6 + metrics.memory_utilization * 0.4

        desired_instances = self.calculate_desired_instances(current_instances, metrics)

        decision = {
            "current_instances": current_instances,
            "desired_instances": desired_instances,
            "avg_utilization": avg_utilization,
            "action": "no_action",
            "reason": "utilization within acceptable range",
        }

        if desired_instances > current_instances:
            if self.should_scale_up(avg_utilization):
                decision["action"] = "scale_up"
                decision["reason"] = (
                    f"utilization ({avg_utilization:.2%}) exceeds scale-up threshold ({self.scale_up_threshold:.2%})"
                )
                decision["scale_by"] = desired_instances - current_instances
                self.last_scale_action_time = time.time()
        elif desired_instances < current_instances:
            if self.should_scale_down(avg_utilization):
                decision["action"] = "scale_down"
                decision["reason"] = (
                    f"utilization ({avg_utilization:.2%}) below scale-down threshold ({self.scale_down_threshold:.2%})"
                )
                decision["scale_by"] = current_instances - desired_instances
                self.last_scale_action_time = time.time()

        # Record in history
        self.scaling_history.append({"timestamp": time.time(), "decision": decision.copy()})

        # Keep only last 100 decisions
        if len(self.scaling_history) > 100:
            self.scaling_history = self.scaling_history[-100:]

        return decision

    def get_scaling_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent scaling history."""
        return self.scaling_history[-limit:]
