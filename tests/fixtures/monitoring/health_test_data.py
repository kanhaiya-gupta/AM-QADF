"""
Test fixtures for health test data.

Provides generators for health status and health metrics.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from am_qadf.monitoring.health_monitor import HealthStatus


def generate_health_metrics(component_type: str = "system", healthy: bool = True) -> Dict[str, float]:
    """
    Generate health metrics for a component.

    Args:
        component_type: Type of component ('system', 'process', 'custom')
        healthy: Whether component is healthy (affects metric values)

    Returns:
        Dictionary of health metrics
    """
    if component_type == "system":
        if healthy:
            metrics = {
                "cpu_percent": np.random.uniform(20.0, 60.0),
                "memory_percent": np.random.uniform(30.0, 70.0),
                "disk_percent": np.random.uniform(40.0, 80.0),
            }
        else:
            metrics = {
                "cpu_percent": np.random.uniform(85.0, 95.0),
                "memory_percent": np.random.uniform(85.0, 95.0),
                "disk_percent": np.random.uniform(85.0, 95.0),
            }

    elif component_type == "process":
        if healthy:
            metrics = {
                "cpu_percent": np.random.uniform(10.0, 50.0),
                "memory_percent": np.random.uniform(20.0, 60.0),
                "error_rate": np.random.uniform(0.0, 0.05),
                "latency_ms": np.random.uniform(10.0, 100.0),
            }
        else:
            metrics = {
                "cpu_percent": np.random.uniform(80.0, 95.0),
                "memory_percent": np.random.uniform(85.0, 95.0),
                "error_rate": np.random.uniform(0.2, 0.5),
                "latency_ms": np.random.uniform(500.0, 1000.0),
            }

    else:  # custom
        if healthy:
            metrics = {
                "metric1": np.random.uniform(0.0, 50.0),
                "metric2": np.random.uniform(0.0, 50.0),
                "metric3": np.random.uniform(0.0, 50.0),
            }
        else:
            metrics = {
                "metric1": np.random.uniform(80.0, 100.0),
                "metric2": np.random.uniform(80.0, 100.0),
                "metric3": np.random.uniform(80.0, 100.0),
            }

    return metrics


def generate_health_status(
    component_name: str = "system",
    status: str = None,
    healthy: bool = True,
    timestamp: datetime = None,
    component_type: str = "system",
) -> HealthStatus:
    """
    Generate a health status object.

    Args:
        component_name: Component name
        status: Optional status ('healthy', 'degraded', 'unhealthy', 'critical')
                If None, inferred from healthy parameter
        healthy: Whether component is healthy (used if status is None)
        timestamp: Optional timestamp (defaults to now)
        component_type: Type of component for metric generation

    Returns:
        HealthStatus object
    """
    if timestamp is None:
        timestamp = datetime.now()

    # Determine status
    if status is None:
        if healthy:
            status = np.random.choice(["healthy", "degraded"], p=[0.8, 0.2])
        else:
            status = np.random.choice(["unhealthy", "critical"], p=[0.7, 0.3])

    # Generate metrics
    metrics = generate_health_metrics(component_type, healthy)

    # Calculate health score based on metrics
    if component_type == "system":
        # Normalize metrics (inverse for percentages)
        cpu_score = max(0.0, min(1.0, 1.0 - metrics["cpu_percent"] / 100.0))
        memory_score = max(0.0, min(1.0, 1.0 - metrics["memory_percent"] / 100.0))
        disk_score = max(0.0, min(1.0, 1.0 - metrics["disk_percent"] / 100.0))
        health_score = (cpu_score + memory_score + disk_score) / 3.0
    elif component_type == "process":
        cpu_score = max(0.0, min(1.0, 1.0 - metrics["cpu_percent"] / 100.0))
        memory_score = max(0.0, min(1.0, 1.0 - metrics["memory_percent"] / 100.0))
        error_score = max(0.0, min(1.0, 1.0 - metrics["error_rate"] * 10))
        latency_score = max(0.0, min(1.0, 1.0 - metrics["latency_ms"] / 1000.0))
        health_score = (cpu_score + memory_score + error_score + latency_score) / 4.0
    else:
        # Default score
        health_score = 0.9 if healthy else 0.3

    # Identify issues
    issues = []
    if "cpu_percent" in metrics and metrics["cpu_percent"] > 90:
        issues.append(f"High CPU usage: {metrics['cpu_percent']:.1f}%")
    if "memory_percent" in metrics and metrics["memory_percent"] > 90:
        issues.append(f"High memory usage: {metrics['memory_percent']:.1f}%")
    if "error_rate" in metrics and metrics["error_rate"] > 0.1:
        issues.append(f"High error rate: {metrics['error_rate']:.4f}")
    if "latency_ms" in metrics and metrics["latency_ms"] > 500:
        issues.append(f"High latency: {metrics['latency_ms']:.1f}ms")

    return HealthStatus(
        component_name=component_name,
        status=status,
        health_score=health_score,
        timestamp=timestamp,
        metrics=metrics,
        issues=issues,
        metadata={
            "component_type": component_type,
            "test_data": True,
        },
    )


def generate_health_history(
    component_name: str = "system",
    duration_hours: float = 24.0,
    check_interval_minutes: float = 5.0,
    start_healthy: bool = True,
    degrade_at_hour: Optional[float] = None,
) -> List[HealthStatus]:
    """
    Generate health history over time period.

    Args:
        component_name: Component name
        duration_hours: Duration in hours
        check_interval_minutes: Interval between health checks (minutes)
        start_healthy: Whether component starts healthy
        degrade_at_hour: Optional hour at which component degrades (None = stays same)

    Returns:
        List of HealthStatus objects over time period
    """
    start_time = datetime.now() - timedelta(hours=duration_hours)
    n_checks = int((duration_hours * 60.0) / check_interval_minutes)
    interval_seconds = check_interval_minutes * 60.0

    health_history = []
    current_healthy = start_healthy

    for i in range(n_checks):
        timestamp = start_time + timedelta(seconds=i * interval_seconds)

        # Check if component should degrade
        current_hour = i * check_interval_minutes / 60.0
        if degrade_at_hour is not None and current_hour >= degrade_at_hour:
            current_healthy = False

        health_status = generate_health_status(
            component_name=component_name,
            healthy=current_healthy,
            timestamp=timestamp,
            component_type="system" if component_name == "system" else "process",
        )

        health_history.append(health_status)

    return health_history


def generate_health_degradation_scenario(
    component_name: str = "component1", duration_hours: float = 1.0
) -> List[HealthStatus]:
    """
    Generate health degradation scenario (healthy -> degraded -> unhealthy).

    Args:
        component_name: Component name
        duration_hours: Duration of scenario

    Returns:
        List of HealthStatus objects showing degradation
    """
    start_time = datetime.now() - timedelta(hours=duration_hours)
    n_checks = 20
    interval_seconds = (duration_hours * 3600.0) / n_checks

    health_history = []

    for i in range(n_checks):
        timestamp = start_time + timedelta(seconds=i * interval_seconds)

        # Progressive degradation
        progress = i / n_checks
        if progress < 0.3:
            # Healthy phase
            healthy = True
            status = "healthy"
        elif progress < 0.6:
            # Degraded phase
            healthy = True
            status = "degraded"
        elif progress < 0.9:
            # Unhealthy phase
            healthy = False
            status = "unhealthy"
        else:
            # Critical phase
            healthy = False
            status = "critical"

        health_status = generate_health_status(
            component_name=component_name, status=status, healthy=healthy, timestamp=timestamp
        )

        health_history.append(health_status)

    return health_history
