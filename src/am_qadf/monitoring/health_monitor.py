"""
Health Monitor

System and process health monitoring.
Provides health checks, health scores, and health history tracking.
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
import logging
from datetime import datetime
import threading
import time

logger = logging.getLogger(__name__)

# Try to import psutil (optional dependency)
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None


@dataclass
class HealthStatus:
    """System or process health status."""

    component_name: str
    status: str  # 'healthy', 'degraded', 'unhealthy', 'critical'
    health_score: float  # 0.0-1.0
    timestamp: datetime
    metrics: Dict[str, float] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate health status."""
        if self.status not in ["healthy", "degraded", "unhealthy", "critical"]:
            raise ValueError(f"Invalid status: {self.status}")

        if not 0.0 <= self.health_score <= 1.0:
            raise ValueError(f"Health score must be between 0.0 and 1.0, got {self.health_score}")


class HealthMonitor:
    """
    System and process health monitoring.

    Provides:
    - System health checks (CPU, memory, disk, network)
    - Process health monitoring
    - Health score calculation
    - Health history tracking
    - Custom health check registration
    """

    def __init__(self, check_interval_seconds: float = 5.0):
        """
        Initialize health monitor.

        Args:
            check_interval_seconds: Interval between health checks (seconds)
        """
        self.check_interval_seconds = check_interval_seconds
        self._is_monitoring = False
        self._monitoring_thread: Optional[threading.Thread] = None

        # Registered components and their health checkers
        self._components: Dict[str, Callable] = {}

        # Health history
        self._health_history: Dict[str, List[HealthStatus]] = {}

        # Thread safety
        self._lock = threading.Lock()

        logger.info("HealthMonitor initialized")

    def start_monitoring(self) -> None:
        """Start health monitoring."""
        if self._is_monitoring:
            logger.warning("Health monitoring is already running")
            return

        self._is_monitoring = True

        def monitoring_loop():
            """Health monitoring loop running in separate thread."""
            while self._is_monitoring:
                try:
                    # Check system health
                    system_health = self.check_system_health()
                    self._record_health("system", system_health)

                    # Check registered components
                    for component_name in self._components.keys():
                        try:
                            component_health = self.check_process_health(component_name)
                            self._record_health(component_name, component_health)
                        except Exception as e:
                            logger.error(f"Error checking health for {component_name}: {e}")

                    time.sleep(self.check_interval_seconds)

                except Exception as e:
                    logger.error(f"Error in health monitoring loop: {e}")
                    time.sleep(self.check_interval_seconds)

        self._monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self._monitoring_thread.start()

        logger.info("Health monitoring started")

    def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        if not self._is_monitoring:
            logger.warning("Health monitoring is not running")
            return

        self._is_monitoring = False

        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5.0)

        logger.info("Health monitoring stopped")

    def register_component(self, component_name: str, health_checker: Callable) -> None:
        """
        Register component for health monitoring.

        Args:
            component_name: Component name
            health_checker: Callable that returns health metrics dictionary
        """
        self._components[component_name] = health_checker
        logger.info(f"Registered component for health monitoring: {component_name}")

    def check_system_health(self) -> HealthStatus:
        """
        Check overall system health.

        Returns:
            HealthStatus object for system
        """
        if not PSUTIL_AVAILABLE:
            logger.warning("psutil not available, cannot check system health. Install with: pip install psutil")
            return HealthStatus(
                component_name="system",
                status="unhealthy",
                health_score=0.0,
                timestamp=datetime.now(),
                issues=["psutil not available"],
                metadata={"error": "psutil not available"},
            )

        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            # Calculate network I/O (bytes sent/received per second)
            try:
                net_io = psutil.net_io_counters()
                net_metrics = {
                    "bytes_sent": net_io.bytes_sent,
                    "bytes_recv": net_io.bytes_recv,
                    "packets_sent": net_io.packets_sent,
                    "packets_recv": net_io.packets_recv,
                }
            except Exception:
                net_metrics = {}

            metrics = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "memory_total_gb": memory.total / (1024**3),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024**3),
                "disk_total_gb": disk.total / (1024**3),
                **net_metrics,
            }

            # Identify issues
            issues = []
            if cpu_percent > 90:
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
            if memory.percent > 90:
                issues.append(f"High memory usage: {memory.percent:.1f}%")
            if disk.percent > 90:
                issues.append(f"High disk usage: {disk.percent:.1f}%")

            # Calculate health score
            health_score = self._calculate_health_score(
                metrics,
                {
                    "cpu_percent": 0.3,
                    "memory_percent": 0.3,
                    "disk_percent": 0.2,
                },
            )

            # Determine status
            status = self._get_status_from_score(health_score, len(issues))

            return HealthStatus(
                component_name="system",
                status=status,
                health_score=health_score,
                timestamp=datetime.now(),
                metrics=metrics,
                issues=issues,
                metadata={"check_type": "system"},
            )

        except Exception as e:
            logger.error(f"Error checking system health: {e}")
            return HealthStatus(
                component_name="system",
                status="unhealthy",
                health_score=0.0,
                timestamp=datetime.now(),
                issues=[f"Health check error: {str(e)}"],
                metadata={"error": str(e)},
            )

    def check_process_health(self, process_name: str) -> HealthStatus:
        """
        Check specific process health.

        Args:
            process_name: Process/component name

        Returns:
            HealthStatus object for process
        """
        try:
            # If component has custom health checker, use it
            if process_name in self._components:
                health_checker = self._components[process_name]
                metrics = health_checker()

                # Calculate health score from metrics
                health_score = self._calculate_health_score(metrics)

                # Identify issues (metric values outside normal range)
                issues = []
                for metric_name, value in metrics.items():
                    # Simple heuristic: flag if value is very high or very low
                    if isinstance(value, (int, float)):
                        if metric_name.endswith("_percent") and value > 90:
                            issues.append(f"High {metric_name}: {value:.1f}%")
                        elif metric_name.endswith("_error_rate") and value > 0.1:
                            issues.append(f"High {metric_name}: {value:.4f}")

                status = self._get_status_from_score(health_score, len(issues))

                return HealthStatus(
                    component_name=process_name,
                    status=status,
                    health_score=health_score,
                    timestamp=datetime.now(),
                    metrics=metrics,
                    issues=issues,
                    metadata={"check_type": "custom"},
                )

            # Default: try to find process by name
            # This is a simplified implementation
            return HealthStatus(
                component_name=process_name,
                status="healthy",
                health_score=0.8,  # Default score if no custom checker
                timestamp=datetime.now(),
                metrics={},
                issues=[],
                metadata={"check_type": "default", "note": "No custom health checker registered"},
            )

        except Exception as e:
            logger.error(f"Error checking process health for {process_name}: {e}")
            return HealthStatus(
                component_name=process_name,
                status="unhealthy",
                health_score=0.0,
                timestamp=datetime.now(),
                issues=[f"Health check error: {str(e)}"],
                metadata={"error": str(e)},
            )

    def get_health_history(self, component_name: str, start_time: datetime, end_time: datetime) -> List[HealthStatus]:
        """
        Get health history for component.

        Args:
            component_name: Component name
            start_time: Start time for history
            end_time: End time for history

        Returns:
            List of HealthStatus objects in time range
        """
        with self._lock:
            if component_name not in self._health_history:
                return []

            history = [health for health in self._health_history[component_name] if start_time <= health.timestamp <= end_time]

            return history

    def get_all_component_health(self) -> Dict[str, HealthStatus]:
        """
        Get current health status for all registered components.

        Returns:
            Dictionary mapping component names to HealthStatus objects
        """
        health_statuses = {}

        # System health
        health_statuses["system"] = self.check_system_health()

        # Component health
        for component_name in self._components.keys():
            try:
                health_statuses[component_name] = self.check_process_health(component_name)
            except Exception as e:
                logger.error(f"Error getting health for {component_name}: {e}")

        return health_statuses

    def calculate_health_score(self, metrics: Dict[str, float], weights: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate overall health score from metrics.

        Args:
            metrics: Dictionary of metric name -> value
            weights: Optional weights for metrics (default: equal weights)

        Returns:
            Health score between 0.0 and 1.0
        """
        return self._calculate_health_score(metrics, weights)

    def _calculate_health_score(self, metrics: Dict[str, float], weights: Optional[Dict[str, float]] = None) -> float:
        """Internal method to calculate health score."""
        if not metrics:
            return 0.5  # Default neutral score

        # Normalize metrics to 0-1 range
        normalized_scores = {}

        for metric_name, value in metrics.items():
            if not isinstance(value, (int, float)):
                continue

            # Normalize based on metric type
            if metric_name.endswith("_percent"):
                # Percentages: 0-100 -> 1.0-0.0 (inverse, lower is better)
                normalized_scores[metric_name] = max(0.0, min(1.0, 1.0 - value / 100.0))
            elif metric_name.endswith("_error_rate"):
                # Error rates: higher is worse
                normalized_scores[metric_name] = max(0.0, min(1.0, 1.0 - value))
            elif metric_name.endswith("_latency_ms"):
                # Latency: normalize assuming reasonable max (e.g., 1000ms)
                normalized_scores[metric_name] = max(0.0, min(1.0, 1.0 - value / 1000.0))
            else:
                # Default: assume value is already normalized or use a default
                normalized_scores[metric_name] = 0.8

        # Calculate weighted average
        if weights:
            # Filter to only weights that exist in metrics
            filtered_weights = {k: v for k, v in weights.items() if k in normalized_scores}
            if filtered_weights:
                total_weight = sum(filtered_weights.values())
                if total_weight > 0:
                    score = sum(normalized_scores[k] * v for k, v in filtered_weights.items()) / total_weight
                    return max(0.0, min(1.0, score))

        # Equal weights if no weights specified
        if normalized_scores:
            return sum(normalized_scores.values()) / len(normalized_scores)

        return 0.5

    def _get_status_from_score(self, health_score: float, issue_count: int) -> str:
        """Get status string from health score and issue count."""
        if health_score >= 0.8 and issue_count == 0:
            return "healthy"
        elif health_score >= 0.6:
            return "degraded"
        elif health_score >= 0.3:
            return "unhealthy"
        else:
            return "critical"

    def _record_health(self, component_name: str, health_status: HealthStatus) -> None:
        """Record health status in history."""
        with self._lock:
            if component_name not in self._health_history:
                self._health_history[component_name] = []

            self._health_history[component_name].append(health_status)

            # Keep only last 1000 records per component
            if len(self._health_history[component_name]) > 1000:
                self._health_history[component_name] = self._health_history[component_name][-1000:]
