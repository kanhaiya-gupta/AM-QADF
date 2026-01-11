"""
Resource monitoring utilities.

This module provides system and process resource monitoring capabilities,
including CPU, memory, disk, and network usage tracking.
"""

import os
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Callable, Tuple, Dict, Any
from collections import deque

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


@dataclass
class ResourceMetrics:
    """Resource utilization metrics."""

    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_percent: float
    disk_used_gb: float
    disk_available_gb: float
    network_sent_mb: float
    network_recv_mb: float
    process_count: int
    thread_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "memory_used_mb": self.memory_used_mb,
            "memory_available_mb": self.memory_available_mb,
            "disk_percent": self.disk_percent,
            "disk_used_gb": self.disk_used_gb,
            "disk_available_gb": self.disk_available_gb,
            "network_sent_mb": self.network_sent_mb,
            "network_recv_mb": self.network_recv_mb,
            "process_count": self.process_count,
            "thread_count": self.thread_count,
        }


class ResourceMonitor:
    """Monitor system and process resources."""

    def __init__(self, update_interval: float = 5.0, history_size: int = 1000):
        """
        Initialize resource monitor.

        Args:
            update_interval: Interval between metric collections in seconds
            history_size: Maximum number of metrics to keep in history
        """
        if not PSUTIL_AVAILABLE:
            raise ImportError("psutil is required for resource monitoring. Install with: pip install psutil")

        self.update_interval = update_interval
        self.history_size = history_size
        self.metrics_history: deque = deque(maxlen=history_size)
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.callback: Optional[Callable[[ResourceMetrics], None]] = None

        # Network I/O tracking
        self.last_net_io = psutil.net_io_counters()
        self.last_net_io_time = time.time()

    def get_system_metrics(self) -> ResourceMetrics:
        """Get current system resource metrics."""
        if not PSUTIL_AVAILABLE:
            # Return dummy metrics if psutil not available
            return ResourceMetrics(
                timestamp=datetime.now(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_used_mb=0.0,
                memory_available_mb=0.0,
                disk_percent=0.0,
                disk_used_gb=0.0,
                disk_available_gb=0.0,
                network_sent_mb=0.0,
                network_recv_mb=0.0,
                process_count=0,
                thread_count=0,
            )

        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)

        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_mb = memory.used / (1024 * 1024)
        memory_available_mb = memory.available / (1024 * 1024)

        # Disk usage (root partition)
        disk = psutil.disk_usage("/")
        disk_percent = disk.percent
        disk_used_gb = disk.used / (1024 * 1024 * 1024)
        disk_available_gb = disk.free / (1024 * 1024 * 1024)

        # Network I/O (calculate delta since last call)
        current_net_io = psutil.net_io_counters()
        current_time = time.time()
        time_delta = current_time - self.last_net_io_time

        if time_delta > 0:
            network_sent_mb = (current_net_io.bytes_sent - self.last_net_io.bytes_sent) / (1024 * 1024) / time_delta
            network_recv_mb = (current_net_io.bytes_recv - self.last_net_io.bytes_recv) / (1024 * 1024) / time_delta
        else:
            network_sent_mb = 0.0
            network_recv_mb = 0.0

        self.last_net_io = current_net_io
        self.last_net_io_time = current_time

        # Process and thread counts
        process_count = len(psutil.pids())
        thread_count = threading.active_count()

        metrics = ResourceMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_mb=memory_used_mb,
            memory_available_mb=memory_available_mb,
            disk_percent=disk_percent,
            disk_used_gb=disk_used_gb,
            disk_available_gb=disk_available_gb,
            network_sent_mb=network_sent_mb,
            network_recv_mb=network_recv_mb,
            process_count=process_count,
            thread_count=thread_count,
        )

        # Add to history
        self.metrics_history.append(metrics)

        return metrics

    def get_process_metrics(self, pid: Optional[int] = None) -> ResourceMetrics:
        """Get current process resource metrics."""
        if not PSUTIL_AVAILABLE:
            return self.get_system_metrics()  # Fallback to system metrics

        if pid is None:
            pid = os.getpid()

        try:
            process = psutil.Process(pid)

            # CPU usage
            cpu_percent = process.cpu_percent(interval=0.1)

            # Memory usage
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            memory_used_mb = memory_info.rss / (1024 * 1024)

            # Get system memory for available calculation
            system_memory = psutil.virtual_memory()
            memory_available_mb = system_memory.available / (1024 * 1024)

            # Disk usage (same as system for process-level)
            disk = psutil.disk_usage("/")
            disk_percent = disk.percent
            disk_used_gb = disk.used / (1024 * 1024 * 1024)
            disk_available_gb = disk.free / (1024 * 1024 * 1024)

            # Network I/O (process-level network connections)
            connections = process.connections()
            network_sent_mb = 0.0
            network_recv_mb = 0.0
            # Note: Process-level network I/O requires more complex tracking

            # Process and thread counts (for this process)
            thread_count = process.num_threads()
            process_count = 1

            metrics = ResourceMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_mb=memory_used_mb,
                memory_available_mb=memory_available_mb,
                disk_percent=disk_percent,
                disk_used_gb=disk_used_gb,
                disk_available_gb=disk_available_gb,
                network_sent_mb=network_sent_mb,
                network_recv_mb=network_recv_mb,
                process_count=process_count,
                thread_count=thread_count,
            )

            return metrics

        except (psutil.NoSuchProcess, psutil.AccessDenied):
            # Process not found or access denied, return system metrics
            return self.get_system_metrics()

    def _monitoring_loop(self):
        """Internal monitoring loop for continuous monitoring."""
        while self.monitoring_active:
            try:
                metrics = self.get_system_metrics()
                if self.callback:
                    self.callback(metrics)
            except Exception as e:
                # Log error but continue monitoring
                print(f"Error in monitoring loop: {e}")

            time.sleep(self.update_interval)

    def start_monitoring(self, callback: Callable[[ResourceMetrics], None]):
        """Start continuous monitoring with callback."""
        if self.monitoring_active:
            raise RuntimeError("Monitoring is already active")

        self.callback = callback
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()

    def stop_monitoring(self):
        """Stop continuous monitoring."""
        if not self.monitoring_active:
            return

        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=self.update_interval * 2)
        self.callback = None

    def get_metrics_history(self, duration: Optional[float] = None) -> List[ResourceMetrics]:
        """
        Get metrics history for specified duration.

        Args:
            duration: Duration in seconds. If None, return all history.

        Returns:
            List of ResourceMetrics within the specified duration
        """
        if duration is None:
            return list(self.metrics_history)

        cutoff_time = datetime.now().timestamp() - duration
        return [m for m in self.metrics_history if m.timestamp.timestamp() >= cutoff_time]

    def check_resource_limits(
        self, cpu_threshold: float = 0.8, memory_threshold: float = 0.8, disk_threshold: float = 0.9
    ) -> Tuple[bool, List[str]]:
        """
        Check if resource usage exceeds thresholds.

        Args:
            cpu_threshold: CPU usage threshold (0.0-1.0)
            memory_threshold: Memory usage threshold (0.0-1.0)
            disk_threshold: Disk usage threshold (0.0-1.0)

        Returns:
            Tuple of (exceeded: bool, warnings: List[str])
        """
        metrics = self.get_system_metrics()
        warnings = []
        exceeded = False

        if metrics.cpu_percent / 100.0 > cpu_threshold:
            warnings.append(f"CPU usage ({metrics.cpu_percent:.1f}%) exceeds threshold ({cpu_threshold*100:.1f}%)")
            exceeded = True

        if metrics.memory_percent / 100.0 > memory_threshold:
            warnings.append(f"Memory usage ({metrics.memory_percent:.1f}%) exceeds threshold ({memory_threshold*100:.1f}%)")
            exceeded = True

        if metrics.disk_percent / 100.0 > disk_threshold:
            warnings.append(f"Disk usage ({metrics.disk_percent:.1f}%) exceeds threshold ({disk_threshold*100:.1f}%)")
            exceeded = True

        return exceeded, warnings
