# Deployment Module API Reference

## Overview

The Deployment module provides production-ready deployment utilities for AM-QADF, enabling production configuration management, scalability, fault tolerance, resource monitoring, and performance tuning.

## ProductionConfig

Production configuration management class.

```python
from am_qadf.deployment import ProductionConfig

config = ProductionConfig(
    environment: str = 'production',
    log_level: str = 'INFO',
    enable_metrics: bool = True,
    enable_tracing: bool = True,
    enable_profiling: bool = False,
    database_pool_size: int = 20,
    database_max_overflow: int = 10,
    database_timeout: float = 30.0,
    redis_host: str = 'localhost',
    redis_port: int = 6379,
    redis_db: int = 0,
    redis_password: Optional[str] = None,
    kafka_bootstrap_servers: List[str] = None,
    metrics_port: int = 9090,
    health_check_port: int = 8080,
    worker_threads: int = 4,
    max_concurrent_requests: int = 100,
    request_timeout: float = 60.0,
    secrets_manager: str = 'env',
    secrets_path: Optional[str] = None,
    enable_experimental_features: bool = False,
)
```

### Fields

- `environment` (str): Environment name - 'development', 'staging', or 'production' (default: `'production'`)
- `log_level` (str): Logging level - 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL' (default: `'INFO'`)
- `enable_metrics` (bool): Enable metrics collection (default: `True`)
- `enable_tracing` (bool): Enable distributed tracing (default: `True`)
- `enable_profiling` (bool): Enable performance profiling (default: `False`)
- `database_pool_size` (int): Database connection pool size (default: `20`)
- `database_max_overflow` (int): Maximum overflow connections (default: `10`)
- `database_timeout` (float): Connection timeout in seconds (default: `30.0`)
- `redis_host` (str): Redis host address (default: `'localhost'`)
- `redis_port` (int): Redis port (default: `6379`)
- `worker_threads` (int): Number of worker threads (default: `4`)
- `max_concurrent_requests` (int): Maximum concurrent requests (default: `100`)
- `secrets_manager` (str): Secrets manager type - 'env', 'file', 'vault', 'aws_secrets' (default: `'env'`)

### Methods

#### `from_env() -> ProductionConfig`

Load configuration from environment variables.

**Returns**: `ProductionConfig` instance

**Example**:
```python
config = ProductionConfig.from_env()
```

---

#### `from_file(config_path: str) -> ProductionConfig`

Load configuration from YAML or JSON file.

**Parameters**:
- `config_path` (str): Path to configuration file (`.yaml`, `.yml`, or `.json`)

**Returns**: `ProductionConfig` instance

**Raises**:
- `FileNotFoundError`: If file does not exist
- `ValueError`: If file format is unsupported

**Example**:
```python
config = ProductionConfig.from_file('config/production.yaml')
```

---

#### `validate() -> Tuple[bool, List[str]]`

Validate configuration settings.

**Returns**: Tuple of (is_valid, list_of_errors)

**Example**:
```python
is_valid, errors = config.validate()
if not is_valid:
    for error in errors:
        print(f"Config error: {error}")
```

---

#### `get_secret(secret_name: str) -> Optional[str]`

Get secret value from secrets manager.

**Parameters**:
- `secret_name` (str): Secret name (will be prefixed with `AM_QADF_SECRET_` for env manager)

**Returns**: Secret value or None if not found

**Raises**:
- `NotImplementedError`: If secrets manager type is not implemented (Vault, AWS Secrets Manager)

**Example**:
```python
api_key = config.get_secret('api_key')
```

---

#### `to_dict() -> Dict[str, Any]`

Convert configuration to dictionary.

**Returns**: Configuration dictionary (sensitive fields masked with `'***'`)

**Example**:
```python
config_dict = config.to_dict()
```

---

## ResourceMonitor

System and process resource monitoring class.

```python
from am_qadf.deployment import ResourceMonitor

monitor = ResourceMonitor(
    update_interval: float = 5.0,
    history_size: int = 1000,
)
```

### Methods

#### `get_system_metrics() -> ResourceMetrics`

Get current system resource metrics.

**Returns**: `ResourceMetrics` object with CPU, memory, disk, network, process, and thread metrics

**Example**:
```python
metrics = monitor.get_system_metrics()
print(f"CPU: {metrics.cpu_percent}%, Memory: {metrics.memory_percent}%")
```

---

#### `get_process_metrics(pid: Optional[int] = None) -> ResourceMetrics`

Get current process resource metrics.

**Parameters**:
- `pid` (Optional[int]): Process ID. If None, uses current process.

**Returns**: `ResourceMetrics` object for the specified process

**Raises**:
- `psutil.NoSuchProcess`: If process not found (falls back to system metrics)

**Example**:
```python
metrics = monitor.get_process_metrics(pid=12345)
```

---

#### `start_monitoring(callback: Callable[[ResourceMetrics], None]) -> None`

Start continuous resource monitoring with callback.

**Parameters**:
- `callback` (Callable): Function to call with ResourceMetrics on each update

**Raises**:
- `RuntimeError`: If monitoring is already active

**Example**:
```python
def on_metrics(metrics):
    if metrics.cpu_percent > 80:
        print("High CPU usage!")

monitor.start_monitoring(on_metrics)
```

---

#### `stop_monitoring() -> None`

Stop continuous resource monitoring.

**Example**:
```python
monitor.stop_monitoring()
```

---

#### `get_metrics_history(duration: Optional[float] = None) -> List[ResourceMetrics]`

Get metrics history.

**Parameters**:
- `duration` (Optional[float]): Duration in seconds to retrieve history for. If None, returns all history.

**Returns**: List of ResourceMetrics objects

**Example**:
```python
# Get last 5 minutes of metrics
history = monitor.get_metrics_history(duration=300.0)
```

---

#### `check_resource_limits(
    cpu_threshold: float = 0.8,
    memory_threshold: float = 0.8,
    disk_threshold: float = 0.8,
) -> Tuple[bool, List[str]]`

Check if resource usage exceeds thresholds.

**Parameters**:
- `cpu_threshold` (float): CPU usage threshold (0.0 to 1.0, default: `0.8`)
- `memory_threshold` (float): Memory usage threshold (0.0 to 1.0, default: `0.8`)
- `disk_threshold` (float): Disk usage threshold (0.0 to 1.0, default: `0.8`)

**Returns**: Tuple of (exceeded, list_of_warnings)

**Example**:
```python
exceeded, warnings = monitor.check_resource_limits(cpu_threshold=0.9)
if exceeded:
    for warning in warnings:
        print(f"Warning: {warning}")
```

---

## ResourceMetrics

Resource utilization metrics dataclass.

```python
from am_qadf.deployment import ResourceMetrics
from datetime import datetime

metrics = ResourceMetrics(
    timestamp: datetime,
    cpu_percent: float,
    memory_percent: float,
    memory_used_mb: float,
    memory_available_mb: float,
    disk_percent: float,
    disk_used_gb: float,
    disk_available_gb: float,
    network_sent_mb: float,
    network_recv_mb: float,
    process_count: int,
    thread_count: int,
)
```

### Methods

#### `to_dict() -> Dict[str, Any]`

Convert ResourceMetrics to dictionary.

**Returns**: Dictionary with all metrics (timestamp as ISO string)

---

## LoadBalancer

Load balancer for distributing workloads.

```python
from am_qadf.deployment import LoadBalancer, LoadBalancingStrategy

lb = LoadBalancer(strategy: str = 'round_robin')
```

### Parameters

- `strategy` (str): Load balancing strategy - 'round_robin', 'least_connections', 'weighted_round_robin', 'ip_hash' (default: `'round_robin'`)

### Methods

#### `register_worker(worker_id: str, weight: float = 1.0) -> None`

Register worker with load balancer.

**Parameters**:
- `worker_id` (str): Worker identifier
- `weight` (float): Worker weight for weighted strategies (default: `1.0`)

**Example**:
```python
lb.register_worker('worker1', weight=1.0)
lb.register_worker('worker2', weight=2.0)  # Handles 2x load
```

---

#### `unregister_worker(worker_id: str) -> None`

Unregister worker from load balancer.

**Parameters**:
- `worker_id` (str): Worker identifier

---

#### `select_worker(client_ip: Optional[str] = None) -> Optional[str]`

Select worker for next request.

**Parameters**:
- `client_ip` (Optional[str]): Client IP address (required for IP hash strategy)

**Returns**: Worker ID or None if no healthy workers available

**Example**:
```python
worker = lb.select_worker(client_ip='192.168.1.100')
```

---

#### `update_worker_status(worker_id: str, status: str) -> None`

Update worker status.

**Parameters**:
- `worker_id` (str): Worker identifier
- `status` (str): Worker status - 'healthy', 'unhealthy', 'draining', 'down'

**Example**:
```python
lb.update_worker_status('worker1', 'unhealthy')
```

---

#### `record_request(worker_id: str, success: bool = True) -> None`

Record request for worker.

**Parameters**:
- `worker_id` (str): Worker identifier
- `success` (bool): Whether request was successful (default: `True`)

---

#### `record_connection(worker_id: str, connected: bool) -> None`

Record connection event for worker.

**Parameters**:
- `worker_id` (str): Worker identifier
- `connected` (bool): Whether connection was established or closed

---

#### `get_worker_status(worker_id: str) -> Optional[WorkerStatus]`

Get worker status.

**Parameters**:
- `worker_id` (str): Worker identifier

**Returns**: WorkerStatus object or None if worker not found

---

#### `get_all_workers_status() -> Dict[str, WorkerStatus]`

Get status of all workers.

**Returns**: Dictionary mapping worker_id to WorkerStatus

---

#### `get_healthy_workers() -> List[str]`

Get list of healthy worker IDs.

**Returns**: List of healthy worker IDs

---

## AutoScaler

Auto-scaling utility based on metrics.

```python
from am_qadf.deployment import AutoScaler

auto_scaler = AutoScaler(
    min_instances: int = 1,
    max_instances: int = 10,
    target_utilization: float = 0.7,
    scale_up_threshold: float = 0.8,
    scale_down_threshold: float = 0.5,
    cooldown_seconds: float = 300.0,
)
```

### Parameters

- `min_instances` (int): Minimum number of instances (default: `1`, must be >= 1)
- `max_instances` (int): Maximum number of instances (default: `10`, must be >= min_instances)
- `target_utilization` (float): Target resource utilization (0.0 to 1.0, default: `0.7`)
- `scale_up_threshold` (float): Utilization threshold for scaling up (default: `0.8`)
- `scale_down_threshold` (float): Utilization threshold for scaling down (default: `0.5`)
- `cooldown_seconds` (float): Cooldown period between scaling actions in seconds (default: `300.0`)

### Methods

#### `should_scale_up(current_utilization: float) -> bool`

Determine if scaling up is needed.

**Parameters**:
- `current_utilization` (float): Current resource utilization (0.0 to 1.0)

**Returns**: True if scale-up is needed

**Example**:
```python
if auto_scaler.should_scale_up(0.9):
    print("Scale up needed")
```

---

#### `should_scale_down(current_utilization: float) -> bool`

Determine if scaling down is needed.

**Parameters**:
- `current_utilization` (float): Current resource utilization (0.0 to 1.0)

**Returns**: True if scale-down is needed

---

#### `calculate_desired_instances(current_instances: int, metrics: ScalingMetrics) -> int`

Calculate desired number of instances based on metrics.

**Parameters**:
- `current_instances` (int): Current number of instances
- `metrics` (ScalingMetrics): Current scaling metrics

**Returns**: Desired number of instances (clamped between min_instances and max_instances)

**Example**:
```python
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
desired = auto_scaler.calculate_desired_instances(current_instances=5, metrics=metrics)
```

---

#### `get_scaling_decision(current_instances: int, metrics: ScalingMetrics) -> Dict[str, Any]`

Get scaling decision based on current state and metrics.

**Parameters**:
- `current_instances` (int): Current number of instances
- `metrics` (ScalingMetrics): Current scaling metrics

**Returns**: Dictionary with:
  - `action` (str): 'scale_up', 'scale_down', or 'no_action'
  - `current_instances` (int): Current instance count
  - `desired_instances` (int): Desired instance count (if action != 'no_action')
  - `scale_by` (int): Number of instances to scale by (if action != 'no_action')

**Example**:
```python
decision = auto_scaler.get_scaling_decision(current_instances=5, metrics=metrics)
if decision['action'] == 'scale_up':
    print(f"Scale up to {decision['desired_instances']} instances")
```

---

#### `get_scaling_history(limit: int = 10) -> List[Dict[str, Any]]`

Get scaling decision history.

**Parameters**:
- `limit` (int): Maximum number of history entries to return (default: `10`)

**Returns**: List of scaling decision dictionaries (most recent first)

---

## ScalingMetrics

Scaling metrics dataclass.

```python
from am_qadf.deployment import ScalingMetrics

metrics = ScalingMetrics(
    cpu_utilization: float,
    memory_utilization: float,
    request_rate: float,
    error_rate: float,
    queue_depth: int,
    response_time_p50: float,
    response_time_p95: float,
    response_time_p99: float,
)
```

### Fields

- `cpu_utilization` (float): CPU utilization (0.0 to 1.0)
- `memory_utilization` (float): Memory utilization (0.0 to 1.0)
- `request_rate` (float): Requests per second
- `error_rate` (float): Error rate (0.0 to 1.0)
- `queue_depth` (int): Current queue depth
- `response_time_p50` (float): 50th percentile response time in seconds
- `response_time_p95` (float): 95th percentile response time in seconds
- `response_time_p99` (float): 99th percentile response time in seconds

---

## RetryPolicy

Retry policy configuration.

```python
from am_qadf.deployment import RetryPolicy

retry_policy = RetryPolicy(
    max_retries: int = 3,
    backoff_factor: float = 1.0,
    exponential_backoff: bool = True,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    max_backoff_time: float = 60.0,
    initial_backoff_time: float = 1.0,
)
```

### Methods

#### `should_retry(attempt: int, exception: Exception) -> bool`

Determine if operation should be retried.

**Parameters**:
- `attempt` (int): Current attempt number (0-indexed)
- `exception` (Exception): Exception that occurred

**Returns**: True if should retry

---

#### `get_backoff_time(attempt: int) -> float`

Calculate backoff time for retry.

**Parameters**:
- `attempt` (int): Current attempt number (0-indexed)

**Returns**: Backoff time in seconds (capped at max_backoff_time)

---

## CircuitBreaker

Circuit breaker pattern implementation.

```python
from am_qadf.deployment import CircuitBreaker, CircuitState

circuit_breaker = CircuitBreaker(
    failure_threshold: int = 5,
    timeout: float = 60.0,
    half_open_max_calls: int = 3,
    success_threshold: int = 2,
)
```

### Methods

#### `call(func: Callable, *args, **kwargs) -> Any`

Execute function with circuit breaker protection.

**Parameters**:
- `func` (Callable): Function to execute
- `*args`, `**kwargs`: Function arguments

**Returns**: Function return value

**Raises**:
- `CircuitBreakerOpenError`: If circuit is open
- Original exception: If function execution fails

**Example**:
```python
result = circuit_breaker.call(risky_operation, arg1, arg2)
```

---

#### `record_success() -> None`

Record successful operation (resets failure count).

---

#### `record_failure() -> None`

Record failed operation (increments failure count).

---

#### `reset() -> None`

Reset circuit breaker to closed state.

---

#### `@property current_state() -> str`

Get current circuit breaker state - 'closed', 'open', or 'half_open'.

---

## GracefulDegradation

Graceful degradation utilities.

### Static Methods

#### `@staticmethod with_fallback(fallback_value: Any = None, fallback_func: Optional[Callable] = None) -> Callable`

Decorator for graceful degradation with fallback.

**Parameters**:
- `fallback_value` (Any): Default fallback value
- `fallback_func` (Optional[Callable]): Fallback function that receives exception

**Returns**: Decorator function

**Example**:
```python
@GracefulDegradation.with_fallback(fallback_value='default')
def unreliable_function():
    raise ValueError("Error")
```

---

#### `@staticmethod with_timeout(timeout: float, default_value: Any = None) -> Callable`

Decorator for operation timeout.

**Parameters**:
- `timeout` (float): Timeout in seconds
- `default_value` (Any): Value to return on timeout (default: None)

**Returns**: Decorator function

**Example**:
```python
@GracefulDegradation.with_timeout(timeout=5.0, default_value='timeout')
def slow_operation():
    time.sleep(10)
```

---

#### `@staticmethod with_rate_limit(max_calls: int, period: float) -> Callable`

Decorator for rate limiting.

**Parameters**:
- `max_calls` (int): Maximum number of calls allowed
- `period` (float): Time period in seconds

**Returns**: Decorator function

**Raises**:
- `RateLimitExceededError`: If rate limit exceeded

**Example**:
```python
@GracefulDegradation.with_rate_limit(max_calls=10, period=60.0)
def rate_limited_function():
    return "success"
```

---

## PerformanceProfiler

Performance profiling utilities.

```python
from am_qadf.deployment import PerformanceProfiler

profiler = PerformanceProfiler()
```

### Methods

#### `profile_function(func: Callable, *args, **kwargs) -> Dict[str, Any]`

Profile function execution.

**Parameters**:
- `func` (Callable): Function to profile
- `*args`, `**kwargs`: Function arguments

**Returns**: Dictionary with:
  - `function_name` (str): Function name
  - `success` (bool): Whether execution succeeded
  - `result` (Any): Function result (if successful)
  - `exception` (str): Exception message (if failed)
  - `total_time` (float): Total execution time in seconds
  - `total_calls` (int): Number of function calls
  - `profile_output` (str): Profiling output from cProfile

**Example**:
```python
result = profiler.profile_function(my_function, arg1, arg2)
print(f"Execution time: {result['total_time']}s")
```

---

#### `profile_memory(func: Callable, *args, **kwargs) -> Dict[str, Any]`

Profile memory usage of function.

**Parameters**:
- `func` (Callable): Function to profile
- `*args`, `**kwargs`: Function arguments

**Returns**: Dictionary with:
  - `function_name` (str): Function name
  - `success` (bool): Whether execution succeeded
  - `result` (Any): Function result (if successful)
  - `execution_time` (float): Execution time in seconds
  - `current_memory_mb` (float): Current memory usage in MB
  - `peak_memory_mb` (float): Peak memory usage in MB

**Example**:
```python
result = profiler.profile_memory(memory_intensive_function)
print(f"Peak memory: {result['peak_memory_mb']} MB")
```

---

#### `generate_report(profile_data: Dict[str, Any]) -> str`

Generate performance profiling report.

**Parameters**:
- `profile_data` (Dict[str, Any]): Profile data from `profile_function` or `profile_memory`

**Returns**: Formatted report string

---

## PerformanceTuner

Automatic performance tuning utilities.

```python
from am_qadf.deployment import PerformanceTuner

tuner = PerformanceTuner()
```

### Methods

#### `optimize_database_queries(query_logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]`

Analyze and suggest database query optimizations.

**Parameters**:
- `query_logs` (List[Dict]): List of query log dictionaries with keys:
  - `query` (str): SQL query
  - `execution_time` (float): Execution time in seconds
  - `rows_returned` (int): Number of rows returned

**Returns**: List of optimization suggestions (sorted by execution_time descending)

**Example**:
```python
suggestions = tuner.optimize_database_queries(query_logs)
for suggestion in suggestions[:5]:  # Top 5 slowest
    print(f"Query: {suggestion['query']}")
    print(f"Recommendations: {suggestion['recommendations']}")
```

---

#### `optimize_cache_settings(cache_stats: Dict[str, Any]) -> Dict[str, Any]`

Suggest cache optimization settings.

**Parameters**:
- `cache_stats` (Dict): Cache statistics with keys:
  - `hit_rate` (float): Cache hit rate (0.0 to 1.0)
  - `miss_rate` (float): Cache miss rate (0.0 to 1.0)
  - `eviction_count` (int): Number of evictions
  - `total_size_mb` (float): Current cache size in MB
  - `max_size_mb` (float): Maximum cache size in MB

**Returns**: Dictionary with:
  - `current_hit_rate` (float): Current hit rate
  - `recommendations` (List[str]): List of optimization recommendations

**Example**:
```python
suggestions = tuner.optimize_cache_settings(cache_stats)
for recommendation in suggestions['recommendations']:
    print(recommendation)
```

---

#### `optimize_worker_threads(metrics: Dict[str, Any]) -> int`

Suggest optimal number of worker threads.

**Parameters**:
- `metrics` (Dict): Performance metrics with keys:
  - `cpu_utilization` (float): CPU utilization (0.0 to 1.0)
  - `current_threads` (int): Current number of threads
  - `throughput` (float): Requests per second
  - `avg_latency` (float): Average latency in seconds

**Returns**: Suggested number of worker threads (clamped between 1 and 32)

**Example**:
```python
suggested = tuner.optimize_worker_threads({
    'cpu_utilization': 0.5,
    'current_threads': 4,
    'throughput': 50.0,
    'avg_latency': 0.1,
})
print(f"Suggested threads: {suggested}")
```

---

#### `generate_tuning_recommendations(
    config: ProductionConfig,
    metrics: Dict[str, Any]
) -> List[str]`

Generate performance tuning recommendations.

**Parameters**:
- `config` (ProductionConfig): Current production configuration
- `metrics` (Dict): Performance metrics with keys:
  - `cpu_utilization` (float): CPU utilization
  - `throughput` (float): Requests per second
  - `avg_latency` (float): Average latency
  - `p99_latency` (float): 99th percentile latency
  - `db_connection_wait_time` (float): Database connection wait time
  - `queue_depth` (int): Current queue depth

**Returns**: List of tuning recommendation strings

**Example**:
```python
recommendations = tuner.generate_tuning_recommendations(config, metrics)
for rec in recommendations:
    print(rec)
```

---

## Exceptions

### CircuitBreakerOpenError

Exception raised when circuit breaker is open.

```python
from am_qadf.deployment.fault_tolerance import CircuitBreakerOpenError

raise CircuitBreakerOpenError("Circuit is open")
```

### RateLimitExceededError

Exception raised when rate limit is exceeded.

```python
from am_qadf.deployment.fault_tolerance import RateLimitExceededError

raise RateLimitExceededError("Rate limit exceeded")
```

---

## Related

- **[Integration API](integration-api.md)** - Industrial system integration and API gateway
- **[Monitoring API](monitoring-api.md)** - Real-time monitoring and alerting
- **[Deployment Module](../05-modules/deployment.md)** - Module documentation

---

**Parent**: [API Reference](README.md)
