# Monitoring Module API Reference

## Overview

The Monitoring module provides real-time monitoring capabilities for AM-QADF, enabling live process monitoring, alert generation, multi-channel notifications, threshold management, and system health monitoring.

## MonitoringConfig

Configuration dataclass for monitoring operations.

```python
from am_qadf.monitoring import MonitoringConfig

config = MonitoringConfig(
    enable_alerts: bool = True,
    alert_check_interval_seconds: float = 1.0,
    enable_email_notifications: bool = False,
    email_smtp_server: Optional[str] = None,
    email_smtp_port: int = 587,
    email_from_address: Optional[str] = None,
    email_recipients: List[str] = [],
    enable_sms_notifications: bool = False,
    sms_provider: Optional[str] = None,  # 'twilio', 'aws_sns'
    sms_recipients: List[str] = [],
    enable_dashboard_notifications: bool = True,
    websocket_port: int = 8765,
    enable_health_monitoring: bool = True,
    health_check_interval_seconds: float = 60.0,
    alert_cooldown_seconds: float = 300.0
)
```

### Fields

- `enable_alerts` (bool): Enable alert generation (default: `True`)
- `alert_check_interval_seconds` (float): Interval between alert checks in seconds (default: `1.0`)
- `enable_email_notifications` (bool): Enable email notifications (default: `False`)
- `email_smtp_server` (Optional[str]): SMTP server address (default: `None`)
- `email_smtp_port` (int): SMTP server port (default: `587`)
- `email_from_address` (Optional[str]): Email sender address (default: `None`)
- `email_recipients` (List[str]): List of email recipient addresses (default: `[]`)
- `enable_sms_notifications` (bool): Enable SMS notifications (default: `False`)
- `sms_provider` (Optional[str]): SMS provider - 'twilio' or 'aws_sns' (default: `None`)
- `sms_recipients` (List[str]): List of SMS recipient numbers (default: `[]`)
- `enable_dashboard_notifications` (bool): Enable dashboard WebSocket notifications (default: `True`)
- `websocket_port` (int): WebSocket server port (default: `8765`)
- `enable_health_monitoring` (bool): Enable health monitoring (default: `True`)
- `health_check_interval_seconds` (float): Health check interval in seconds (default: `60.0`)
- `alert_cooldown_seconds` (float): Cooldown period between alerts in seconds (default: `300.0`)

---

## MonitoringClient

Unified monitoring interface that coordinates all monitoring operations.

```python
from am_qadf.monitoring import MonitoringClient, MonitoringConfig

client = MonitoringClient(config: Optional[MonitoringConfig] = None)
```

### Methods

#### `__init__(config: Optional[MonitoringConfig] = None)`

Initialize monitoring client.

**Parameters**:
- `config` (Optional[MonitoringConfig]): Monitoring configuration. If None, uses default config.

---

#### `start_monitoring() -> None`

Start monitoring services (health checks, alert monitoring).

**Example**:
```python
client.start_monitoring()
```

---

#### `stop_monitoring() -> None`

Stop monitoring services.

**Example**:
```python
client.stop_monitoring()
```

---

#### `register_metric(metric_name: str, threshold_config: ThresholdConfig) -> None`

Register a metric with threshold configuration.

**Parameters**:
- `metric_name` (str): Metric name (identifier)
- `threshold_config` (ThresholdConfig): Threshold configuration for the metric

**Example**:
```python
from am_qadf.monitoring import ThresholdConfig

threshold_config = ThresholdConfig(
    metric_name='temperature',
    threshold_type='absolute',
    lower_threshold=800.0,
    upper_threshold=1200.0,
)
client.register_metric('temperature', threshold_config)
```

---

#### `update_metric(metric_name: str, value: float, timestamp: Optional[datetime] = None) -> None`

Update metric value and trigger threshold checks.

**Parameters**:
- `metric_name` (str): Metric name
- `value` (float): Current metric value
- `timestamp` (Optional[datetime]): Optional timestamp (defaults to now)

**Example**:
```python
from datetime import datetime

client.update_metric('temperature', 1000.0)
client.update_metric('pressure', 150.0, timestamp=datetime.now())
```

---

#### `get_current_metrics() -> Dict[str, float]`

Get current values of all monitored metrics.

**Returns**: Dictionary mapping metric names to current values

**Example**:
```python
metrics = client.get_current_metrics()
print(f"Current temperature: {metrics.get('temperature', 'N/A')}")
```

---

#### `get_health_status(component_name: Optional[str] = None) -> Dict[str, HealthStatus]`

Get health status of components.

**Parameters**:
- `component_name` (Optional[str]): Optional component name (None = all components)

**Returns**: Dictionary mapping component names to HealthStatus objects

**Example**:
```python
# Get all component health
all_health = client.get_health_status()
for name, status in all_health.items():
    print(f"{name}: {status.status} (score: {status.health_score:.2f})")

# Get specific component health
component_health = client.get_health_status('my_component')
```

---

#### `acknowledge_alert(alert_id: str, acknowledged_by: str) -> None`

Acknowledge an alert.

**Parameters**:
- `alert_id` (str): Alert ID
- `acknowledged_by` (str): Username or identifier of person acknowledging

**Example**:
```python
client.acknowledge_alert('alert_123', 'operator1')
```

---

## Alert

Alert dataclass representing a monitoring alert.

```python
from am_qadf.monitoring import Alert
from datetime import datetime

alert = Alert(
    alert_id: str,  # Auto-generated UUID
    alert_type: str,
    severity: str,  # 'low', 'medium', 'high', 'critical'
    message: str,
    timestamp: datetime,
    source: str,
    metadata: Dict[str, Any] = {},
    acknowledged: bool = False,
    acknowledged_by: Optional[str] = None,
    acknowledged_at: Optional[datetime] = None
)
```

### Fields

- `alert_id` (str): Unique alert identifier (UUID)
- `alert_type` (str): Type of alert (e.g., 'quality_threshold', 'spc_out_of_control')
- `severity` (str): Alert severity - 'low', 'medium', 'high', or 'critical'
- `message` (str): Human-readable alert message
- `timestamp` (datetime): Alert generation timestamp
- `source` (str): Source component or system
- `metadata` (Dict[str, Any]): Additional metadata dictionary
- `acknowledged` (bool): Whether alert is acknowledged (default: `False`)
- `acknowledged_by` (Optional[str]): Username who acknowledged (default: `None`)
- `acknowledged_at` (Optional[datetime]): Acknowledgment timestamp (default: `None`)

---

## AlertSystem

Alert generation, escalation, and management system.

```python
from am_qadf.monitoring import AlertSystem, MonitoringConfig

alert_system = AlertSystem(config: MonitoringConfig)
```

### Methods

#### `__init__(config: MonitoringConfig)`

Initialize alert system.

**Parameters**:
- `config` (MonitoringConfig): Monitoring configuration

---

#### `generate_alert(alert_type: str, severity: str, message: str, source: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[Alert]`

Generate an alert.

**Parameters**:
- `alert_type` (str): Type of alert
- `severity` (str): Alert severity - 'low', 'medium', 'high', or 'critical'
- `message` (str): Alert message
- `source` (str): Source component
- `metadata` (Optional[Dict[str, Any]]): Optional metadata dictionary

**Returns**: Generated `Alert` object, or `None` if in cooldown period

**Example**:
```python
alert = alert_system.generate_alert(
    alert_type='quality_threshold',
    severity='high',
    message='Temperature threshold exceeded',
    source='Sensor1',
    metadata={'value': 1300.0, 'threshold': 1200.0}
)

if alert:
    print(f"Alert generated: {alert.alert_id}")
```

---

#### `get_active_alerts(severity: Optional[str] = None, alert_type: Optional[str] = None) -> List[Alert]`

Get active (unacknowledged) alerts.

**Parameters**:
- `severity` (Optional[str]): Filter by severity (default: `None`)
- `alert_type` (Optional[str]): Filter by alert type (default: `None`)

**Returns**: List of active `Alert` objects

**Example**:
```python
# Get all active alerts
all_alerts = alert_system.get_active_alerts()

# Get only high severity alerts
high_alerts = alert_system.get_active_alerts(severity='high')

# Get alerts of specific type
quality_alerts = alert_system.get_active_alerts(alert_type='quality_threshold')
```

---

#### `get_alert_history(start_time: datetime, end_time: datetime, filters: Optional[Dict[str, Any]] = None) -> List[Alert]`

Get alert history within time range.

**Parameters**:
- `start_time` (datetime): Start time
- `end_time` (datetime): End time
- `filters` (Optional[Dict[str, Any]]): Additional filters (e.g., `{'alert_type': 'quality_threshold'}`)

**Returns**: List of `Alert` objects within time range

**Example**:
```python
from datetime import datetime, timedelta

start_time = datetime.now() - timedelta(hours=24)
end_time = datetime.now()

# Get all alerts in last 24 hours
history = alert_system.get_alert_history(start_time, end_time)

# Get filtered history
quality_history = alert_system.get_alert_history(
    start_time,
    end_time,
    filters={'alert_type': 'quality_threshold', 'severity': 'high'}
)
```

---

#### `acknowledge_alert(alert_id: str, acknowledged_by: str) -> bool`

Acknowledge an alert.

**Parameters**:
- `alert_id` (str): Alert ID
- `acknowledged_by` (str): Username or identifier

**Returns**: `True` if successful, `False` if alert not found

**Example**:
```python
success = alert_system.acknowledge_alert('alert_123', 'operator1')
if success:
    print("Alert acknowledged")
```

---

#### `escalate_alert(alert_id: str) -> bool`

Escalate an alert to higher severity.

**Parameters**:
- `alert_id` (str): Alert ID

**Returns**: `True` if successful, `False` if alert not found or already at maximum severity

**Example**:
```python
success = alert_system.escalate_alert('alert_123')
if success:
    print("Alert escalated")
```

---

## ThresholdConfig

Configuration dataclass for alert thresholds.

```python
from am_qadf.monitoring import ThresholdConfig

config = ThresholdConfig(
    metric_name: str,
    threshold_type: str = 'absolute',  # 'absolute', 'relative', 'rate_of_change', 'spc_limit'
    lower_threshold: Optional[float] = None,
    upper_threshold: Optional[float] = None,
    window_size: int = 100,
    enable_spc_integration: bool = False,
    spc_baseline_id: Optional[str] = None,
    severity_mapping: Dict[str, str] = {}
)
```

### Fields

- `metric_name` (str): Metric name (required)
- `threshold_type` (str): Threshold type - 'absolute', 'relative', 'rate_of_change', or 'spc_limit' (default: `'absolute'`)
- `lower_threshold` (Optional[float]): Lower threshold value (default: `None`)
- `upper_threshold` (Optional[float]): Upper threshold value (default: `None`)
- `window_size` (int): Moving window size for relative/rate thresholds (default: `100`)
- `enable_spc_integration` (bool): Enable SPC integration (default: `False`)
- `spc_baseline_id` (Optional[str]): SPC baseline ID (default: `None`)
- `severity_mapping` (Dict[str, str]): Map threshold violations to severities (default: `{}`)

---

## ThresholdManager

Alert threshold management with multiple threshold types.

```python
from am_qadf.monitoring import ThresholdManager, ThresholdConfig

threshold_manager = ThresholdManager()
```

### Methods

#### `__init__()`

Initialize threshold manager.

---

#### `add_threshold(metric_name: str, config: ThresholdConfig) -> None`

Add threshold configuration for a metric.

**Parameters**:
- `metric_name` (str): Metric name
- `config` (ThresholdConfig): Threshold configuration

**Example**:
```python
config = ThresholdConfig(
    metric_name='temperature',
    threshold_type='absolute',
    upper_threshold=1200.0,
)
threshold_manager.add_threshold('temperature', config)
```

---

#### `check_value(metric_name: str, value: float, timestamp: datetime) -> Optional[Alert]`

Check value against threshold and generate alert if violated.

**Parameters**:
- `metric_name` (str): Metric name
- `value` (float): Current metric value
- `timestamp` (datetime): Current timestamp

**Returns**: `Alert` object if threshold violated, `None` otherwise

**Example**:
```python
from datetime import datetime

# Check value (generates alert if threshold violated)
alert = threshold_manager.check_value('temperature', 1300.0, datetime.now())

if alert:
    print(f"Alert: {alert.message} (Severity: {alert.severity})")
```

---

#### `remove_threshold(metric_name: str) -> None`

Remove threshold configuration for a metric.

**Parameters**:
- `metric_name` (str): Metric name

**Example**:
```python
threshold_manager.remove_threshold('temperature')
```

---

#### `get_threshold(metric_name: str) -> Optional[ThresholdConfig]`

Get threshold configuration for a metric.

**Parameters**:
- `metric_name` (str): Metric name

**Returns**: `ThresholdConfig` if found, `None` otherwise

**Example**:
```python
config = threshold_manager.get_threshold('temperature')
if config:
    print(f"Upper threshold: {config.upper_threshold}")
```

---

#### `integrate_spc_baseline(metric_name: str, baseline: 'BaselineStatistics') -> None`

Integrate SPC baseline with threshold manager.

**Parameters**:
- `metric_name` (str): Metric name
- `baseline` (BaselineStatistics): SPC baseline statistics

**Example**:
```python
from am_qadf.analytics.spc import SPCClient
import numpy as np

spc_client = SPCClient()
baseline_data = np.random.normal(100.0, 10.0, 100)
baseline = spc_client.establish_baseline(baseline_data)

threshold_manager.integrate_spc_baseline('spc_metric', baseline)
```

---

## HealthStatus

Health status dataclass.

```python
from am_qadf.monitoring import HealthStatus
from datetime import datetime

status = HealthStatus(
    component_name: str,
    status: str,  # 'healthy', 'degraded', 'unhealthy', 'critical'
    health_score: float,  # 0.0 to 1.0
    timestamp: datetime,
    metrics: Dict[str, float] = {},
    issues: List[str] = [],
    metadata: Dict[str, Any] = {}
)
```

### Fields

- `component_name` (str): Component name
- `status` (str): Health status - 'healthy', 'degraded', 'unhealthy', or 'critical'
- `health_score` (float): Health score from 0.0 to 1.0 (1.0 = healthy, 0.0 = critical)
- `timestamp` (datetime): Status timestamp
- `metrics` (Dict[str, float]): Component health metrics (default: `{}`)
- `issues` (List[str]): List of identified issues (default: `[]`)
- `metadata` (Dict[str, Any]): Additional metadata (default: `{}`)

---

## HealthMonitor

System and process health monitoring.

```python
from am_qadf.monitoring import HealthMonitor

health_monitor = HealthMonitor(check_interval_seconds: float = 60.0)
```

### Methods

#### `__init__(check_interval_seconds: float = 60.0)`

Initialize health monitor.

**Parameters**:
- `check_interval_seconds` (float): Health check interval in seconds (default: `60.0`)

---

#### `check_system_health() -> HealthStatus`

Check system health (CPU, memory, disk, network).

**Returns**: `HealthStatus` object for system

**Example**:
```python
system_health = health_monitor.check_system_health()
print(f"System status: {system_health.status}")
print(f"Health score: {system_health.health_score:.2f}")
print(f"Issues: {system_health.issues}")
```

---

#### `register_component(component_name: str, health_checker: Callable[[], Dict[str, float]]) -> None`

Register a custom component with health checker function.

**Parameters**:
- `component_name` (str): Component name
- `health_checker` (Callable[[], Dict[str, float]]): Health checker function that returns metrics dictionary

**Example**:
```python
def component_health_checker():
    return {
        'cpu_percent': 45.0,
        'memory_percent': 60.0,
        'error_rate': 0.01,
        'latency_ms': 50.0,
    }

health_monitor.register_component('my_component', component_health_checker)
```

---

#### `check_process_health(component_name: str) -> HealthStatus`

Check health of a registered component.

**Parameters**:
- `component_name` (str): Component name

**Returns**: `HealthStatus` object for component

**Example**:
```python
component_health = health_monitor.check_process_health('my_component')
print(f"Component status: {component_health.status}")
```

---

#### `get_all_component_health() -> Dict[str, HealthStatus]`

Get health status for all registered components.

**Returns**: Dictionary mapping component names to `HealthStatus` objects

**Example**:
```python
all_health = health_monitor.get_all_component_health()
for name, status in all_health.items():
    print(f"{name}: {status.status} (score: {status.health_score:.2f})")
```

---

#### `get_health_history(component_name: str, start_time: datetime, end_time: datetime) -> List[HealthStatus]`

Get health history for a component.

**Parameters**:
- `component_name` (str): Component name
- `start_time` (datetime): Start time
- `end_time` (datetime): End time

**Returns**: List of `HealthStatus` objects within time range

**Example**:
```python
from datetime import datetime, timedelta

start_time = datetime.now() - timedelta(hours=1)
end_time = datetime.now()
history = health_monitor.get_health_history('system', start_time, end_time)
print(f"Found {len(history)} health records")
```

---

#### `start_monitoring() -> None`

Start health monitoring in background thread.

**Example**:
```python
health_monitor.start_monitoring()
```

---

#### `stop_monitoring() -> None`

Stop health monitoring.

**Example**:
```python
health_monitor.stop_monitoring()
```

---

## NotificationChannels

Multi-channel notification system.

```python
from am_qadf.monitoring import NotificationChannels, MonitoringConfig

notification_channels = NotificationChannels(config: MonitoringConfig)
```

### Methods

#### `__init__(config: MonitoringConfig)`

Initialize notification channels.

**Parameters**:
- `config` (MonitoringConfig): Monitoring configuration

---

#### `configure_email(smtp_server: str, smtp_port: int, username: str, password: str) -> None`

Configure email notifications.

**Parameters**:
- `smtp_server` (str): SMTP server address
- `smtp_port` (int): SMTP server port
- `username` (str): SMTP username
- `password` (str): SMTP password

**Example**:
```python
notification_channels.configure_email(
    smtp_server='smtp.example.com',
    smtp_port=587,
    username='user@example.com',
    password='password'
)
```

---

#### `configure_sms(provider: str, api_key: str, api_secret: str, from_number: str) -> None`

Configure SMS notifications.

**Parameters**:
- `provider` (str): SMS provider - 'twilio' or 'aws_sns'
- `api_key` (str): Provider API key
- `api_secret` (str): Provider API secret
- `from_number` (str): Sender phone number

**Example**:
```python
# Twilio
notification_channels.configure_sms(
    provider='twilio',
    api_key='your_account_sid',
    api_secret='your_auth_token',
    from_number='+1234567890'
)

# AWS SNS
notification_channels.configure_sms(
    provider='aws_sns',
    api_key='your_access_key',
    api_secret='your_secret_key',
    from_number='+1234567890'
)
```

---

#### `broadcast_alert(alert: Alert, channels: Optional[List[str]] = None) -> Dict[str, bool]`

Broadcast alert to notification channels.

**Parameters**:
- `alert` (Alert): Alert to broadcast
- `channels` (Optional[List[str]]): List of channels - 'email', 'sms', 'dashboard'. If None, uses all enabled channels.

**Returns**: Dictionary mapping channel names to success status (True/False)

**Example**:
```python
alert = Alert(...)
results = notification_channels.broadcast_alert(alert)
# Returns: {'email': True, 'sms': False, 'dashboard': True}

# Broadcast to specific channels
results = notification_channels.broadcast_alert(alert, channels=['email', 'dashboard'])
```

---

#### `send_email_alert(alert: Alert) -> bool`

Send email alert.

**Parameters**:
- `alert` (Alert): Alert to send

**Returns**: `True` if successful, `False` otherwise

**Example**:
```python
success = notification_channels.send_email_alert(alert)
```

---

#### `send_sms_alert(alert: Alert) -> bool`

Send SMS alert.

**Parameters**:
- `alert` (Alert): Alert to send

**Returns**: `True` if successful, `False` otherwise

**Example**:
```python
success = notification_channels.send_sms_alert(alert)
```

---

#### `send_dashboard_notification(alert: Alert) -> bool`

Send dashboard notification via WebSocket.

**Parameters**:
- `alert` (Alert): Alert to send

**Returns**: `True` if successful, `False` otherwise

**Example**:
```python
success = notification_channels.send_dashboard_notification(alert)
```

---

#### `broadcast_websocket(data: Dict[str, Any]) -> None`

Broadcast arbitrary data to WebSocket clients.

**Parameters**:
- `data` (Dict[str, Any]): Data to broadcast (will be JSON serialized)

**Example**:
```python
notification_channels.broadcast_websocket({
    'type': 'metric_update',
    'metric': 'temperature',
    'value': 1000.0,
    'timestamp': datetime.now().isoformat()
})
```

---

## MonitoringStorage

Alert and notification history storage.

```python
from am_qadf.monitoring import MonitoringStorage

storage = MonitoringStorage(
    mongo_client: Any,
    database_name: str = 'am_qadf_monitoring',
    alerts_collection: str = 'alerts',
    notifications_collection: str = 'notifications',
    health_collection: str = 'health_history'
)
```

### Methods

#### `__init__(mongo_client: Any, database_name: str = 'am_qadf_monitoring', alerts_collection: str = 'alerts', notifications_collection: str = 'notifications', health_collection: str = 'health_history')`

Initialize monitoring storage.

**Parameters**:
- `mongo_client` (Any): MongoDB client
- `database_name` (str): Database name (default: `'am_qadf_monitoring'`)
- `alerts_collection` (str): Alerts collection name (default: `'alerts'`)
- `notifications_collection` (str): Notifications collection name (default: `'notifications'`)
- `health_collection` (str): Health history collection name (default: `'health_history'`)

---

#### `store_alert(alert: Alert) -> str`

Store alert in MongoDB.

**Parameters**:
- `alert` (Alert): Alert to store

**Returns**: Document ID of stored alert

**Example**:
```python
alert_id = storage.store_alert(alert)
print(f"Alert stored with ID: {alert_id}")
```

---

#### `store_notification(notification_data: Dict[str, Any]) -> str`

Store notification delivery record.

**Parameters**:
- `notification_data` (Dict[str, Any]): Notification data dictionary

**Returns**: Document ID of stored notification

**Example**:
```python
notification_data = {
    'alert_id': 'alert_123',
    'channel': 'email',
    'recipient': 'admin@example.com',
    'sent_at': datetime.now(),
    'status': 'delivered'
}
notification_id = storage.store_notification(notification_data)
```

---

#### `store_health_status(health_status: HealthStatus) -> str`

Store health status in MongoDB.

**Parameters**:
- `health_status` (HealthStatus): Health status to store

**Returns**: Document ID of stored health status

**Example**:
```python
health_id = storage.store_health_status(health_status)
```

---

#### `query_alert_history(start_time: datetime, end_time: datetime, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]`

Query alert history.

**Parameters**:
- `start_time` (datetime): Start time
- `end_time` (datetime): End time
- `filters` (Optional[Dict[str, Any]]): Additional MongoDB query filters

**Returns**: List of alert documents

**Example**:
```python
from datetime import datetime, timedelta

start_time = datetime.now() - timedelta(days=7)
end_time = datetime.now()
alerts = storage.query_alert_history(
    start_time,
    end_time,
    filters={'severity': 'high'}
)
print(f"Found {len(alerts)} high-severity alerts")
```

---

#### `create_indexes() -> None`

Create indexes for efficient queries.

**Example**:
```python
storage.create_indexes()
```

---

## Related

- [Monitoring Module](../05-modules/monitoring.md) - Module documentation
- [Streaming API](streaming-api.md) - Streaming integration
- [SPC API](spc-api.md) - SPC integration

---

**Parent**: [API Reference](README.md)
