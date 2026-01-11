# Integration Module API Reference

## Overview

The Integration module provides industrial system integration capabilities for AM-QADF, enabling integration with Manufacturing Process Management (MPM) systems, manufacturing equipment, REST API gateway, and authentication and authorization.

## MPMClient

Client for MPM system integration.

```python
from am_qadf.integration import MPMClient

mpm_client = MPMClient(
    base_url: str,
    api_key: Optional[str] = None,
    timeout: float = 30.0,
    verify_ssl: bool = True,
    retry_on_failure: bool = True,
    max_retries: int = 3,
)
```

### Parameters

- `base_url` (str): Base URL of MPM system API (required)
- `api_key` (Optional[str]): API key for authentication (default: `None`)
- `timeout` (float): Request timeout in seconds (default: `30.0`)
- `verify_ssl` (bool): Whether to verify SSL certificates (default: `True`)
- `retry_on_failure` (bool): Whether to retry on failure (default: `True`)
- `max_retries` (int): Maximum number of retries (default: `3`)

### Methods

#### `get_process_data(process_id: str) -> MPMProcessData`

Get process data from MPM system.

**Parameters**:
- `process_id` (str): Process identifier

**Returns**: `MPMProcessData` object

**Raises**:
- `requests.RequestException`: If request fails
- `ValueError`: If process not found

**Example**:
```python
process_data = mpm_client.get_process_data('proc123')
print(f"Process: {process_data.process_id}, Status: {process_data.status}")
```

---

#### `update_process_status(
    process_id: str,
    status: str,
    metadata: Optional[Dict[str, Any]] = None
) -> bool`

Update process status in MPM system.

**Parameters**:
- `process_id` (str): Process identifier
- `status` (str): New status (valid MPMStatus value)
- `metadata` (Optional[Dict]): Optional metadata to include

**Returns**: True if successful

**Raises**:
- `ValueError`: If status is invalid or process not found
- `requests.RequestException`: If request fails

**Example**:
```python
mpm_client.update_process_status('proc123', MPMStatus.COMPLETED.value, {
    'completed_by': 'user1',
    'notes': 'Process completed successfully'
})
```

---

#### `submit_quality_results(process_id: str, results: Dict[str, Any]) -> bool`

Submit quality assessment results to MPM system.

**Parameters**:
- `process_id` (str): Process identifier
- `results` (Dict): Quality assessment results dictionary with keys:
  - `overall_score` (float): Overall quality score
  - `quality_scores` (Dict): Quality scores by category
  - `defects` (List): List of detected defects
  - Other quality-related data

**Returns**: True if successful

**Raises**:
- `ValueError`: If process not found
- `requests.RequestException`: If request fails

**Example**:
```python
results = {
    'overall_score': 0.95,
    'quality_scores': {
        'data_quality': 0.9,
        'signal_quality': 1.0,
        'alignment': 0.88,
    },
    'defects': [],
}
mpm_client.submit_quality_results('proc123', results)
```

---

#### `get_process_parameters(process_id: str) -> Dict[str, Any]`

Get optimized process parameters from MPM system.

**Parameters**:
- `process_id` (str): Process identifier

**Returns**: Dictionary containing process parameters (keys may include: `laser_power`, `scan_speed`, `layer_thickness`, etc.)

**Raises**:
- `ValueError`: If process not found
- `requests.RequestException`: If request fails

**Example**:
```python
parameters = mpm_client.get_process_parameters('proc123')
print(f"Laser Power: {parameters.get('laser_power')}W")
```

---

#### `list_processes(
    build_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
) -> List[MPMProcessData]`

List processes from MPM system.

**Parameters**:
- `build_id` (Optional[str]): Filter by build ID
- `status` (Optional[str]): Filter by status
- `limit` (int): Maximum number of processes to return (default: `100`)
- `offset` (int): Offset for pagination (default: `0`)

**Returns**: List of `MPMProcessData` objects

**Example**:
```python
processes = mpm_client.list_processes(
    build_id='build123',
    status=MPMStatus.RUNNING.value,
    limit=50,
)
```

---

#### `health_check() -> bool`

Check MPM system health.

**Returns**: True if system is healthy

**Example**:
```python
if mpm_client.health_check():
    print("MPM system is healthy")
else:
    print("MPM system is unavailable")
```

---

#### `close() -> None`

Close client session.

**Example**:
```python
mpm_client.close()
```

---

#### Context Manager Support

```python
with MPMClient(base_url='https://mpm.example.com', api_key='key') as client:
    process_data = client.get_process_data('proc123')
# Session automatically closed
```

---

## MPMProcessData

Process data from MPM system dataclass.

```python
from am_qadf.integration import MPMProcessData
from datetime import datetime

process_data = MPMProcessData(
    process_id: str,
    build_id: str,
    material: str,
    parameters: Dict[str, Any] = {},
    status: str = 'pending',
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    metadata: Dict[str, Any] = {},
)
```

### Methods

#### `to_dict() -> Dict[str, Any]`

Convert MPMProcessData to dictionary (timestamps as ISO strings).

---

#### `@classmethod from_dict(data: Dict[str, Any]) -> MPMProcessData`

Create MPMProcessData from dictionary (parses ISO timestamp strings).

---

## EquipmentClient

Client for manufacturing equipment integration.

```python
from am_qadf.integration import EquipmentClient, EquipmentType

equipment = EquipmentClient(
    equipment_type: str,  # EquipmentType enum value or string
    equipment_id: str,
    connection_config: Dict[str, Any],
)
```

### Parameters

- `equipment_type` (str): Equipment type - 'cnc_mill', 'cnc_lathe', '3d_printer', 'laser_cutter', 'sensor', 'plc', 'robot_arm', 'cmm', 'other'
- `equipment_id` (str): Unique equipment identifier
- `connection_config` (Dict): Connection configuration dictionary with keys:
  - `type` (str): Connection type - 'network', 'serial', 'file'
  - For network: `host`, `port`, `protocol` (optional)
  - For serial: `port`, `baudrate`, `timeout` (optional)
  - For file: `file_path`, `format` (optional)
  - Authentication: `username`, `password`, `api_key` (if needed)

### Methods

#### `connect() -> bool`

Connect to manufacturing equipment.

**Returns**: True if connection successful

**Example**:
```python
if equipment.connect():
    print(f"Connected to {equipment.equipment_id}")
else:
    print("Connection failed")
```

---

#### `disconnect() -> None`

Disconnect from manufacturing equipment (stops monitoring if active).

---

#### `read_sensor_data(sensor_id: str) -> Dict[str, Any]`

Read data from sensor.

**Parameters**:
- `sensor_id` (str): Sensor identifier

**Returns**: Dictionary with keys:
  - `sensor_id` (str): Sensor identifier
  - `value` (float): Sensor reading value
  - `unit` (str): Unit of measurement
  - `timestamp` (str): Reading timestamp (ISO format)
  - `status` (str): Sensor status ('ok', 'error', etc.)
  - `metadata` (Dict): Additional sensor-specific data

**Raises**:
- `ConnectionError`: If not connected

**Example**:
```python
sensor_data = equipment.read_sensor_data('temperature_sensor')
print(f"Temperature: {sensor_data['value']} {sensor_data['unit']}")
```

---

#### `send_command(command: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]`

Send command to equipment.

**Parameters**:
- `command` (str): Command name (equipment-specific)
- `parameters` (Optional[Dict]): Command parameters

**Returns**: Dictionary with keys:
  - `success` (bool): Whether command was successful
  - `message` (str): Response message
  - `data` (Dict): Response data (if any)
  - `timestamp` (str): Command execution timestamp (ISO format)

**Raises**:
- `ConnectionError`: If not connected

**Example**:
```python
result = equipment.send_command('start_print', {
    'file': 'model.stl',
    'layer_height': 0.1,
    'temperature': 230.0,
})
if result['success']:
    print(f"Command executed: {result['message']}")
```

---

#### `get_equipment_status() -> EquipmentStatus`

Get current equipment status.

**Returns**: `EquipmentStatus` object

**Raises**:
- `ConnectionError`: If not connected

**Example**:
```python
status = equipment.get_equipment_status()
print(f"Status: {status.status}, Progress: {status.progress*100}%")
```

---

#### `register_status_callback(callback: Callable[[EquipmentStatus], None]) -> None`

Register callback for status updates.

**Parameters**:
- `callback` (Callable): Function to call with EquipmentStatus on each update

**Example**:
```python
def on_status_update(status):
    print(f"Equipment {status.equipment_id}: {status.status}")

equipment.register_status_callback(on_status_update)
```

---

#### `start_monitoring(update_interval: float = 5.0) -> None`

Start continuous status monitoring.

**Parameters**:
- `update_interval` (float): Status update interval in seconds (default: `5.0`)

**Raises**:
- `ConnectionError`: If not connected
- `RuntimeError`: If monitoring already active

---

#### `stop_monitoring() -> None`

Stop continuous status monitoring.

---

#### `@property is_connected() -> bool`

Check if equipment is connected.

---

#### `@property is_monitoring() -> bool`

Check if monitoring is active.

---

#### Context Manager Support

```python
with EquipmentClient('3d_printer', 'printer1', connection_config) as equipment:
    status = equipment.get_equipment_status()
    data = equipment.read_sensor_data('temp1')
# Automatically disconnected
```

---

## EquipmentStatus

Equipment status information dataclass.

```python
from am_qadf.integration import EquipmentStatus, EquipmentStatusValue

status = EquipmentStatus(
    equipment_id: str,
    status: str,  # EquipmentStatusValue enum value
    current_operation: Optional[str] = None,
    progress: float = 0.0,  # 0.0 to 1.0
    error_message: Optional[str] = None,
    timestamp: datetime = datetime.now(),
    metadata: Dict[str, Any] = {},
)
```

### Methods

#### `to_dict() -> Dict[str, Any]`

Convert EquipmentStatus to dictionary (timestamp as ISO string).

---

#### `@classmethod from_dict(data: Dict[str, Any]) -> EquipmentStatus`

Create EquipmentStatus from dictionary (parses ISO timestamp string).

---

## APIGateway

REST API gateway for industrial access.

```python
from am_qadf.integration import APIGateway

gateway = APIGateway(
    base_path: str = '/api/v1',
    enable_cors: bool = True,
    enable_logging: bool = True,
    default_timeout: float = 30.0,
)
```

### Methods

#### `register_endpoint(endpoint: APIEndpoint) -> None`

Register API endpoint.

**Parameters**:
- `endpoint` (APIEndpoint): APIEndpoint instance

**Example**:
```python
from am_qadf.integration import APIEndpoint, APIResponse

def health_handler(request):
    return APIResponse(status_code=200, body={'status': 'healthy'})

endpoint = APIEndpoint(
    path='/health',
    method='GET',
    handler=health_handler,
    requires_auth=False,
)
gateway.register_endpoint(endpoint)
```

---

#### `add_middleware(middleware: APIMiddleware) -> None`

Add middleware to API gateway.

**Parameters**:
- `middleware` (APIMiddleware): APIMiddleware instance

---

#### `handle_request(
    method: str,
    path: str,
    headers: Optional[Dict[str, str]] = None,
    query_params: Optional[Dict[str, Any]] = None,
    body: Optional[Dict[str, Any]] = None,
    user_id: Optional[str] = None,
    client_ip: Optional[str] = None
) -> APIResponse`

Handle API request.

**Parameters**:
- `method` (str): HTTP method (GET, POST, PUT, DELETE, etc.)
- `path` (str): Request path
- `headers` (Optional[Dict]): Request headers
- `query_params` (Optional[Dict]): Query parameters
- `body` (Optional[Dict]): Request body
- `user_id` (Optional[str]): User ID (if authenticated)
- `client_ip` (Optional[str]): Client IP address

**Returns**: `APIResponse` object

**Example**:
```python
response = gateway.handle_request(
    method='GET',
    path='/api/v1/health',
    headers={'Authorization': 'Bearer token123'},
    query_params={'detailed': 'true'},
)
print(f"Status: {response.status_code}, Body: {response.body}")
```

---

#### `get_health_status() -> Dict[str, Any]`

Get API gateway health status.

**Returns**: Dictionary with:
  - `status` (str): Health status
  - `uptime` (str): System uptime
  - `request_count` (int): Total requests processed
  - `error_count` (int): Total errors
  - `endpoint_count` (int): Number of registered endpoints
  - `middleware_count` (int): Number of middleware components

---

#### `get_metrics() -> Dict[str, Any]`

Get API gateway metrics.

**Returns**: Dictionary with metrics

---

## APIEndpoint

API endpoint definition.

```python
from am_qadf.integration import APIEndpoint, APIRequest, APIResponse

endpoint = APIEndpoint(
    path: str,
    method: str,
    handler: Callable[[APIRequest], APIResponse],
    summary: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[List[str]] = None,
    requires_auth: bool = True,
    rate_limit: Optional[int] = None,
)
```

### Parameters

- `path` (str): Endpoint path (e.g., '/api/v1/health', supports `:param` for path parameters)
- `method` (str): HTTP method (GET, POST, PUT, DELETE, etc.)
- `handler` (Callable): Handler function that takes `APIRequest` and returns `APIResponse`
- `summary` (Optional[str]): Short summary of endpoint
- `description` (Optional[str]): Detailed description
- `tags` (Optional[List[str]]): Endpoint tags for grouping
- `requires_auth` (bool): Whether authentication is required (default: `True`)
- `rate_limit` (Optional[int]): Rate limit in requests per minute (default: `None`)

### Handler Function Signature

```python
def handler(request: APIRequest) -> APIResponse:
    """
    Process API request.
    
    Args:
        request: APIRequest object with:
            - method (str): HTTP method
            - path (str): Request path
            - headers (Dict): Request headers
            - query_params (Dict): Query parameters
            - path_params (Dict): Path parameters (from :param in path)
            - body (Dict): Request body
            - request_id (str): Unique request ID
            - user_id (Optional[str]): User ID (if authenticated)
    
    Returns:
        APIResponse object
    """
    # Process request
    return APIResponse(
        status_code=200,
        body={'result': 'success'},
    )
```

---

## APIRequest

API request object.

```python
from am_qadf.integration import APIRequest

request = APIRequest(
    method: str,
    path: str,
    headers: Dict[str, str] = {},
    query_params: Dict[str, Any] = {},
    path_params: Dict[str, str] = {},
    body: Optional[Dict[str, Any]] = None,
    request_id: str = uuid4(),
    timestamp: datetime = datetime.now(),
    user_id: Optional[str] = None,
    client_ip: Optional[str] = None,
)
```

---

## APIResponse

API response object.

```python
from am_qadf.integration import APIResponse

response = APIResponse(
    status_code: int = 200,
    headers: Dict[str, str] = {},
    body: Optional[Dict[str, Any]] = None,
    error: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None,
    timestamp: datetime = datetime.now(),
)
```

### Error Response Format

```python
error_response = APIResponse(
    status_code=400,
    error={
        'code': 'AM_QADF_400',
        'message': 'Error description',
        'category': 'validation',
        'details': {},
        'timestamp': '2024-01-01T00:00:00Z',
        'request_id': 'uuid',
    },
)
```

---

## AuthenticationManager

Authentication and authorization manager.

```python
from am_qadf.integration import AuthenticationManager, AuthMethod

auth_manager = AuthenticationManager(
    auth_method: str = 'jwt',  # 'jwt', 'oauth2', 'api_key', 'ldap', 'kerberos', 'basic'
    config: Optional[Dict[str, Any]] = None,
)
```

### Configuration Dictionary

**For JWT**:
```python
config = {
    'secret_key': 'your-secret-key',
    'algorithm': 'HS256',
    'expiration_seconds': 3600,
}
```

**For API Key**:
```python
config = {
    'api_keys': {
        'api_key_123': {
            'user_id': 'user1',
            'username': 'apiuser',
            'email': 'api@example.com',
            'roles': ['operator'],
            'permissions': ['quality:read'],
        },
    },
}
```

**For OAuth2**:
```python
config = {
    'client_id': 'your_client_id',
    'client_secret': 'your_client_secret',
    'token_url': 'https://oauth.example.com/token',
}
```

### Methods

#### `authenticate(credentials: Dict[str, str]) -> Tuple[bool, Optional[str]]`

Authenticate user and return token.

**Parameters**:
- `credentials` (Dict): Credentials dictionary:
  - For JWT/Basic: `{'username': str, 'password': str}`
  - For API key: `{'api_key': str}`
  - For OAuth2: `{'code': str}` or `{'token': str}`

**Returns**: Tuple of (success, token_or_error_message)

**Example**:
```python
success, token = auth_manager.authenticate({
    'username': 'user1',
    'password': 'password123',
})
if success:
    print(f"Token: {token}")
else:
    print(f"Error: {token}")
```

---

#### `validate_token(token: str) -> Tuple[bool, Optional[Dict]]`

Validate authentication token.

**Parameters**:
- `token` (str): Authentication token

**Returns**: Tuple of (is_valid, user_info_dict_or_error)

**Example**:
```python
is_valid, user_info = auth_manager.validate_token(token)
if is_valid:
    print(f"User: {user_info.get('username')}")
```

---

#### `authorize(token: str, resource: str, action: str) -> bool`

Authorize user action on resource.

**Parameters**:
- `token` (str): Authentication token
- `resource` (str): Resource identifier (e.g., 'quality', 'optimization')
- `action` (str): Action identifier (e.g., 'assess', 'read', 'write')

**Returns**: True if authorized

**Example**:
```python
if auth_manager.authorize(token, 'quality', 'assess'):
    # Proceed with quality assessment
    pass
else:
    # Return 403 Forbidden
    pass
```

---

#### `refresh_token(token: str) -> Optional[str]`

Refresh authentication token.

**Parameters**:
- `token` (str): Current authentication token

**Returns**: New token if refresh successful, None otherwise

---

#### `revoke_token(token: str) -> bool`

Revoke token (add to blacklist).

**Parameters**:
- `token` (str): Token to revoke

**Returns**: True if successful

---

#### `register_user(
    username: str,
    email: str,
    password: str,
    roles: Optional[List[str]] = None,
    permissions: Optional[List[str]] = None,
    organization: str = ''
) -> User`

Register new user (basic implementation - use database in production).

**Returns**: Created `User` object

---

#### `get_user(user_id: str) -> Optional[User]`

Get user by ID.

**Returns**: `User` object or None if not found

---

## User

User information dataclass.

```python
from am_qadf.integration import User

user = User(
    user_id: str,
    username: str,
    email: str,
    roles: List[str] = [],
    permissions: List[str] = [],
    organization: str = '',
    metadata: Dict[str, Any] = {},
    created_at: datetime = datetime.now(),
    updated_at: datetime = datetime.now(),
)
```

### Methods

#### `has_permission(permission: str) -> bool`

Check if user has specific permission.

**Parameters**:
- `permission` (str): Permission string (e.g., 'quality:assess')

**Returns**: True if user has permission

---

#### `has_role(role: str) -> bool`

Check if user has specific role.

**Parameters**:
- `role` (str): Role name (e.g., 'admin', 'engineer')

**Returns**: True if user has role

---

#### `to_dict() -> Dict[str, Any]`

Convert User to dictionary (timestamps as ISO strings).

---

## RoleBasedAccessControl

Role-based access control (RBAC) implementation.

```python
from am_qadf.integration import RoleBasedAccessControl

rbac = RoleBasedAccessControl()
```

### Default Roles

- **admin**: Full access to all resources
- **quality_analyst**: Quality assessment and read access
- **engineer**: Quality read, optimization run, model read, streaming read
- **operator**: Quality read, streaming read
- **viewer**: Read-only access to quality, optimization, and models

### Methods

#### `assign_role(user_id: str, role: str) -> None`

Assign role to user.

**Parameters**:
- `user_id` (str): User identifier
- `role` (str): Role name

---

#### `remove_role(user_id: str, role: str) -> None`

Remove role from user.

---

#### `check_permission(user: User, resource: str, action: str) -> bool`

Check if user has permission for action on resource.

**Parameters**:
- `user` (User): User object
- `resource` (str): Resource identifier
- `action` (str): Action identifier

**Returns**: True if user has permission

**Example**:
```python
has_permission = rbac.check_permission(user, 'quality', 'assess')
```

---

#### `get_user_permissions(user_id: str) -> List[str]`

Get all permissions for user (including role-based and inherited).

**Parameters**:
- `user_id` (str): User identifier

**Returns**: List of permission strings

---

#### `define_role(role: str, permissions: List[str]) -> None`

Define new role with permissions.

**Parameters**:
- `role` (str): Role name
- `permissions` (List[str]): List of permission strings

---

#### `add_role_hierarchy(parent_role: str, child_role: str) -> None`

Add role hierarchy (parent inherits child permissions).

**Parameters**:
- `parent_role` (str): Parent role name
- `child_role` (str): Child role name

---

## Permission

Common permissions enum.

```python
from am_qadf.integration import Permission

# Quality assessment
Permission.QUALITY_ASSESS  # 'quality:assess'
Permission.QUALITY_READ    # 'quality:read'
Permission.QUALITY_WRITE   # 'quality:write'

# Process optimization
Permission.OPTIMIZATION_RUN   # 'optimization:run'
Permission.OPTIMIZATION_READ  # 'optimization:read'

# Model management
Permission.MODEL_READ    # 'model:read'
Permission.MODEL_WRITE   # 'model:write'
Permission.MODEL_DELETE  # 'model:delete'

# Streaming
Permission.STREAMING_START  # 'streaming:start'
Permission.STREAMING_STOP   # 'streaming:stop'
Permission.STREAMING_READ   # 'streaming:read'

# System administration
Permission.ADMIN_CONFIG   # 'admin:config'
Permission.ADMIN_MONITOR  # 'admin:monitor'
Permission.ADMIN_USERS    # 'admin:users'
```

---

## Related

- **[Deployment API](deployment-api.md)** - Production deployment utilities
- **[Monitoring API](monitoring-api.md)** - Real-time monitoring and alerting
- **[Integration Module](../05-modules/integration.md)** - Module documentation

---

**Parent**: [API Reference](README.md)
