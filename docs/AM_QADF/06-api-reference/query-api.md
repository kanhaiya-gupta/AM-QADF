# Query Module API Reference

## Overview

The Query module provides a unified interface for accessing multi-source data from the NoSQL data warehouse.

## UnifiedQueryClient

Main client for querying all data sources.

```python
from am_qadf.query import UnifiedQueryClient

client = UnifiedQueryClient(mongo_client: Optional[MongoDBClient] = None)
```

### Methods

#### `__init__(mongo_client: Optional[MongoDBClient] = None)`

Initialize unified query client.

**Parameters**:
- `mongo_client` (Optional[MongoDBClient]): MongoDB client instance

#### `set_mongo_client(mongo_client: MongoDBClient) -> None`

Set MongoDB client for all query clients.

**Parameters**:
- `mongo_client` (MongoDBClient): MongoDB client instance

#### `query(model_id: str, sources: List[str], spatial_bbox: Optional[Tuple] = None, temporal_range: Optional[Tuple] = None, signal_types: Optional[List[str]] = None) -> QueryResult`

Query data from multiple sources.

**Parameters**:
- `model_id` (str): Model identifier
- `sources` (List[str]): List of source names ('hatching', 'laser', 'ct', 'ispm', etc.)
- `spatial_bbox` (Optional[Tuple]): Spatial bounding box `((x_min, y_min, z_min), (x_max, y_max, z_max))`
- `temporal_range` (Optional[Tuple]): Temporal range `(start_time, end_time)` in seconds
- `signal_types` (Optional[List[str]]): List of signal types to retrieve

**Returns**: `QueryResult` with points, signals, and metadata

**Example**:
```python
result = client.query(
    model_id="my_model",
    sources=['hatching', 'laser'],
    spatial_bbox=((-50, -50, -50), (50, 50, 50)),
    temporal_range=(0, 1000)
)
```

#### `get_all_data(model_id: str) -> Dict[str, Any]`

Get all data for a model from all sources.

**Parameters**:
- `model_id` (str): Model identifier

**Returns**: Dictionary containing data from all sources

---

## BaseQueryClient

Abstract base class for all query clients.

```python
from am_qadf.query import BaseQueryClient

class MyQueryClient(BaseQueryClient):
    def query(self, spatial, temporal, signal_types):
        # Implementation
        pass
```

### Abstract Methods

#### `query(spatial: Optional[SpatialQuery], temporal: Optional[TemporalQuery], signal_types: Optional[List[SignalType]]) -> QueryResult`

Execute a query with spatial, temporal, and signal type filters.

**Parameters**:
- `spatial` (Optional[SpatialQuery]): Spatial query parameters
- `temporal` (Optional[TemporalQuery]): Temporal query parameters
- `signal_types` (Optional[List[SignalType]]): Signal types to retrieve

**Returns**: `QueryResult` with points, signals, and metadata

#### `get_available_signals() -> List[SignalType]`

Get list of available signal types.

**Returns**: List of available `SignalType` enums

#### `get_bounding_box(component_id: Optional[str] = None) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]`

Get bounding box for data.

**Parameters**:
- `component_id` (Optional[str]): Component identifier

**Returns**: `((x_min, y_min, z_min), (x_max, y_max, z_max))`

---

## QueryResult

Result container for query operations.

```python
@dataclass
class QueryResult:
    points: np.ndarray  # (N, 3) array of points
    signals: Dict[str, np.ndarray]  # Dictionary of signal arrays
    metadata: Dict[str, Any]  # Additional metadata
```

### Attributes

- **points** (`np.ndarray`): Array of 3D points (N, 3)
- **signals** (`Dict[str, np.ndarray]`): Dictionary mapping signal names to arrays
- **metadata** (`Dict[str, Any]`): Additional metadata

---

## SpatialQuery

Spatial query parameters.

```python
@dataclass
class SpatialQuery:
    bbox_min: Optional[Tuple[float, float, float]] = None
    bbox_max: Optional[Tuple[float, float, float]] = None
    component_id: Optional[str] = None
    layer_range: Optional[Tuple[int, int]] = None
```

### Attributes

- **bbox_min** (Optional[Tuple[float, float, float]]): Minimum bounding box corner
- **bbox_max** (Optional[Tuple[float, float, float]]): Maximum bounding box corner
- **component_id** (Optional[str]): Component identifier
- **layer_range** (Optional[Tuple[int, int]]): Layer range `(start_layer, end_layer)`

---

## TemporalQuery

Temporal query parameters.

```python
@dataclass
class TemporalQuery:
    start_time: Optional[float] = None  # seconds
    end_time: Optional[float] = None  # seconds
    layer_range: Optional[Tuple[int, int]] = None
```

### Attributes

- **start_time** (Optional[float]): Start time in seconds
- **end_time** (Optional[float]): End time in seconds
- **layer_range** (Optional[Tuple[int, int]]): Layer range `(start_layer, end_layer)`

---

## Specialized Query Clients

### HatchingClient

Query hatching path data.

```python
from am_qadf.query import HatchingClient

client = HatchingClient(mongo_client)
```

#### Methods

##### `get_layers(model_id: str, layer_range: Optional[Tuple[int, int]] = None) -> List[Dict]`

Get hatching layers.

**Parameters**:
- `model_id` (str): Model identifier
- `layer_range` (Optional[Tuple[int, int]]): Layer range

**Returns**: List of layer dictionaries

##### `get_all_points(model_id: str, spatial_bbox: Optional[Tuple] = None) -> np.ndarray`

Get all hatching points.

**Parameters**:
- `model_id` (str): Model identifier
- `spatial_bbox` (Optional[Tuple]): Spatial bounding box

**Returns**: Array of points (N, 3)

### LaserParameterClient

Query laser parameter data.

```python
from am_qadf.query import LaserParameterClient

client = LaserParameterClient(mongo_client, use_mongodb=True)
```

### CTScanClient

Query CT scan data.

```python
from am_qadf.query import CTScanClient

client = CTScanClient(mongo_client, use_mongodb=True)
```

### InSituMonitoringClient

Query ISPM monitoring data.

```python
from am_qadf.query import InSituMonitoringClient

client = InSituMonitoringClient(mongo_client, use_mongodb=True)
```

### STLModelClient

Query STL model data.

```python
from am_qadf.query import STLModelClient

client = STLModelClient(mongo_client)
```

#### Methods

##### `get_model_bounding_box(model_id: str) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]`

Get model bounding box.

**Parameters**:
- `model_id` (str): Model identifier

**Returns**: `((x_min, y_min, z_min), (x_max, y_max, z_max))`

## Related

- [Query Module Documentation](../05-modules/query.md) - Module overview
- [All API References](README.md) - Other API references

---

**Parent**: [API Reference](README.md)

