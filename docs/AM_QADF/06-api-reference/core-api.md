# Core Module API Reference

## Overview

The Core module provides foundational domain entities, value objects, and exceptions used throughout the AM-QADF framework.

## Entities

### VoxelData

Represents data stored in a single voxel.

```python
from am_qadf.core import VoxelData

voxel_data = VoxelData(
    signals: Dict[str, float] = {},
    count: int = 0
)
```

#### Attributes

- **signals** (`Dict[str, float]`): Dictionary mapping signal names to values
- **count** (`int`): Number of data points contributing to this voxel

#### Methods

##### `add_signal(signal_name: str, value: float, aggregation: str = 'mean') -> None`

Add a signal value to this voxel.

**Parameters**:
- `signal_name` (str): Name of the signal (e.g., 'power', 'temperature')
- `value` (float): Signal value to add
- `aggregation` (str): How to aggregate multiple values ('mean', 'max', 'min', 'sum')

**Example**:
```python
voxel_data = VoxelData()
voxel_data.add_signal('power', 200.0)
voxel_data.add_signal('temperature', 1000.0)
```

##### `finalize(aggregation: str = 'mean') -> None`

Finalize voxel data by aggregating multiple values.

**Parameters**:
- `aggregation` (str): Aggregation method ('mean', 'max', 'min', 'sum')

**Example**:
```python
voxel_data.finalize(aggregation='mean')
# Signals are now aggregated to single values
```

---

## Value Objects

### VoxelCoordinates

Immutable value object representing voxel coordinates.

```python
from am_qadf.core import VoxelCoordinates

coords = VoxelCoordinates(
    x: float,
    y: float,
    z: float,
    voxel_size: float = 0.1,
    voxel_volume: Optional[float] = None,
    rotation_x: float = 0.0,
    rotation_y: float = 0.0,
    rotation_z: float = 0.0,
    is_solid: bool = True,
    is_processed: bool = False,
    is_defective: bool = False,
    material_density: Optional[float] = None,
    material_type: Optional[str] = None,
    layer_number: Optional[int] = None,
    scan_vector_id: Optional[str] = None,
    processing_timestamp: Optional[datetime] = None,
    quality_score: Optional[float] = None,
    temperature_peak: Optional[float] = None,
    cooling_rate: Optional[float] = None
)
```

#### Attributes

- **x, y, z** (float): 3D coordinates in mm
- **voxel_size** (float): Voxel size in mm (default: 0.1)
- **voxel_volume** (Optional[float]): Voxel volume in mm³ (auto-calculated)
- **rotation_x, rotation_y, rotation_z** (float): Euler angles in degrees
- **is_solid, is_processed, is_defective** (bool): Voxel state flags
- **material_density** (Optional[float]): Material density in g/cm³
- **material_type** (Optional[str]): Material type identifier
- **layer_number** (Optional[int]): Build layer number
- **quality_score** (Optional[float]): Quality score (0-100)

#### Methods

##### `validate() -> None`

Validate voxel coordinates. Raises `ValueError` if invalid.

**Raises**:
- `ValueError`: If coordinates, voxel size, or other values are invalid

##### `get_coordinates() -> Tuple[float, float, float]`

Get coordinates as tuple.

**Returns**: `(x, y, z)` tuple

##### `get_rotations() -> Tuple[float, float, float]`

Get rotations as tuple.

**Returns**: `(rotation_x, rotation_y, rotation_z)` tuple

##### `distance_to(other: VoxelCoordinates) -> float`

Calculate Euclidean distance to another voxel.

**Parameters**:
- `other` (VoxelCoordinates): Other voxel coordinates

**Returns**: Distance in mm

**Example**:
```python
coords1 = VoxelCoordinates(x=0, y=0, z=0)
coords2 = VoxelCoordinates(x=10, y=10, z=10)
distance = coords1.distance_to(coords2)  # ~17.32 mm
```

---

### QualityMetric

Immutable value object representing a quality metric.

```python
from am_qadf.core import QualityMetric

metric = QualityMetric(
    value: float,
    metric_name: str,
    unit: Optional[str] = None,
    timestamp: Optional[datetime] = None
)
```

#### Attributes

- **value** (float): Metric value
- **metric_name** (str): Name of the metric
- **unit** (Optional[str]): Unit of measurement
- **timestamp** (Optional[datetime]): Timestamp of measurement

#### Methods

##### `validate() -> None`

Validate quality metric. Raises `ValueError` if invalid.

**Raises**:
- `ValueError`: If value is not numeric or name is empty

---

## Exceptions

### AMQADFError

Base exception for all AM-QADF framework errors.

```python
from am_qadf.core import AMQADFError

try:
    # Framework operation
    pass
except AMQADFError as e:
    print(f"Framework error: {e}")
```

### VoxelGridError

Exception raised for voxel grid related errors.

```python
from am_qadf.core import VoxelGridError
```

### SignalMappingError

Exception raised for signal mapping related errors.

```python
from am_qadf.core import SignalMappingError
```

### InterpolationError

Exception raised for interpolation related errors. Inherits from `SignalMappingError`.

```python
from am_qadf.core import InterpolationError
```

### FusionError

Exception raised for data fusion related errors.

```python
from am_qadf.core import FusionError
```

### QueryError

Exception raised for query related errors.

```python
from am_qadf.core import QueryError
```

### StorageError

Exception raised for storage related errors.

```python
from am_qadf.core import StorageError
```

### ValidationError

Exception raised for validation errors.

```python
from am_qadf.core import ValidationError
```

### ConfigurationError

Exception raised for configuration errors.

```python
from am_qadf.core import ConfigurationError
```

### CoordinateSystemError

Exception raised for coordinate system transformation errors.

```python
from am_qadf.core import CoordinateSystemError
```

### QualityAssessmentError

Exception raised for quality assessment related errors.

```python
from am_qadf.core import QualityAssessmentError
```

## Exception Hierarchy

```
AMQADFError (base)
├── VoxelGridError
├── SignalMappingError
│   └── InterpolationError
├── FusionError
├── QueryError
├── StorageError
├── ValidationError
├── ConfigurationError
├── CoordinateSystemError
└── QualityAssessmentError
```

## Related

- [Core Module Documentation](../05-modules/core.md) - Module overview
- [All API References](README.md) - Other API references

---

**Parent**: [API Reference](README.md)

