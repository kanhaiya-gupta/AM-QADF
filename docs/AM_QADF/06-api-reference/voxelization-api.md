# Voxelization Module API Reference

## Overview

The Voxelization module provides core data structures for representing 3D spatial data as voxel grids.

## VoxelGrid

Core voxel grid data structure.

```python
from am_qadf.voxelization import VoxelGrid

grid = VoxelGrid(
    bbox_min: Tuple[float, float, float],
    bbox_max: Tuple[float, float, float],
    resolution: float,
    aggregation: str = 'mean'
)
```

### Attributes

- **bbox_min** (`np.ndarray`): Minimum bounding box corner (x_min, y_min, z_min) in mm
- **bbox_max** (`np.ndarray`): Maximum bounding box corner (x_max, y_max, z_max) in mm
- **resolution** (`float`): Voxel size in mm (cubic voxels)
- **aggregation** (`str`): Aggregation method ('mean', 'max', 'min', 'sum')
- **size** (`np.ndarray`): Grid size (bbox_max - bbox_min)
- **dims** (`np.ndarray`): Grid dimensions (number of voxels in each dimension)
- **actual_size** (`np.ndarray`): Actual grid size (may be slightly larger than bbox due to rounding)
- **voxels** (`Dict[Tuple[int, int, int], VoxelData]`): Sparse voxel data dictionary
- **available_signals** (`set`): Set of available signal names

### Methods

#### `add_point(x: float, y: float, z: float, signals: Dict[str, float]) -> None`

Add a data point to the voxel grid.

**Parameters**:
- `x, y, z` (float): World coordinates in mm
- `signals` (Dict[str, float]): Dictionary of signal names to values

**Example**:
```python
grid.add_point(10.0, 20.0, 30.0, signals={'power': 200.0, 'temperature': 1000.0})
```

#### `finalize() -> None`

Finalize voxel grid by aggregating multiple values per voxel.

**Example**:
```python
grid.finalize()  # Aggregates all signals using configured aggregation method
```

#### `get_voxel(i: int, j: int, k: int) -> Optional[VoxelData]`

Get voxel data at given indices.

**Parameters**:
- `i, j, k` (int): Voxel indices

**Returns**: VoxelData if voxel exists, None otherwise

#### `get_signal_array(signal_name: str, default: float = 0.0) -> np.ndarray`

Get signal array for entire grid.

**Parameters**:
- `signal_name` (str): Name of the signal
- `default` (float): Default value for empty voxels

**Returns**: 3D array of signal values with shape `(dims[0], dims[1], dims[2])`

**Example**:
```python
power_array = grid.get_signal_array('power', default=0.0)
```

#### `get_signal(x: float, y: float, z: float, signal_name: str, default: float = 0.0) -> float`

Get signal value at specific world coordinates.

**Parameters**:
- `x, y, z` (float): World coordinates in mm
- `signal_name` (str): Name of the signal
- `default` (float): Default value if voxel is empty

**Returns**: Signal value

#### `get_bounding_box() -> Tuple[np.ndarray, np.ndarray]`

Get the bounding box of the grid.

**Returns**: Tuple of (bbox_min, bbox_max) as numpy arrays

#### `get_statistics() -> Dict[str, Any]`

Get statistics about the voxel grid.

**Returns**: Dictionary with grid statistics including:
- `dimensions`: Grid dimensions tuple
- `resolution_mm`: Resolution in mm
- `bounding_box_min`: Minimum bounding box
- `bounding_box_max`: Maximum bounding box
- `total_voxels`: Total number of voxels
- `filled_voxels`: Number of filled voxels
- `fill_ratio`: Fill ratio (0-1)
- `available_signals`: List of available signal names
- Per-signal statistics (mean, min, max, std)

#### `has_signal(signal_name: str) -> bool`

Check if signal exists in grid.

**Parameters**:
- `signal_name` (str): Name of the signal

**Returns**: True if signal exists

#### `get_voxel_count() -> int`

Get total number of voxels in grid.

**Returns**: Total voxel count (dims[0] * dims[1] * dims[2])

#### `get_filled_voxel_count() -> int`

Get number of voxels with data.

**Returns**: Number of non-empty voxels

---

## Adaptive Resolution

### SpatialResolutionMap

Maps spatial regions to resolution levels.

```python
from am_qadf.voxelization import SpatialResolutionMap

@dataclass
class SpatialResolutionMap:
    regions: List[Tuple[Tuple[float, float, float], Tuple[float, float, float], float]] = []
    # Each region: (bbox_min, bbox_max, resolution)
    default_resolution: float = 1.0
```

### TemporalResolutionMap

Maps temporal regions to resolution levels.

```python
from am_qadf.voxelization import TemporalResolutionMap

@dataclass
class TemporalResolutionMap:
    time_ranges: List[Tuple[float, float, float]] = []
    # Each range: (time_start, time_end, resolution)
    layer_ranges: List[Tuple[int, int, float]] = []
    # Each range: (layer_start, layer_end, resolution)
    default_resolution: float = 1.0
```

### AdaptiveResolutionGrid

Voxel grid with spatially and temporally variable resolution.

```python
from am_qadf.voxelization import AdaptiveResolutionGrid, SpatialResolutionMap, TemporalResolutionMap

grid = AdaptiveResolutionGrid(
    bbox_min: Tuple[float, float, float],
    bbox_max: Tuple[float, float, float],
    base_resolution: float = 1.0,
    spatial_resolution_map: Optional[SpatialResolutionMap] = None,
    temporal_resolution_map: Optional[TemporalResolutionMap] = None
)
```

### Methods

#### `add_point(x: float, y: float, z: float, signals: Dict[str, float], timestamp: Optional[float] = None, layer_index: Optional[int] = None) -> None`

Add point to grid with adaptive resolution.

**Parameters**:
- `x, y, z` (float): Point coordinates
- `signals` (Dict[str, float]): Signal values dictionary
- `timestamp` (Optional[float]): Optional timestamp (seconds)
- `layer_index` (Optional[int]): Optional layer index

#### `get_resolution_for_point(x: float, y: float, z: float, timestamp: Optional[float] = None, layer_index: Optional[int] = None) -> float`

Get resolution for a specific point based on spatial and temporal maps.

**Parameters**:
- `x, y, z` (float): Point coordinates
- `timestamp` (Optional[float]): Optional timestamp
- `layer_index` (Optional[int]): Optional layer index

**Returns**: Resolution in mm

#### `finalize(adaptive_density: bool = True) -> None`

Finalize grid by creating region-specific grids.

**Parameters**:
- `adaptive_density` (bool): If True, adjust resolution based on local data density

#### `get_signal_array(signal_name: str, target_resolution: Optional[float] = None, default: float = 0.0) -> np.ndarray`

Get signal array at a target resolution.

**Parameters**:
- `signal_name` (str): Name of signal
- `target_resolution` (Optional[float]): Target resolution (if None, uses finest available)
- `default` (float): Default value for empty voxels

**Returns**: Signal array

#### `get_statistics() -> Dict[str, Any]`

Get statistics for the adaptive grid.

**Returns**: Statistics dictionary with:
- `finalized`: Whether grid is finalized
- `num_regions`: Number of resolution regions
- `resolutions`: List of resolution info
- `total_points`: Total number of points
- `available_signals`: Set of available signals

#### `add_spatial_region(bbox_min: Tuple[float, float, float], bbox_max: Tuple[float, float, float], resolution: float) -> None`

Add a spatial region with specific resolution.

**Parameters**:
- `bbox_min` (Tuple[float, float, float]): Minimum bounding box corner
- `bbox_max` (Tuple[float, float, float]): Maximum bounding box corner
- `resolution` (float): Resolution for this region (mm)

#### `add_temporal_range(time_start: float, time_end: float, resolution: float) -> None`

Add a temporal range with specific resolution.

**Parameters**:
- `time_start` (float): Start time (seconds)
- `time_end` (float): End time (seconds)
- `resolution` (float): Resolution for this time range (mm)

#### `add_layer_range(layer_start: int, layer_end: int, resolution: float) -> None`

Add a layer range with specific resolution.

**Parameters**:
- `layer_start` (int): Start layer index
- `layer_end` (int): End layer index
- `resolution` (float): Resolution for this layer range (mm)

---

### `create_spatial_only_grid(adaptive_grid: AdaptiveResolutionGrid) -> AdaptiveResolutionGrid`

Create a spatial-only adaptive grid from an existing adaptive grid.

**Parameters**:
- `adaptive_grid` (AdaptiveResolutionGrid): Source adaptive resolution grid

**Returns**: New AdaptiveResolutionGrid with only spatial resolution mapping

---

### `create_temporal_only_grid(adaptive_grid: AdaptiveResolutionGrid) -> AdaptiveResolutionGrid`

Create a temporal-only adaptive grid from an existing adaptive grid.

**Parameters**:
- `adaptive_grid` (AdaptiveResolutionGrid): Source adaptive resolution grid

**Returns**: New AdaptiveResolutionGrid with only temporal resolution mapping

---

## Multi-Resolution

### ResolutionLevel

Resolution levels for hierarchical grids.

```python
from am_qadf.voxelization import ResolutionLevel

# Available levels:
ResolutionLevel.COARSE      # Low resolution, fast
ResolutionLevel.MEDIUM      # Medium resolution, balanced
ResolutionLevel.FINE         # High resolution, detailed
ResolutionLevel.ULTRA_FINE  # Very high resolution, maximum detail
```

### MultiResolutionGrid

Hierarchical voxel grid with multiple resolution levels.

```python
from am_qadf.voxelization import MultiResolutionGrid

grid = MultiResolutionGrid(
    bbox_min: Tuple[float, float, float],
    bbox_max: Tuple[float, float, float],
    base_resolution: float = 1.0,
    num_levels: int = 3,
    level_ratio: float = 2.0
)
```

### Methods

#### `get_resolution(level: int) -> float`

Get resolution for a specific level.

**Parameters**:
- `level` (int): Resolution level (0 = coarsest, num_levels-1 = finest)

**Returns**: Resolution in mm

#### `get_level(level: int) -> VoxelGrid`

Get voxel grid for a specific level.

**Parameters**:
- `level` (int): Resolution level

**Returns**: VoxelGrid object

#### `add_point(x: float, y: float, z: float, signals: Dict[str, float], level: Optional[int] = None) -> None`

Add point to grid(s).

**Parameters**:
- `x, y, z` (float): Point coordinates
- `signals` (Dict[str, float]): Signal values dictionary
- `level` (Optional[int]): Specific level to add to (if None, adds to all levels)

#### `finalize() -> None`

Finalize all grids.

#### `get_signal_array(signal_name: str, level: int = 0, default: float = 0.0) -> np.ndarray`

Get signal array for a specific level.

**Parameters**:
- `signal_name` (str): Name of signal
- `level` (int): Resolution level
- `default` (float): Default value for empty voxels

**Returns**: Signal array

#### `get_statistics(level: int = 0) -> Dict[str, Any]`

Get statistics for a specific level.

**Parameters**:
- `level` (int): Resolution level

**Returns**: Statistics dictionary

#### `select_appropriate_level(target_resolution: float, prefer_coarse: bool = False) -> int`

Select appropriate resolution level based on target resolution.

**Parameters**:
- `target_resolution` (float): Desired resolution (mm)
- `prefer_coarse` (bool): If True, prefer coarser level if close

**Returns**: Best matching level index

#### `get_level_for_view_distance(view_distance: float, min_resolution: float = 0.1, max_resolution: float = 5.0) -> int`

Select resolution level based on view distance.

**Parameters**:
- `view_distance` (float): Distance from camera to object (mm)
- `min_resolution` (float): Minimum resolution to use (mm)
- `max_resolution` (float): Maximum resolution to use (mm)

**Returns**: Appropriate level index

#### `downsample_from_finer(source_level: int, target_level: int) -> None`

Downsample data from finer level to coarser level.

**Parameters**:
- `source_level` (int): Source (finer) level
- `target_level` (int): Target (coarser) level

---

### ResolutionSelector

Select appropriate resolution based on various criteria.

```python
from am_qadf.voxelization import ResolutionSelector

selector = ResolutionSelector(performance_mode: str = 'balanced')
# performance_mode: 'fast', 'balanced', 'quality'
```

### Methods

#### `select_for_performance(grid: MultiResolutionGrid, num_points: int, available_memory: Optional[float] = None) -> int`

Select resolution level based on performance requirements.

**Parameters**:
- `grid` (MultiResolutionGrid): MultiResolutionGrid object
- `num_points` (int): Number of data points
- `available_memory` (Optional[float]): Available memory in GB (if None, estimates)

**Returns**: Recommended level index

#### `select_for_data_density(grid: MultiResolutionGrid, data_density: float) -> int`

Select resolution based on data density.

**Parameters**:
- `grid` (MultiResolutionGrid): MultiResolutionGrid object
- `data_density` (float): Data density (points per mm³)

**Returns**: Recommended level index

#### `select_for_view(grid: MultiResolutionGrid, view_parameters: Dict[str, Any]) -> int`

Select resolution based on view parameters.

**Parameters**:
- `grid` (MultiResolutionGrid): MultiResolutionGrid object
- `view_parameters` (Dict[str, Any]): Dictionary with view parameters:
  - `'distance'`: View distance (mm)
  - `'zoom'`: Zoom level
  - `'region_size'`: Size of viewed region (mm)

**Returns**: Recommended level index

---

## Coordinate Systems

### CoordinateSystemType

Types of coordinate systems.

```python
from am_qadf.voxelization import CoordinateSystemType

# Available types:
CoordinateSystemType.STL              # STL file coordinate system
CoordinateSystemType.BUILD_PLATFORM   # Build platform coordinate system
CoordinateSystemType.GLOBAL           # Global/world coordinate system
CoordinateSystemType.COMPONENT_LOCAL  # Component-local coordinate system
```

### CoordinateSystem

Represents a coordinate system with origin and orientation.

```python
from am_qadf.voxelization import CoordinateSystem

system = CoordinateSystem(
    name: str,
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    scale: float = 1.0
)
```

### Methods

#### `transform_point(point: np.ndarray) -> np.ndarray`

Transform a point from this coordinate system to parent.

**Parameters**:
- `point` (np.ndarray): Point (x, y, z) in this coordinate system

**Returns**: Point in parent coordinate system

#### `inverse_transform_point(point: np.ndarray) -> np.ndarray`

Transform a point from parent to this coordinate system.

**Parameters**:
- `point` (np.ndarray): Point (x, y, z) in parent coordinate system

**Returns**: Point in this coordinate system

#### `get_bounding_box() -> Tuple[np.ndarray, np.ndarray]`

Get bounding box in parent coordinate system.

**Returns**: Tuple of (bbox_min, bbox_max)

---

### CoordinateSystemRegistry

Registry for managing multiple coordinate systems and transformations.

```python
from am_qadf.voxelization import CoordinateSystemRegistry

registry = CoordinateSystemRegistry()
```

### Methods

#### `register(name: str, origin: Tuple[float, float, float] = (0.0, 0.0, 0.0), rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0), scale: float = 1.0, parent: Optional[str] = None) -> None`

Register a coordinate system.

**Parameters**:
- `name` (str): Name/identifier of the coordinate system
- `origin` (Tuple[float, float, float]): Origin point (x, y, z) in parent coordinate system
- `rotation` (Tuple[float, float, float]): Rotation angles (rx, ry, rz) in degrees
- `scale` (float): Scale factor
- `parent` (Optional[str]): Name of parent coordinate system (None for root)

#### `get(name: str) -> Optional[CoordinateSystem]`

Get a coordinate system by name.

**Parameters**:
- `name` (str): Name of the coordinate system

**Returns**: CoordinateSystem if found, None otherwise

#### `transform(point: np.ndarray, from_system: str, to_system: str) -> np.ndarray`

Transform a point from one coordinate system to another.

**Parameters**:
- `point` (np.ndarray): Point (x, y, z) to transform
- `from_system` (str): Source coordinate system name
- `to_system` (str): Target coordinate system name

**Returns**: Transformed point

#### `list_systems() -> List[str]`

List all registered coordinate systems.

**Returns**: List of coordinate system names

---

## CoordinateSystemTransformer

Transforms points between coordinate systems.

```python
from am_qadf.voxelization import CoordinateSystemTransformer

transformer = CoordinateSystemTransformer()
```

### Methods

#### `transform_point(point: Tuple[float, float, float], from_system: Dict[str, Any], to_system: Dict[str, Any]) -> Tuple[float, float, float]`

Transform a single point.

**Parameters**:
- `point` (Tuple[float, float, float]): Point coordinates (x, y, z) in mm
- `from_system` (Dict[str, Any]): Source coordinate system dictionary with keys:
  - `'origin'`: Origin point (x, y, z)
  - `'rotation'`: Optional rotation dictionary with 'x', 'y', 'z' or 'x_deg', 'y_deg', 'z_deg'
  - `'scale_factor'`: Optional scale factor (dict with 'x', 'y', 'z' or array)
- `to_system` (Dict[str, Any]): Target coordinate system dictionary (same format)

**Returns**: Transformed point coordinates (x, y, z) in mm

#### `transform_points(points: np.ndarray, from_system: Dict[str, Any], to_system: Dict[str, Any]) -> np.ndarray`

Transform multiple points (vectorized).

**Parameters**:
- `points` (np.ndarray): Array of points (N, 3)
- `from_system` (Dict[str, Any]): Source coordinate system dictionary
- `to_system` (Dict[str, Any]): Target coordinate system dictionary

**Returns**: Array of transformed points (N, 3)

#### `align_data_sources(model_id: str, stl_coord_system: Dict[str, Any], hatching_coord_system: Optional[Dict[str, Any]] = None, ct_coord_system: Optional[Dict[str, Any]] = None, ispm_coord_system: Optional[Dict[str, Any]] = None, target_system: str = 'build_platform') -> Dict[str, Any]`

Align coordinate systems from multiple data sources.

**Parameters**:
- `model_id` (str): Model UUID
- `stl_coord_system` (Dict[str, Any]): STL model coordinate system
- `hatching_coord_system` (Optional[Dict[str, Any]]): Hatching layer coordinate system
- `ct_coord_system` (Optional[Dict[str, Any]]): CT scan coordinate system
- `ispm_coord_system` (Optional[Dict[str, Any]]): ISPM monitoring coordinate system
- `target_system` (str): Target coordinate system name ('build_platform', 'ct_scan', 'ispm')

**Returns**: Dictionary with aligned coordinate systems and transformation matrices

#### `validate_coordinate_system(coord_system: Dict[str, Any]) -> Tuple[bool, Optional[str]]`

Validate a coordinate system dictionary.

**Parameters**:
- `coord_system` (Dict[str, Any]): Coordinate system dictionary

**Returns**: Tuple of (is_valid, error_message)

---

## Related

- [Voxelization Module Documentation](../05-modules/voxelization.md) - Module overview
- [All API References](README.md) - Other API references

---

**Parent**: [API Reference](README.md)
