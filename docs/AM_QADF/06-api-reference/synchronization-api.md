# Synchronization Module API Reference

## Overview

The Synchronization module provides temporal and spatial alignment capabilities for multi-source data. **Spatial alignment** is performed via bounding-box corner correspondence (see [SPATIAL_ALIGNMENT_DESIGN.md](../../Infrastructure/SPATIAL_ALIGNMENT_DESIGN.md)); the main Python API is `UnifiedQueryClient.query_and_transform_points`.

---

## UnifiedQueryClient.query_and_transform_points

Query points from multiple sources, compute transformation from bbox corners (24 permutations × 56 triplets), validate, and return transformed points in the reference coordinate system.

```python
from am_qadf.query import UnifiedQueryClient

client = UnifiedQueryClient(...)
result = client.query_and_transform_points(
    model_id: str,
    source_types: List[str],
    reference_source: str = "hatching",
    layer_range: Optional[Tuple[int, int]] = None,
    bbox: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None,
    use_full_extent_for_transform: bool = True,
    validation_tolerance: float = 1e-6,
    save_processed_data: bool = False,
    mongo_uri: Optional[str] = None,
    db_name: Optional[str] = None,
) -> Dict[str, Any]
```

### Parameters

| Parameter | Type | Description |
|-----------|------|--------------|
| **model_id** | str | Model UUID |
| **source_types** | List[str] | Source keys, e.g. `['hatching', 'ispm_thermal', 'ispm_optical']` |
| **reference_source** | str | Reference frame (e.g. `"hatching"`). Non-reference sources are transformed to this. |
| **layer_range** | Optional[Tuple[int,int]] | Optional `(layer_start, layer_end)` to filter which points are returned/saved |
| **bbox** | Optional[tuple] | Optional `((x_min,y_min,z_min), (x_max,y_max,z_max))` to filter points returned/saved |
| **use_full_extent_for_transform** | bool | If `True` (default), bboxes and transform are computed from **full** data; filters only affect returned/saved points. If `False`, bbox/transform use the queried subset. |
| **validation_tolerance** | float | Minimum tolerance; adaptive tolerance is `max(0.01 * max_extent, 1e-3, validation_tolerance)` (1% error bar) |
| **save_processed_data** | bool | If `True`, save transformed points via MongoDBWriter |
| **mongo_uri**, **db_name** | Optional[str] | Required when `save_processed_data=True` |

### Returns

Dict with:

| Key | Description |
|-----|--------------|
| **transformed_points** | `Dict[str, np.ndarray]`: `{ source_type: (N, 3) }` points in reference frame |
| **signals** | `Dict[str, Dict[str, np.ndarray]]`: signals per source |
| **unified_bounds** | `BoundingBox`: union of returned point sets |
| **transformations** | Per non-reference source: `matrix`, `quality`, `fit_errors`, `best_fit`, `fit_errors_summary`, optional `correspondence_validation` |
| **validation_results** | `Dict[str, ValidationResult]`: pass/fail and errors (uses best_ref_corners internally) |
| **raw_results** | `Dict[str, QueryResult]`: query results before transform |

### transformations[source] structure

- **matrix** (np.ndarray 4×4): Similarity transform from source to reference
- **quality**: `TransformationQuality` (rms_error, max_error, mean_error, alignment_quality, confidence)
- **fit_errors**: List of 24×56 per-fit dicts: `permutation_index`, `triplet_index`, `max_error`, `mean_error`, `rms_error`
- **best_fit**: Dict for the chosen fit: `permutation_index`, `triplet_index`, `max_error`, `mean_error`, `rms_error`
- **fit_errors_summary**: `min_max_error`, `max_max_error`, `num_fits` (1344)
- **correspondence_validation** (optional): `mean_distance`, `max_distance`, `num_pairs` (9), `type` ("corners_and_centre")

### Requirements

- **am_qadf_native** (C++ bindings) must be built; provides `TransformationComputer`, `TransformationValidator`, `PointTransformer`, `UnifiedBoundsComputer`, `BoundingBox`, `points_to_eigen_matrix`, etc.

---

## TimePoint

Represents a point in time with associated data.

```python
from am_qadf.synchronization import TimePoint

time_point = TimePoint(
    timestamp: float,
    layer_index: Optional[int] = None,
    z_height: Optional[float] = None,
    data: Optional[Dict[str, Any]] = None
)
```

### Attributes

- **timestamp** (float): Timestamp in seconds since build start
- **layer_index** (Optional[int]): Layer index
- **z_height** (Optional[float]): Z height in mm
- **data** (Optional[Dict[str, Any]]): Associated data dictionary

---

## LayerTimeMapper

Maps build layers to timestamps and vice versa.

```python
from am_qadf.synchronization import LayerTimeMapper

mapper = LayerTimeMapper(
    layer_thickness: float = 0.04,
    base_z: float = 0.0,
    time_per_layer: Optional[float] = None
)
```

### Methods

#### `add_layer_time(layer_index: int, timestamp: float, z_height: Optional[float] = None) -> None`

Add a known layer-time mapping.

**Parameters**:
- `layer_index` (int): Layer index
- `timestamp` (float): Timestamp in seconds
- `z_height` (Optional[float]): Z height in mm (auto-computed if None)

#### `layer_to_z(layer_index: int) -> float`

Convert layer index to Z height.

**Parameters**:
- `layer_index` (int): Layer index

**Returns**: Z height in mm

#### `z_to_layer(z_height: float) -> int`

Convert Z height to layer index.

**Parameters**:
- `z_height` (float): Z height in mm

**Returns**: Layer index

#### `layer_to_time(layer_index: int) -> Optional[float]`

Convert layer index to timestamp.

**Parameters**:
- `layer_index` (int): Layer index

**Returns**: Timestamp in seconds or None if not mapped

#### `time_to_layer(timestamp: float) -> Optional[int]`

Convert timestamp to layer index.

**Parameters**:
- `timestamp` (float): Timestamp in seconds

**Returns**: Layer index or None if not mapped

---

## TemporalAligner

Aligns data temporally using layer-based mapping.

```python
from am_qadf.synchronization import TemporalAligner, LayerTimeMapper

aligner = TemporalAligner(layer_mapper: Optional[LayerTimeMapper] = None)
```

### Methods

#### `add_time_point(timestamp: float, layer_index: Optional[int] = None, z_height: Optional[float] = None, data: Optional[Dict[str, Any]] = None) -> None`

Add a time point with associated data.

**Parameters**:
- `timestamp` (float): Timestamp in seconds
- `layer_index` (Optional[int]): Layer index (auto-computed if None)
- `z_height` (Optional[float]): Z height (auto-computed if None)
- `data` (Optional[Dict[str, Any]]): Associated data dictionary

#### `align_to_layers(target_layers: List[int], interpolation_method: str = 'linear') -> List[Dict[str, Any]]`

Align temporal data to specific layers.

**Parameters**:
- `target_layers` (List[int]): List of target layer indices
- `interpolation_method` (str): Interpolation method ('linear', 'nearest', 'zero')

**Returns**: List of data dictionaries, one per target layer

#### `get_layer_data(layer_index: int) -> Optional[Dict[str, Any]]`

Get data for a specific layer.

**Parameters**:
- `layer_index` (int): Layer index

**Returns**: Data dictionary or None

#### `handle_missing_temporal_data(required_layers: List[int], default_data: Optional[Dict[str, Any]] = None) -> Dict[int, Dict[str, Any]]`

Handle missing temporal data by filling with defaults or interpolation.

**Parameters**:
- `required_layers` (List[int]): List of required layer indices
- `default_data` (Optional[Dict[str, Any]]): Default data for missing layers

**Returns**: Dictionary mapping layer_index to data

---

## TransformationMatrix

Represents a 4x4 homogeneous transformation matrix.

```python
from am_qadf.synchronization import TransformationMatrix

# Create identity matrix
transform = TransformationMatrix.identity()

# Create translation
transform = TransformationMatrix.translation(tx: float, ty: float, tz: float)

# Create rotation (angle in radians)
transform = TransformationMatrix.rotation(axis: str, angle: float)

# Create scaling
transform = TransformationMatrix.scale(sx: float, sy: float, sz: float)
```

### Class Methods

#### `identity() -> TransformationMatrix`

Create identity transformation matrix.

#### `translation(tx: float, ty: float, tz: float) -> TransformationMatrix`

Create translation transformation.

**Parameters**:
- `tx, ty, tz` (float): Translation in x, y, z directions

#### `rotation(axis: str, angle: float) -> TransformationMatrix`

Create rotation transformation.

**Parameters**:
- `axis` (str): Rotation axis ('x', 'y', 'z')
- `angle` (float): Rotation angle in radians

#### `scale(sx: float, sy: float, sz: float) -> TransformationMatrix`

Create scaling transformation.

**Parameters**:
- `sx, sy, sz` (float): Scale factors in x, y, z directions

### Methods

#### `apply(points: np.ndarray) -> np.ndarray`

Apply transformation to points.

**Parameters**:
- `points` (np.ndarray): Array of points (N, 3) or (3,)

**Returns**: Transformed points

#### `inverse() -> TransformationMatrix`

Compute inverse transformation.

**Returns**: Inverse transformation matrix

---

## SpatialTransformer

Transforms data between coordinate systems.

```python
from am_qadf.synchronization import SpatialTransformer, TransformationMatrix

transformer = SpatialTransformer()
```

### Methods

#### `register_transformation(name: str, transformation: TransformationMatrix) -> None`

Register a named transformation.

**Parameters**:
- `name` (str): Transformation name
- `transformation` (TransformationMatrix): Transformation matrix

#### `get_transformation(name: str) -> Optional[TransformationMatrix]`

Get a named transformation.

**Parameters**:
- `name` (str): Transformation name

**Returns**: TransformationMatrix or None

#### `transform_points(points: np.ndarray, transformation_name: Optional[str] = None, transformation: Optional[TransformationMatrix] = None) -> np.ndarray`

Transform points using a transformation.

**Parameters**:
- `points` (np.ndarray): Array of points (N, 3)
- `transformation_name` (Optional[str]): Name of registered transformation
- `transformation` (Optional[TransformationMatrix]): Direct transformation matrix

**Returns**: Transformed points (N, 3)

#### `align_coordinate_systems(source_points: np.ndarray, target_points: np.ndarray, method: str = 'umeyama') -> TransformationMatrix`

Align coordinate systems using point correspondences.

**Parameters**:
- `source_points` (np.ndarray): Source points (N, 3)
- `target_points` (np.ndarray): Target points (N, 3)
- `method` (str): Alignment method ('umeyama' for rigid transformation)

**Returns**: TransformationMatrix from source to target

---

## TransformationManager

Manages coordinate system transformations.

```python
from am_qadf.synchronization import TransformationManager

manager = TransformationManager()
```

### Methods

#### `register_coordinate_system(name: str, origin: Optional[Tuple[float, float, float]] = None, axes: Optional[Dict[str, Tuple[float, float, float]]] = None) -> None`

Register a coordinate system.

**Parameters**:
- `name` (str): Coordinate system name
- `origin` (Optional[Tuple[float, float, float]]): Origin point (x, y, z)
- `axes` (Optional[Dict[str, Tuple[float, float, float]]]): Axis vectors {'x': (1,0,0), 'y': (0,1,0), 'z': (0,0,1)}

#### `register_transformation(source_system: str, target_system: str, transformation: TransformationMatrix) -> None`

Register a transformation between coordinate systems.

**Parameters**:
- `source_system` (str): Source coordinate system name
- `target_system` (str): Target coordinate system name
- `transformation` (TransformationMatrix): Transformation matrix

#### `get_transformation(from_system: str, to_system: str, visited: Optional[set] = None) -> Optional[TransformationMatrix]`

Get transformation matrix between systems.

**Parameters**:
- `from_system` (str): Source coordinate system name
- `to_system` (str): Target coordinate system name
- `visited` (Optional[set]): Internal parameter for cycle detection

**Returns**: TransformationMatrix or None

---

## DataFusion

Base class for data fusion operations.

```python
from am_qadf.synchronization import DataFusion
from am_qadf.synchronization.data_fusion import FusionStrategy

fusion = DataFusion(
    default_strategy: FusionStrategy = FusionStrategy.WEIGHTED_AVERAGE,
    default_weights: Optional[Dict[str, float]] = None
)
```

### Methods

#### `register_source_quality(source_name: str, quality_score: float) -> None`

Register quality score for a data source.

**Parameters**:
- `source_name` (str): Source name
- `quality_score` (float): Quality score (0.0 to 1.0, higher is better)

#### `compute_weights(source_names: List[str], use_quality: bool = True) -> np.ndarray`

Compute weights for data sources.

**Parameters**:
- `source_names` (List[str]): List of source names
- `use_quality` (bool): Whether to use quality scores for weighting

**Returns**: Array of normalized weights (sums to 1.0)

#### `fuse_signals(signals: Dict[str, np.ndarray], strategy: Optional[FusionStrategy] = None, weights: Optional[Dict[str, float]] = None, mask: Optional[np.ndarray] = None) -> np.ndarray`

Fuse multiple signals.

**Parameters**:
- `signals` (Dict[str, np.ndarray]): Dictionary mapping signal names to arrays
- `strategy` (Optional[FusionStrategy]): Fusion strategy (None = use default)
- `weights` (Optional[Dict[str, float]]): Custom weights for each signal
- `mask` (Optional[np.ndarray]): Optional mask for valid voxels (True = valid)

**Returns**: Fused signal array

#### `fuse_multiple_signals(signal_dicts: List[Dict[str, np.ndarray]], signal_names: List[str], strategy: Optional[FusionStrategy] = None) -> Dict[str, np.ndarray]`

Fuse multiple signals from multiple sources.

**Parameters**:
- `signal_dicts` (List[Dict[str, np.ndarray]]): List of signal dictionaries, one per source
- `signal_names` (List[str]): List of signal names to fuse
- `strategy` (Optional[FusionStrategy]): Fusion strategy

**Returns**: Dictionary of fused signals

#### `handle_conflicts(signals: Dict[str, np.ndarray], conflict_threshold: float = 0.1, method: str = 'weighted_average') -> Tuple[np.ndarray, np.ndarray]`

Detect and handle conflicting data between sources.

**Parameters**:
- `signals` (Dict[str, np.ndarray]): Dictionary mapping source names to signal arrays
- `conflict_threshold` (float): Relative difference threshold for conflict detection
- `method` (str): Method to resolve conflicts ('weighted_average', 'quality_based', 'median')

**Returns**: Tuple of (fused_signal, conflict_mask)

#### `compute_fusion_quality(signals: Dict[str, np.ndarray], fused: np.ndarray) -> Dict[str, float]`

Compute quality metrics for fused signal.

**Parameters**:
- `signals` (Dict[str, np.ndarray]): Dictionary of source signals
- `fused` (np.ndarray): Fused signal array

**Returns**: Dictionary of quality metrics (coefficient_of_variation, agreement, coverage, mean, std)

---

## FusionStrategy

Enumeration of fusion strategies.

```python
from am_qadf.synchronization.data_fusion import FusionStrategy

# Available strategies:
FusionStrategy.AVERAGE          # Simple average
FusionStrategy.WEIGHTED_AVERAGE # Weighted by quality/confidence
FusionStrategy.MEDIAN           # Median value
FusionStrategy.MAX              # Maximum value
FusionStrategy.MIN              # Minimum value
FusionStrategy.FIRST            # First available value
FusionStrategy.LAST             # Last available value
FusionStrategy.QUALITY_BASED    # Use highest quality source
```

---

## Related

- [Synchronization Module Documentation](../05-modules/synchronization.md) - Module overview
- [Fusion API](fusion-api.md) - Voxel-level fusion
- [All API References](README.md) - Other API references

---

**Parent**: [API Reference](README.md)

