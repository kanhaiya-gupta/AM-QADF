# Signal Mapping Module API Reference

## ⭐ CRITICAL MODULE

The Signal Mapping module is the core of the AM-QADF framework, providing interpolation methods for mapping point cloud signals to voxel grids.

## Interpolation Methods

### InterpolationMethod (Base Class)

Abstract base class for all interpolation methods.

```python
from am_qadf.signal_mapping.methods.base import InterpolationMethod

class MyInterpolation(InterpolationMethod):
    def interpolate(self, points, signals, voxel_grid):
        # Implementation
        pass
```

### NearestNeighborInterpolation

Fast nearest neighbor interpolation.

```python
from am_qadf.signal_mapping.methods import NearestNeighborInterpolation

interpolator = NearestNeighborInterpolation()
```

#### Methods

##### `interpolate(points: np.ndarray, signals: Dict[str, np.ndarray], voxel_grid: VoxelGrid) -> VoxelGrid`

Perform nearest neighbor interpolation.

**Parameters**:
- `points` (np.ndarray): Array of points (N, 3)
- `signals` (Dict[str, np.ndarray]): Dictionary mapping signal names to arrays
- `voxel_grid` (VoxelGrid): Target voxel grid

**Returns**: `VoxelGrid` with interpolated signals

**Example**:
```python
result = interpolator.interpolate(
    points=points_array,
    signals={'power': power_array},
    voxel_grid=grid
)
```

### LinearInterpolation

Smooth linear interpolation using k-nearest neighbors.

```python
from am_qadf.signal_mapping.methods import LinearInterpolation

interpolator = LinearInterpolation(k_neighbors: int = 8)
```

#### Methods

##### `interpolate(points: np.ndarray, signals: Dict[str, np.ndarray], voxel_grid: VoxelGrid) -> VoxelGrid`

Perform linear interpolation.

**Parameters**:
- `points` (np.ndarray): Array of points (N, 3)
- `signals` (Dict[str, np.ndarray]): Dictionary mapping signal names to arrays
- `voxel_grid` (VoxelGrid): Target voxel grid
- `k_neighbors` (int): Number of nearest neighbors to use (default: 8)

**Returns**: `VoxelGrid` with interpolated signals

### IDWInterpolation

Inverse Distance Weighting interpolation.

```python
from am_qadf.signal_mapping.methods import IDWInterpolation

interpolator = IDWInterpolation(power: float = 2.0, k_neighbors: int = 8)
```

#### Methods

##### `interpolate(points: np.ndarray, signals: Dict[str, np.ndarray], voxel_grid: VoxelGrid) -> VoxelGrid`

Perform IDW interpolation.

**Parameters**:
- `points` (np.ndarray): Array of points (N, 3)
- `signals` (Dict[str, np.ndarray]): Dictionary mapping signal names to arrays
- `voxel_grid` (VoxelGrid): Target voxel grid
- `power` (float): IDW power parameter (default: 2.0)
- `k_neighbors` (int): Number of nearest neighbors (default: 8)

**Returns**: `VoxelGrid` with interpolated signals

### GaussianKDEInterpolation

Gaussian Kernel Density Estimation interpolation.

```python
from am_qadf.signal_mapping.methods import GaussianKDEInterpolation

interpolator = GaussianKDEInterpolation(
    bandwidth: Optional[float] = None,
    adaptive: bool = False
)
```

#### Methods

##### `interpolate(points: np.ndarray, signals: Dict[str, np.ndarray], voxel_grid: VoxelGrid) -> VoxelGrid`

Perform Gaussian KDE interpolation.

**Parameters**:
- `points` (np.ndarray): Array of points (N, 3)
- `signals` (Dict[str, np.ndarray]): Dictionary mapping signal names to arrays
- `voxel_grid` (VoxelGrid): Target voxel grid
- `bandwidth` (Optional[float]): KDE bandwidth parameter (None = auto-estimate using Silverman's rule)
- `adaptive` (bool): Whether to use adaptive bandwidth (default: False)

**Returns**: `VoxelGrid` with interpolated signals

**Note**: Method name is `'gaussian_kde'` (not `'kde'`)

### RBFInterpolation

Radial Basis Functions interpolation providing exact interpolation at data points with smooth interpolation between points.

```python
from am_qadf.signal_mapping.methods import RBFInterpolation

interpolator = RBFInterpolation(
    kernel: str = 'gaussian',
    epsilon: Optional[float] = None,
    smoothing: float = 0.0,
    use_sparse: bool = False,
    max_points: Optional[int] = None
)
```

#### Methods

##### `interpolate(points: np.ndarray, signals: Dict[str, np.ndarray], voxel_grid: VoxelGrid) -> VoxelGrid`

Perform RBF interpolation.

**Parameters**:
- `points` (np.ndarray): Array of points (N, 3)
- `signals` (Dict[str, np.ndarray]): Dictionary mapping signal names to arrays
- `voxel_grid` (VoxelGrid): Target voxel grid

**Returns**: `VoxelGrid` with interpolated signals

**Parameters**:
- `kernel` (str): RBF kernel type. Options:
  - `'gaussian'`: exp(-(epsilon*r)²) - Smooth, bounded
  - `'multiquadric'`: sqrt(1 + (epsilon*r)²) - General purpose
  - `'inverse_multiquadric'`: 1/sqrt(1 + (epsilon*r)²) - Smooth, bounded
  - `'thin_plate_spline'`: r² * log(r) - Exact interpolation
  - `'linear'`: r - Simple, fast
  - `'cubic'`: r³ - Smooth
  - `'quintic'`: r⁵ - Very smooth
- `epsilon` (Optional[float]): Shape parameter for kernel. If None, auto-estimated based on point distribution
- `smoothing` (float): Smoothing parameter. 0.0 = exact interpolation at data points (default)
- `use_sparse` (bool): Use sparse matrices for large N (experimental, default: False)
- `max_points` (Optional[int]): Maximum points before warning about performance (None = no limit)

**Complexity**: O(N³) - Use Spark backend for large datasets (N > 10,000)

**Example**:
```python
# Basic usage
interpolator = RBFInterpolation(kernel='gaussian')
result = interpolator.interpolate(
    points=points_array,
    signals={'power': power_array},
    voxel_grid=grid
)

# With custom parameters
interpolator = RBFInterpolation(
    kernel='thin_plate_spline',
    epsilon=1.0,
    smoothing=0.0  # Exact interpolation
)
result = interpolator.interpolate(points, signals, grid)

# With auto-estimated epsilon
interpolator = RBFInterpolation(
    kernel='gaussian',
    epsilon=None  # Will be auto-estimated
)
result = interpolator.interpolate(points, signals, grid)
```

**Note**: 
- Method name is `'rbf'`
- For large datasets (N > 10,000), consider using Spark backend or alternative methods (linear, IDW, KDE) for better performance
- RBF provides exact interpolation at data points when smoothing=0.0
- Auto-estimated epsilon uses average nearest neighbor distance as a heuristic

---

## Execution Strategies

### Sequential Execution

Single-threaded sequential execution (with optional parallel/Spark support).

```python
from am_qadf.signal_mapping.execution.sequential import interpolate_to_voxels

result = interpolate_to_voxels(
    points: np.ndarray,
    signals: Dict[str, np.ndarray],
    voxel_grid: VoxelGrid,
    method: str = 'nearest',
    use_vectorized: bool = True,
    use_parallel: bool = False,
    use_spark: bool = False,
    spark_session: Optional[SparkSession] = None,
    max_workers: Optional[int] = None,
    chunk_size: Optional[int] = None,
    **method_kwargs
) -> VoxelGrid
```

**Parameters**:
- `points` (np.ndarray): Array of points (N, 3)
- `signals` (Dict[str, np.ndarray]): Dictionary of signal arrays
- `voxel_grid` (VoxelGrid): Target voxel grid
- `method` (str): Interpolation method ('nearest', 'linear', 'idw', 'gaussian_kde')
- `use_vectorized` (bool): Use vectorized implementation (default: True)
- `use_parallel` (bool): Use parallel processing (default: False)
- `use_spark` (bool): Use Spark for distributed processing (default: False)
- `spark_session` (Optional[SparkSession]): Spark session (required if use_spark=True)
- `max_workers` (Optional[int]): Max worker threads/processes (for parallel)
- `chunk_size` (Optional[int]): Points per chunk (for parallel, auto-calculated if None)
- `**method_kwargs`: Method-specific parameters:
  - `'linear'`: `k_neighbors` (int), `radius` (float)
  - `'idw'`: `power` (float), `k_neighbors` (int), `radius` (float)
  - `'gaussian_kde'`: `bandwidth` (float), `adaptive` (bool)

**Returns**: `VoxelGrid` with interpolated signals

### Parallel Execution

Multi-threaded/process parallel execution.

#### Option 1: Using interpolate_to_voxels with use_parallel flag

```python
from am_qadf.signal_mapping.execution.sequential import interpolate_to_voxels

result = interpolate_to_voxels(
    points, signals, voxel_grid,
    method='linear',
    use_parallel=True,
    max_workers=4,
    chunk_size=1000
)
```

#### Option 2: Using ParallelInterpolationExecutor directly

```python
from am_qadf.signal_mapping.execution.parallel import ParallelInterpolationExecutor

executor = ParallelInterpolationExecutor(
    max_workers: Optional[int] = None,
    chunk_size: Optional[int] = None,
    use_processes: bool = True
)
```

**Parameters**:
- `max_workers` (Optional[int]): Maximum worker threads/processes (None = cpu_count())
- `chunk_size` (Optional[int]): Points per chunk (None = auto-calculated)
- `use_processes` (bool): Use ProcessPoolExecutor (True) or ThreadPoolExecutor (False)

#### Methods

##### `execute_parallel(method: str, points: np.ndarray, signals: Dict[str, np.ndarray], voxel_grid: VoxelGrid, method_kwargs: Optional[Dict] = None) -> VoxelGrid`

Execute interpolation in parallel.

**Parameters**:
- `method` (str): Interpolation method ('nearest', 'linear', 'idw', 'gaussian_kde')
- `points` (np.ndarray): Array of points (N, 3)
- `signals` (Dict[str, np.ndarray]): Dictionary of signal arrays
- `voxel_grid` (VoxelGrid): Target voxel grid
- `method_kwargs` (Optional[Dict]): Method-specific parameters

**Returns**: `VoxelGrid` with interpolated signals

### Spark Execution

Distributed execution with Apache Spark.

```python
from am_qadf.signal_mapping.execution.spark import interpolate_to_voxels_spark

result = interpolate_to_voxels_spark(
    spark_session: SparkSession,
    points: np.ndarray,
    signals: Dict[str, np.ndarray],
    voxel_grid: VoxelGrid,
    method: str = 'nearest'
) -> VoxelGrid
```

**Parameters**:
- `spark_session` (SparkSession): Apache Spark session
- `points` (np.ndarray): Array of points (N, 3)
- `signals` (Dict[str, np.ndarray]): Dictionary of signal arrays
- `voxel_grid` (VoxelGrid): Target voxel grid
- `method` (str): Interpolation method ('nearest', 'linear', 'idw', 'gaussian_kde')

**Returns**: `VoxelGrid` with interpolated signals

---

## Specialized Functions

### `interpolate_hatching_paths`

Interpolate hatching paths (polylines) to voxel grid.

```python
from am_qadf.signal_mapping.execution.sequential import interpolate_hatching_paths

result = interpolate_hatching_paths(
    paths: List[np.ndarray],
    signals: Dict[str, List[np.ndarray]],
    voxel_grid: VoxelGrid,
    points_per_mm: float = 10.0,
    interpolation_method: str = 'nearest',
    use_parallel: bool = False,
    use_spark: bool = False,
    spark_session: Optional[SparkSession] = None,
    max_workers: Optional[int] = None,
    **method_kwargs
) -> VoxelGrid
```

**Parameters**:
- `paths` (List[np.ndarray]): List of path arrays, each shape (N, 3) with (x, y, z) coordinates
- `signals` (Dict[str, List[np.ndarray]]): Dictionary mapping signal names to lists of arrays (one per path)
- `voxel_grid` (VoxelGrid): Target voxel grid
- `points_per_mm` (float): Sampling density along paths (default: 10.0)
- `interpolation_method` (str): Method ('nearest', 'linear', 'idw', 'gaussian_kde')
- `use_parallel` (bool): Use parallel processing (default: False)
- `use_spark` (bool): Use Spark (default: False)
- `spark_session` (Optional[SparkSession]): Spark session
- `max_workers` (Optional[int]): Max workers for parallel
- `**method_kwargs`: Method-specific parameters

**Returns**: `VoxelGrid` with interpolated signals

**Example**:
```python
paths = [path1, path2, path3]  # Each path is (N, 3) array
signals = {
    'power': [power1, power2, power3],  # One array per path
    'velocity': [vel1, vel2, vel3]
}

result = interpolate_hatching_paths(
    paths=paths,
    signals=signals,
    voxel_grid=grid,
    points_per_mm=10.0,
    interpolation_method='linear'
)
```

---

## Utility Functions

### Coordinate Utilities

```python
from am_qadf.signal_mapping.utils.coordinate_utils import (
    transform_coordinates,
    align_to_voxel_grid
)
```

#### `transform_coordinates(points: np.ndarray, from_system: Dict[str, Any], to_system: Dict[str, Any]) -> np.ndarray`

Transform points between coordinate systems.

**Parameters**:
- `points` (np.ndarray): Array of points (N, 3)
- `from_system` (Dict[str, Any]): Source coordinate system
- `to_system` (Dict[str, Any]): Target coordinate system

**Returns**: Transformed points (N, 3)

#### `align_to_voxel_grid(points: np.ndarray, voxel_grid_origin: Tuple[float, float, float], voxel_resolution: float) -> np.ndarray`

Align points to voxel grid coordinate system.

**Parameters**:
- `points` (np.ndarray): Array of points (N, 3)
- `voxel_grid_origin` (Tuple[float, float, float]): Grid origin (x, y, z)
- `voxel_resolution` (float): Voxel resolution in mm

**Returns**: Aligned points (N, 3)

---

## Method Registry

The framework maintains a registry of available interpolation methods:

```python
INTERPOLATION_METHODS = {
    'nearest': NearestNeighborInterpolation,
    'linear': LinearInterpolation,
    'idw': IDWInterpolation,
    'gaussian_kde': GaussianKDEInterpolation,
}
```

**Note**: Use `'gaussian_kde'` (not `'kde'`) as the method name.

---

## Related

- [Signal Mapping Module Documentation](../05-modules/signal-mapping.md) - Module overview
- [All API References](README.md) - Other API references

---

**Parent**: [API Reference](README.md)

