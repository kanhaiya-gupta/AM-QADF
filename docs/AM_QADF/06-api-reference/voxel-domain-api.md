# Voxel Domain Module API Reference

## Overview

The Voxel Domain module is the main orchestrator for creating and managing unified voxel representations of multi-source data.

## VoxelDomainClient

Main client for voxel domain operations.

```python
from am_qadf.voxel_domain import VoxelDomainClient

client = VoxelDomainClient(
    unified_query_client: UnifiedQueryClient,
    base_resolution: float = 1.0,
    adaptive: bool = False,
    mongo_client: Optional[MongoDBClient] = None
)
```

### Methods

#### `create_voxel_grid(model_id: str, resolution: Optional[float] = None, bbox_min: Optional[Tuple[float, float, float]] = None, bbox_max: Optional[Tuple[float, float, float]] = None, adaptive: Optional[bool] = None, signals: Optional[List[str]] = None) -> Union[VoxelGrid, AdaptiveResolutionGrid]`

Create a voxel grid for a given model.

**Parameters**:
- `model_id` (str): Model ID to create grid for
- `resolution` (Optional[float]): Voxel resolution in mm (uses base_resolution if None)
- `bbox_min` (Optional[Tuple[float, float, float]]): Minimum bounding box (uses STL bbox if None)
- `bbox_max` (Optional[Tuple[float, float, float]]): Maximum bounding box (uses STL bbox if None)
- `adaptive` (Optional[bool]): Whether to use adaptive resolution (uses self.adaptive if None)
- `signals` (Optional[List[str]]): List of signal names to include (optional, for metadata)

**Returns**: `VoxelGrid` or `AdaptiveResolutionGrid` instance

**Example**:
```python
grid = client.create_voxel_grid(
    model_id="my_model",
    resolution=0.5,
    adaptive=True
)
```

#### `map_signals_to_voxels(model_id: str, voxel_grid: VoxelGrid, sources: List[str], signals: Optional[List[str]] = None, method: str = 'nearest', n_workers: Optional[int] = None) -> VoxelGrid`

Map signals from data sources to voxel grid.

**Parameters**:
- `model_id` (str): Model ID
- `voxel_grid` (VoxelGrid): Target voxel grid
- `sources` (List[str]): List of source names ('hatching', 'laser', 'ct', 'ispm', etc.)
- `signals` (Optional[List[str]]): List of signal names to map (None = all available)
- `method` (str): Interpolation method ('nearest', 'linear', 'idw', 'kde')
- `n_workers` (Optional[int]): Number of workers for parallel execution (None = sequential)

**Returns**: `VoxelGrid` with mapped signals

**Example**:
```python
grid = client.map_signals_to_voxels(
    model_id="my_model",
    voxel_grid=grid,
    sources=['hatching', 'laser'],
    method='linear',
    n_workers=4
)
```

#### `fuse_signals(voxel_grid: VoxelGrid, signals: List[str], fusion_strategy: Optional[FusionStrategy] = None, quality_scores: Optional[Dict[str, float]] = None, output_signal_name: str = 'fused') -> VoxelGrid`

Fuse multiple signals in voxel grid.

**Parameters**:
- `voxel_grid` (VoxelGrid): Voxel grid with signals
- `signals` (List[str]): List of signal names to fuse
- `fusion_strategy` (Optional[FusionStrategy]): Fusion strategy (None = weighted average)
- `quality_scores` (Optional[Dict[str, float]]): Quality scores per signal
- `output_signal_name` (str): Name for fused signal (default: 'fused')

**Returns**: `VoxelGrid` with fused signal

**Example**:
```python
grid = client.fuse_signals(
    voxel_grid=grid,
    signals=['hatching_power', 'laser_power'],
    quality_scores={'hatching_power': 0.9, 'laser_power': 0.8}
)
```

#### `perform_quality_assessment(voxel_grid: VoxelGrid, signals: Optional[List[str]] = None) -> Dict[str, Any]`

Perform quality assessment on voxel grid.

**Parameters**:
- `voxel_grid` (VoxelGrid): Voxel grid to assess
- `signals` (Optional[List[str]]): List of signal names (None = all signals)

**Returns**: Dictionary of quality metrics

#### `perform_analytics(voxel_grid: VoxelGrid, analysis_type: str = 'statistical', **kwargs) -> Dict[str, Any]`

Perform analytics on voxel grid.

**Parameters**:
- `voxel_grid` (VoxelGrid): Voxel grid to analyze
- `analysis_type` (str): Analysis type ('statistical', 'sensitivity', 'process')
- `**kwargs`: Analysis-specific parameters

**Returns**: Dictionary of analysis results

#### `save_voxel_grid(voxel_grid: VoxelGrid, model_id: str, grid_id: Optional[str] = None) -> str`

Save voxel grid to storage.

**Parameters**:
- `voxel_grid` (VoxelGrid): Voxel grid to save
- `model_id` (str): Model ID
- `grid_id` (Optional[str]): Grid ID (auto-generated if None)

**Returns**: Grid ID

#### `load_voxel_grid(model_id: str, grid_id: str) -> VoxelGrid`

Load voxel grid from storage.

**Parameters**:
- `model_id` (str): Model ID
- `grid_id` (str): Grid ID

**Returns**: `VoxelGrid` instance

---

## VoxelGridStorage

Storage interface for voxel grids.

```python
from am_qadf.voxel_domain import VoxelGridStorage

storage = VoxelGridStorage(mongo_client: MongoDBClient)
```

### Methods

#### `save(grid: VoxelGrid, model_id: str, grid_id: str) -> None`

Save voxel grid.

**Parameters**:
- `grid` (VoxelGrid): Voxel grid to save
- `model_id` (str): Model ID
- `grid_id` (str): Grid ID

#### `load(model_id: str, grid_id: str) -> VoxelGrid`

Load voxel grid.

**Parameters**:
- `model_id` (str): Model ID
- `grid_id` (str): Grid ID

**Returns**: `VoxelGrid` instance

#### `list_grids(model_id: str) -> List[str]`

List all grid IDs for a model.

**Parameters**:
- `model_id` (str): Model ID

**Returns**: List of grid IDs

---

## Related

- [Voxel Domain Module Documentation](../05-modules/voxel-domain.md) - Module overview
- [All API References](README.md) - Other API references

---

**Parent**: [API Reference](README.md)

