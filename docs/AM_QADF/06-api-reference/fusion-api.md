# Fusion Module API Reference

## Overview

The Fusion module provides multi-source data fusion capabilities for combining signals from different sources. The primary interface is `MultiSourceFusion`, which creates comprehensive fused grids with complete signal preservation and metadata.

## MultiSourceFusion

**Comprehensive multi-source fusion engine** - The recommended interface for production use.

```python
from am_qadf.fusion import MultiSourceFusion, FusionStrategy

fuser = MultiSourceFusion(
    default_strategy: FusionStrategy = FusionStrategy.WEIGHTED_AVERAGE,
    use_quality_scores: bool = True,
    normalize_weights: bool = True
)
```

### Methods

#### `fuse_sources(source_grids: Dict[str, Dict[str, Any]], source_weights: Optional[Dict[str, float]] = None, quality_scores: Optional[Dict[str, float]] = None, fusion_strategy: Optional[FusionStrategy] = None, grid_name: Optional[str] = None, grid_id: Optional[str] = None) -> Dict[str, Any]`

Fuse multiple source grids into a comprehensive fused grid.

**Parameters**:
- `source_grids` (Dict[str, Dict[str, Any]]): Dictionary mapping source names to grid data
  - Each grid data should contain:
    - `signal_arrays` (Dict[str, np.ndarray]): Signal arrays from the source
    - `metadata` (Dict): Grid metadata
    - `grid_id` (str): Source grid ID
    - `grid_name` (str): Source grid name
    - `quality_score` (float): Optional quality score (0-1)
    - `coverage` (float): Optional coverage score (0-1)
- `source_weights` (Optional[Dict[str, float]]): Optional weights for each source
- `quality_scores` (Optional[Dict[str, float]]): Optional quality scores for each source
- `fusion_strategy` (Optional[FusionStrategy]): Fusion strategy (None = use default)
- `grid_name` (Optional[str]): Name for the fused grid
- `grid_id` (Optional[str]): ID for the fused grid

**Returns**: Dictionary with:
- `signal_arrays` (Dict[str, np.ndarray]): All signals (original + source-specific fused + multi-source fused)
- `metadata` (Dict): Comprehensive metadata including:
  - Grid metadata
  - Fusion metadata
  - Signal categorization
  - Source mapping
  - Multi-source fusion metadata
  - Signal statistics
  - Fusion quality metrics
  - Configuration metadata
  - Provenance & lineage

**Example**:
```python
fused_result = fuser.fuse_sources(
    source_grids={
        'laser': {
            'signal_arrays': {'laser_power': array1, 'laser_velocity': array2},
            'metadata': {...},
            'grid_id': '...',
            'quality_score': 0.85
        },
        'ispm': {
            'signal_arrays': {'ispm_temperature': array3},
            'metadata': {...},
            'grid_id': '...',
            'quality_score': 0.88
        }
    },
    source_weights={'laser': 0.5, 'ispm': 0.5},
    quality_scores={'laser': 0.85, 'ispm': 0.88}
)

# Access results
signal_arrays = fused_result['signal_arrays']  # All signals
metadata = fused_result['metadata']  # Complete metadata

# Original signals
laser_power = signal_arrays['laser_power']

# Source-specific fused
laser_power_fused = signal_arrays['laser_power_fused']

# Multi-source fused (if multiple sources have same signal type)
temperature_fused = signal_arrays.get('temperature_fused')
```

---

## VoxelFusion

Core fusion engine for voxel-level signal fusion.

```python
from am_qadf.fusion import VoxelFusion
from am_qadf.synchronization.data_fusion import FusionStrategy

fusion = VoxelFusion(
    default_strategy: FusionStrategy = FusionStrategy.WEIGHTED_AVERAGE,
    use_quality_scores: bool = True
)
```

### Methods

#### `fuse_voxel_signals(voxel_data: Any, signals: List[str], fusion_strategy: Optional[FusionStrategy] = None, quality_scores: Optional[Dict[str, float]] = None, output_signal_name: str = "fused") -> np.ndarray`

Fuse multiple signals in voxel domain.

**Parameters**:
- `voxel_data` (Any): Voxel domain data object (VoxelGrid)
- `signals` (List[str]): List of signal names to fuse
- `fusion_strategy` (Optional[FusionStrategy]): Fusion strategy (None = use default)
- `quality_scores` (Optional[Dict[str, float]]): Quality scores per signal
- `output_signal_name` (str): Name for the fused signal (default: "fused")

**Returns**: Fused signal array

**Example**:
```python
fused = fusion.fuse_voxel_signals(
    voxel_data=grid,
    signals=['hatching_power', 'laser_power'],
    quality_scores={'hatching_power': 0.9, 'laser_power': 0.8}
)
```

#### `fuse_with_quality_weights(voxel_data: Any, signals: List[str], quality_scores: Dict[str, float], output_signal_name: str = "fused_quality_weighted") -> np.ndarray`

Fuse signals using quality-based weighting.

**Parameters**:
- `voxel_data` (Any): Voxel domain data object
- `signals` (List[str]): List of signal names to fuse
- `quality_scores` (Dict[str, float]): Quality scores per signal (0-1, higher is better)
- `output_signal_name` (str): Name for the fused signal

**Returns**: Fused signal array

#### `fuse_per_voxel(voxel_data: Any, signals: List[str], fusion_func: Callable[[List[float]], float], output_signal_name: str = "fused_custom") -> np.ndarray`

Fuse signals using a custom per-voxel function.

**Parameters**:
- `voxel_data` (Any): Voxel domain data object
- `signals` (List[str]): List of signal names to fuse
- `fusion_func` (Callable[[List[float]], float]): Function that takes a list of values and returns fused value
- `output_signal_name` (str): Name for the fused signal

**Returns**: Fused signal array

---

## Fusion Methods

### FusionMethod

Base class for fusion method implementations.

```python
from am_qadf.fusion import FusionMethod
from am_qadf.synchronization.data_fusion import FusionStrategy

# This is a base class - use subclasses instead
method = FusionMethod(strategy: FusionStrategy)
```

#### Methods

##### `fuse(signals: Dict[str, np.ndarray], weights: Optional[Dict[str, float]] = None, quality_scores: Optional[Dict[str, float]] = None, mask: Optional[np.ndarray] = None) -> np.ndarray`

Fuse multiple signals using the configured strategy.

**Parameters**:
- `signals` (Dict[str, np.ndarray]): Dictionary mapping signal names to arrays
- `weights` (Optional[Dict[str, float]]): Optional weights for each signal
- `quality_scores` (Optional[Dict[str, float]]): Optional quality scores for each signal
- `mask` (Optional[np.ndarray]): Optional mask for valid voxels

**Returns**: Fused signal array

---

### WeightedAverageFusion

Weight-based fusion strategy.

```python
from am_qadf.fusion import WeightedAverageFusion

fusion = WeightedAverageFusion(default_weights: Optional[Dict[str, float]] = None)
```

#### Methods

##### `fuse(signals: Dict[str, np.ndarray], weights: Optional[Dict[str, float]] = None, quality_scores: Optional[Dict[str, float]] = None, mask: Optional[np.ndarray] = None) -> np.ndarray`

Fuse signals using weighted average.

**Parameters**:
- `signals` (Dict[str, np.ndarray]): Dictionary mapping signal names to arrays
- `weights` (Optional[Dict[str, float]]): Weights for each signal (if None, uses quality_scores or equal weights)
- `quality_scores` (Optional[Dict[str, float]]): Quality scores for weighting
- `mask` (Optional[np.ndarray]): Mask for valid voxels

**Returns**: Fused signal array

### MedianFusion

Robust median fusion strategy.

```python
from am_qadf.fusion import MedianFusion

fusion = MedianFusion()
```

#### Methods

##### `fuse(signals: Dict[str, np.ndarray], weights: Optional[Dict[str, float]] = None, quality_scores: Optional[Dict[str, float]] = None, mask: Optional[np.ndarray] = None) -> np.ndarray`

Fuse signals using median.

**Parameters**:
- `signals` (Dict[str, np.ndarray]): Dictionary mapping signal names to arrays
- `weights` (Optional[Dict[str, float]]): Ignored for median fusion
- `quality_scores` (Optional[Dict[str, float]]): Ignored for median fusion
- `mask` (Optional[np.ndarray]): Mask for valid voxels

**Returns**: Fused signal array (median at each voxel)

### QualityBasedFusion

Quality-weighted fusion strategy.

```python
from am_qadf.fusion import QualityBasedFusion

fusion = QualityBasedFusion()
```

#### Methods

##### `fuse(signals: Dict[str, np.ndarray], weights: Optional[Dict[str, float]] = None, quality_scores: Optional[Dict[str, float]] = None, mask: Optional[np.ndarray] = None) -> np.ndarray`

Fuse signals using quality-based selection.

**Parameters**:
- `signals` (Dict[str, np.ndarray]): Dictionary mapping signal names to arrays
- `weights` (Optional[Dict[str, float]]): Ignored for quality-based fusion
- `quality_scores` (Optional[Dict[str, float]]): Required for quality-based fusion (falls back to weighted average if None)
- `mask` (Optional[np.ndarray]): Mask for valid voxels

**Returns**: Fused signal array (highest quality signal at each voxel)

### AverageFusion

Simple average fusion strategy.

```python
from am_qadf.fusion import AverageFusion

fusion = AverageFusion()
```

#### Methods

##### `fuse(signals: Dict[str, np.ndarray], weights: Optional[Dict[str, float]] = None, quality_scores: Optional[Dict[str, float]] = None, mask: Optional[np.ndarray] = None) -> np.ndarray`

Fuse signals using simple average.

**Parameters**:
- `signals` (Dict[str, np.ndarray]): Dictionary mapping signal names to arrays
- `weights` (Optional[Dict[str, float]]): Ignored for average fusion
- `quality_scores` (Optional[Dict[str, float]]): Ignored for average fusion
- `mask` (Optional[np.ndarray]): Mask for valid voxels

**Returns**: Fused signal array (average at each voxel)

---

### MaxFusion

Maximum value fusion strategy.

```python
from am_qadf.fusion import MaxFusion

fusion = MaxFusion()
```

#### Methods

##### `fuse(signals: Dict[str, np.ndarray], weights: Optional[Dict[str, float]] = None, quality_scores: Optional[Dict[str, float]] = None, mask: Optional[np.ndarray] = None) -> np.ndarray`

Fuse signals using maximum.

**Parameters**:
- `signals` (Dict[str, np.ndarray]): Dictionary mapping signal names to arrays
- `weights` (Optional[Dict[str, float]]): Ignored for max fusion
- `quality_scores` (Optional[Dict[str, float]]): Ignored for max fusion
- `mask` (Optional[np.ndarray]): Mask for valid voxels

**Returns**: Fused signal array (maximum at each voxel)

---

### MinFusion

Minimum value fusion strategy.

```python
from am_qadf.fusion import MinFusion

fusion = MinFusion()
```

#### Methods

##### `fuse(signals: Dict[str, np.ndarray], weights: Optional[Dict[str, float]] = None, quality_scores: Optional[Dict[str, float]] = None, mask: Optional[np.ndarray] = None) -> np.ndarray`

Fuse signals using minimum.

**Parameters**:
- `signals` (Dict[str, np.ndarray]): Dictionary mapping signal names to arrays
- `weights` (Optional[Dict[str, float]]): Ignored for min fusion
- `quality_scores` (Optional[Dict[str, float]]): Ignored for min fusion
- `mask` (Optional[np.ndarray]): Mask for valid voxels

**Returns**: Fused signal array (minimum at each voxel)

---

### `get_fusion_method(strategy: FusionStrategy) -> FusionMethod`

Get a fusion method instance for the given strategy.

**Parameters**:
- `strategy` (FusionStrategy): Fusion strategy

**Returns**: FusionMethod instance

---

## Fusion Quality

### FusionQualityMetrics

Quality metrics for fusion operations.

```python
from am_qadf.fusion import FusionQualityMetrics

@dataclass
class FusionQualityMetrics:
    fusion_accuracy: float  # Accuracy of fusion (0-1, higher is better)
    signal_consistency: float  # Consistency with source signals (0-1)
    fusion_completeness: float  # Coverage of fused signal (0-1)
    quality_score: float  # Overall quality score (0-1)
    per_signal_accuracy: Dict[str, float]  # Accuracy per source signal
    coverage_ratio: float  # Ratio of voxels with fused data
    residual_errors: Optional[np.ndarray] = None  # Residual errors per voxel
```

#### Methods

##### `to_dict() -> Dict[str, Any]`

Convert metrics to dictionary.

**Returns**: Dictionary representation

---

### FusionQualityAssessor

Assesses quality of fusion operations.

```python
from am_qadf.fusion import FusionQualityAssessor

assessor = FusionQualityAssessor()
```

#### Methods

##### `assess_fusion_quality(fused_array: np.ndarray, source_arrays: Dict[str, np.ndarray], fusion_weights: Optional[Dict[str, float]] = None) -> FusionQualityMetrics`

Assess quality of fused signal.

**Parameters**:
- `fused_array` (np.ndarray): Fused signal array
- `source_arrays` (Dict[str, np.ndarray]): Dictionary mapping signal names to source arrays
- `fusion_weights` (Optional[Dict[str, float]]): Optional weights used for fusion

**Returns**: `FusionQualityMetrics` object

##### `compare_fusion_strategies(voxel_data: Any, signals: List[str], strategies: List[str], quality_scores: Optional[Dict[str, float]] = None) -> Dict[str, FusionQualityMetrics]`

Compare different fusion strategies.

**Parameters**:
- `voxel_data` (Any): Voxel domain data object
- `signals` (List[str]): List of signal names to fuse
- `strategies` (List[str]): List of fusion strategy names to compare ('weighted_average', 'average', 'median', 'max', 'min')
- `quality_scores` (Optional[Dict[str, float]]): Optional quality scores per signal

**Returns**: Dictionary mapping strategy names to FusionQualityMetrics

---

## Related

- [Fusion Module Documentation](../05-modules/fusion.md) - Module overview
- [Fused Grid Structure Reference](../05-modules/fusion-grid-structure.md) - Complete structure reference
- [Synchronization API](synchronization-api.md) - DataFusion base class
- [All API References](README.md) - Other API references

---

**Parent**: [API Reference](README.md)

