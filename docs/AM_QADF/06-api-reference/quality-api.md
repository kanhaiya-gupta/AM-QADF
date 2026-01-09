# Quality Module API Reference

## Overview

The Quality module provides comprehensive quality assessment capabilities for voxel domain data.

## QualityAssessmentClient

Main client for quality assessment operations.

```python
from am_qadf.quality import QualityAssessmentClient

client = QualityAssessmentClient(
    max_acceptable_error: float = 0.1,  # mm
    noise_floor: float = 1e-6
)
```

### Methods

#### `assess_data_quality(voxel_data: Any, signals: Optional[List[str]] = None, layer_range: Optional[Tuple[int, int]] = None) -> DataQualityMetrics`

Assess overall data quality.

**Parameters**:
- `voxel_data` (Any): Voxel domain data object (VoxelGrid)
- `signals` (Optional[List[str]]): List of signal names to check (None = all signals)
- `layer_range` (Optional[Tuple[int, int]]): Layer range `(min_layer, max_layer)`

**Returns**: `DataQualityMetrics` object

#### `assess_signal_quality(signal_name: str, signal_array: np.ndarray, noise_estimate: Optional[np.ndarray] = None, measurement_uncertainty: Optional[float] = None, store_maps: bool = True) -> SignalQualityMetrics`

Assess quality for a single signal.

**Parameters**:
- `signal_name` (str): Name of the signal
- `signal_array` (np.ndarray): Signal array
- `noise_estimate` (Optional[np.ndarray]): Optional noise estimate
- `measurement_uncertainty` (Optional[float]): Optional measurement uncertainty
- `store_maps` (bool): Whether to store per-voxel quality maps

**Returns**: `SignalQualityMetrics` object

#### `assess_all_signals(voxel_data: Any, signals: Optional[List[str]] = None, store_maps: bool = True) -> Dict[str, SignalQualityMetrics]`

Assess quality for all signals.

**Parameters**:
- `voxel_data` (Any): Voxel domain data object
- `signals` (Optional[List[str]]): List of signal names (None = all signals)
- `store_maps` (bool): Whether to store per-voxel quality maps

**Returns**: Dictionary mapping signal names to `SignalQualityMetrics`

#### `assess_alignment_accuracy(voxel_data: Any, coordinate_transformer: Optional[Any] = None, reference_data: Optional[Any] = None) -> AlignmentAccuracyMetrics`

Assess alignment accuracy.

**Parameters**:
- `voxel_data` (Any): Voxel domain data object
- `coordinate_transformer` (Optional[Any]): Optional coordinate system transformer
- `reference_data` (Optional[Any]): Optional reference data for comparison

**Returns**: `AlignmentAccuracyMetrics` object

#### `assess_completeness(voxel_data: Any, signals: Optional[List[str]] = None, store_details: bool = True) -> CompletenessMetrics`

Assess data completeness.

**Parameters**:
- `voxel_data` (Any): Voxel domain data object
- `signals` (Optional[List[str]]): List of signal names to check (None = all signals)
- `store_details` (bool): Whether to store detailed information

**Returns**: `CompletenessMetrics` object

#### `fill_gaps(signal_array: np.ndarray, strategy: GapFillingStrategy = GapFillingStrategy.LINEAR) -> np.ndarray`

Fill missing data gaps.

**Parameters**:
- `signal_array` (np.ndarray): Signal array with missing data
- `strategy` (GapFillingStrategy): Gap filling strategy

**Returns**: Filled signal array

#### `comprehensive_assessment(voxel_data: Any, signals: Optional[List[str]] = None, layer_range: Optional[Tuple[int, int]] = None, coordinate_transformer: Optional[Any] = None, store_maps: bool = True) -> Dict[str, Any]`

Perform comprehensive quality assessment.

**Parameters**:
- `voxel_data` (Any): Voxel domain data object
- `signals` (Optional[List[str]]): List of signal names to check (None = all signals)
- `layer_range` (Optional[Tuple[int, int]]): (min_layer, max_layer) range for temporal coverage
- `coordinate_transformer` (Optional[Any]): Optional coordinate system transformer
- `store_maps` (bool): Whether to store per-voxel quality maps

**Returns**: Dictionary containing all quality assessment results with keys:
- `'data_quality'`: DataQualityMetrics
- `'signal_quality'`: Dict[str, SignalQualityMetrics]
- `'alignment_accuracy'`: AlignmentAccuracyMetrics
- `'completeness'`: CompletenessMetrics
- `'summary'`: Dictionary with overall scores

#### `generate_quality_report(assessment_results: Dict[str, Any], output_file: Optional[str] = None) -> str`

Generate a human-readable quality report.

**Parameters**:
- `assessment_results` (Dict[str, Any]): Results from `comprehensive_assessment()`
- `output_file` (Optional[str]): Optional file path to save report

**Returns**: Report string

---

## DataQualityAnalyzer

Analyzes overall data quality.

```python
from am_qadf.quality import DataQualityAnalyzer

analyzer = DataQualityAnalyzer()
```

### Methods

#### `calculate_completeness(voxel_data: Any, signals: Optional[List[str]] = None) -> float`

Calculate completeness: percentage of voxels with data.

**Parameters**:
- `voxel_data` (Any): Voxel domain data object
- `signals` (Optional[List[str]]): List of signal names to check (None = all signals)

**Returns**: Completeness ratio (0-1)

#### `calculate_spatial_coverage(voxel_data: Any, signals: Optional[List[str]] = None) -> float`

Calculate spatial coverage: ratio of spatial region covered by data.

**Parameters**:
- `voxel_data` (Any): Voxel domain data object
- `signals` (Optional[List[str]]): List of signal names to check (None = all signals)

**Returns**: Spatial coverage ratio (0-1)

#### `calculate_temporal_coverage(voxel_data: Any, layer_range: Optional[Tuple[int, int]] = None) -> float`

Calculate temporal coverage: ratio of temporal range covered by data.

**Parameters**:
- `voxel_data` (Any): Voxel domain data object
- `layer_range` (Optional[Tuple[int, int]]): (min_layer, max_layer) range to check

**Returns**: Temporal coverage ratio (0-1)

#### `calculate_consistency(voxel_data: Any, signals: Optional[List[str]] = None) -> float`

Calculate consistency: consistency across data sources.

**Parameters**:
- `voxel_data` (Any): Voxel domain data object
- `signals` (Optional[List[str]]): List of signal names to check (None = all signals)

**Returns**: Consistency score (0-1)

#### `identify_missing_regions(voxel_data: Any, signals: Optional[List[str]] = None, min_region_size: int = 10) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]`

Identify missing data regions.

**Parameters**:
- `voxel_data` (Any): Voxel domain data object
- `signals` (Optional[List[str]]): List of signal names to check (None = all signals)
- `min_region_size` (int): Minimum voxel count for a region to be reported

**Returns**: List of (bbox_min, bbox_max) tuples for missing regions

#### `assess_quality(voxel_data: Any, signals: Optional[List[str]] = None, layer_range: Optional[Tuple[int, int]] = None) -> DataQualityMetrics`

Assess overall data quality.

**Parameters**:
- `voxel_data` (Any): Voxel domain data object
- `signals` (Optional[List[str]]): List of signal names to check (None = all signals)
- `layer_range` (Optional[Tuple[int, int]]): (min_layer, max_layer) range for temporal coverage

**Returns**: `DataQualityMetrics` object

---

## SignalQualityAnalyzer

Analyzes signal quality metrics.

```python
from am_qadf.quality import SignalQualityAnalyzer

analyzer = SignalQualityAnalyzer(noise_floor: float = 1e-6)
```

### Methods

#### `calculate_snr(signal_array: np.ndarray, noise_estimate: Optional[np.ndarray] = None, store_map: bool = True) -> Tuple[float, float, float, float, Optional[np.ndarray]]`

Calculate Signal-to-Noise Ratio (SNR).

**Parameters**:
- `signal_array` (np.ndarray): Signal array
- `noise_estimate` (Optional[np.ndarray]): Optional noise estimate array (if None, estimates from signal)
- `store_map` (bool): Whether to store per-voxel SNR map

**Returns**: Tuple of (mean_snr, std_snr, min_snr, max_snr, snr_map)

#### `calculate_uncertainty(signal_array: np.ndarray, measurement_uncertainty: Optional[float] = None, interpolation_uncertainty: Optional[np.ndarray] = None, store_map: bool = True) -> Tuple[float, Optional[np.ndarray]]`

Calculate uncertainty in signal values.

**Parameters**:
- `signal_array` (np.ndarray): Signal array
- `measurement_uncertainty` (Optional[float]): Base measurement uncertainty (fraction or absolute)
- `interpolation_uncertainty` (Optional[np.ndarray]): Optional interpolation uncertainty map
- `store_map` (bool): Whether to store per-voxel uncertainty map

**Returns**: Tuple of (mean_uncertainty, uncertainty_map)

#### `calculate_confidence(signal_array: np.ndarray, snr_map: Optional[np.ndarray] = None, uncertainty_map: Optional[np.ndarray] = None, store_map: bool = True) -> Tuple[float, Optional[np.ndarray]]`

Calculate confidence scores for signal values.

**Parameters**:
- `signal_array` (np.ndarray): Signal array
- `snr_map` (Optional[np.ndarray]): Optional SNR map
- `uncertainty_map` (Optional[np.ndarray]): Optional uncertainty map
- `store_map` (bool): Whether to store per-voxel confidence map

**Returns**: Tuple of (mean_confidence, confidence_map)

#### `assess_signal_quality(signal_name: str, signal_array: np.ndarray, noise_estimate: Optional[np.ndarray] = None, measurement_uncertainty: Optional[float] = None, store_maps: bool = True) -> SignalQualityMetrics`

Assess quality for a single signal.

**Parameters**:
- `signal_name` (str): Name of the signal
- `signal_array` (np.ndarray): Signal array
- `noise_estimate` (Optional[np.ndarray]): Optional noise estimate
- `measurement_uncertainty` (Optional[float]): Optional measurement uncertainty
- `store_maps` (bool): Whether to store per-voxel quality maps

**Returns**: `SignalQualityMetrics` object

---

## AlignmentAccuracyAnalyzer

Analyzes alignment accuracy.

```python
from am_qadf.quality import AlignmentAccuracyAnalyzer

analyzer = AlignmentAccuracyAnalyzer(max_acceptable_error: float = 0.1)  # mm
```

### Methods

#### `validate_coordinate_alignment(source_points: np.ndarray, target_points: np.ndarray, transformation_matrix: Optional[np.ndarray] = None) -> float`

Validate coordinate system alignment accuracy.

**Parameters**:
- `source_points` (np.ndarray): Source coordinate points (N, 3)
- `target_points` (np.ndarray): Target coordinate points (N, 3)
- `transformation_matrix` (Optional[np.ndarray]): Optional transformation matrix (4x4)

**Returns**: Mean alignment error (mm)

#### `validate_temporal_alignment(source_times: np.ndarray, target_times: np.ndarray, tolerance: float = 0.1) -> float`

Validate temporal alignment accuracy.

**Parameters**:
- `source_times` (np.ndarray): Source timestamps or layer indices
- `target_times` (np.ndarray): Target timestamps or layer indices
- `tolerance` (float): Acceptable temporal difference

**Returns**: Mean temporal alignment error

#### `calculate_registration_residuals(reference_points: np.ndarray, aligned_points: np.ndarray) -> Tuple[float, float, np.ndarray]`

Calculate spatial registration residuals.

**Parameters**:
- `reference_points` (np.ndarray): Reference point cloud (N, 3)
- `aligned_points` (np.ndarray): Aligned point cloud (N, 3)

**Returns**: Tuple of (mean_residual, std_residual, residual_map)

#### `assess_alignment_accuracy(voxel_data: Any, coordinate_transformer: Optional[Any] = None, reference_data: Optional[Any] = None) -> AlignmentAccuracyMetrics`

Assess overall alignment accuracy.

**Parameters**:
- `voxel_data` (Any): Voxel domain data object
- `coordinate_transformer` (Optional[Any]): Optional coordinate system transformer
- `reference_data` (Optional[Any]): Optional reference data for comparison

**Returns**: `AlignmentAccuracyMetrics` object

---

## CompletenessAnalyzer

Analyzes data completeness and coverage.

```python
from am_qadf.quality import CompletenessAnalyzer

analyzer = CompletenessAnalyzer()
```

### Methods

#### `detect_missing_data(signal_array: np.ndarray, store_indices: bool = True) -> Tuple[int, Optional[np.ndarray]]`

Detect missing data in signal array.

**Parameters**:
- `signal_array` (np.ndarray): Signal array
- `store_indices` (bool): Whether to store indices of missing voxels

**Returns**: Tuple of (missing_count, missing_indices)

#### `analyze_coverage(voxel_data: Any, signals: Optional[List[str]] = None) -> Tuple[float, float]`

Analyze spatial and temporal coverage.

**Parameters**:
- `voxel_data` (Any): Voxel domain data object
- `signals` (Optional[List[str]]): List of signal names to check (None = all signals)

**Returns**: Tuple of (spatial_coverage, temporal_coverage)

#### `identify_missing_regions(signal_array: np.ndarray, bbox_min: Tuple[float, float, float], resolution: float, min_region_size: int = 10) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]`

Identify missing data regions.

**Parameters**:
- `signal_array` (np.ndarray): Signal array
- `bbox_min` (Tuple[float, float, float]): Minimum bounding box coordinates
- `resolution` (float): Voxel resolution
- `min_region_size` (int): Minimum voxel count for a region

**Returns**: List of (bbox_min, bbox_max) tuples for missing regions

#### `fill_gaps(signal_array: np.ndarray, strategy: GapFillingStrategy = GapFillingStrategy.LINEAR) -> np.ndarray`

Fill missing data gaps using specified strategy.

**Parameters**:
- `signal_array` (np.ndarray): Signal array with missing data
- `strategy` (GapFillingStrategy): Gap filling strategy

**Returns**: Filled signal array

#### `assess_completeness(voxel_data: Any, signals: Optional[List[str]] = None, store_details: bool = True) -> CompletenessMetrics`

Assess overall completeness.

**Parameters**:
- `voxel_data` (Any): Voxel domain data object
- `signals` (Optional[List[str]]): List of signal names to check (None = all signals)
- `store_details` (bool): Whether to store detailed information

**Returns**: `CompletenessMetrics` object

---

## Quality Metrics

### DataQualityMetrics

```python
from am_qadf.quality import DataQualityMetrics

@dataclass
class DataQualityMetrics:
    completeness: float  # Percentage of voxels with data (0-1)
    coverage_spatial: float  # Spatial coverage ratio (0-1)
    coverage_temporal: float  # Temporal coverage ratio (0-1)
    consistency_score: float  # Consistency across sources (0-1)
    accuracy_score: float  # Overall accuracy score (0-1)
    reliability_score: float  # Overall reliability score (0-1)
    filled_voxels: int  # Number of voxels with data
    total_voxels: int  # Total number of voxels
    sources_count: int  # Number of data sources contributing
    missing_regions: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]  # Missing region bounding boxes
```

#### Methods

##### `to_dict() -> Dict[str, Any]`

Convert metrics to dictionary.

**Returns**: Dictionary representation

---

### SignalQualityMetrics

```python
from am_qadf.quality import SignalQualityMetrics

@dataclass
class SignalQualityMetrics:
    signal_name: str
    snr_mean: float  # Mean SNR across all voxels
    snr_std: float  # Standard deviation of SNR
    snr_min: float  # Minimum SNR
    snr_max: float  # Maximum SNR
    uncertainty_mean: float  # Mean uncertainty
    confidence_mean: float  # Mean confidence score (0-1)
    quality_score: float  # Overall quality score (0-1)
    snr_map: Optional[np.ndarray] = None  # Per-voxel SNR map
    uncertainty_map: Optional[np.ndarray] = None  # Per-voxel uncertainty map
    confidence_map: Optional[np.ndarray] = None  # Per-voxel confidence map
```

#### Methods

##### `to_dict() -> Dict[str, Any]`

Convert metrics to dictionary.

**Returns**: Dictionary representation

---

### AlignmentAccuracyMetrics

```python
from am_qadf.quality import AlignmentAccuracyMetrics

@dataclass
class AlignmentAccuracyMetrics:
    coordinate_alignment_error: float  # Mean coordinate transformation error (mm)
    temporal_alignment_error: float  # Mean temporal alignment error (layers or seconds)
    spatial_registration_error: float  # Mean spatial registration error (mm)
    residual_error_mean: float  # Mean residual error after alignment
    residual_error_std: float  # Standard deviation of residual errors
    alignment_score: float  # Overall alignment score (0-1, higher is better)
    transformation_errors: List[float]  # Per-transformation errors
    registration_residuals: Optional[np.ndarray] = None  # Residual map
```

#### Methods

##### `to_dict() -> Dict[str, Any]`

Convert metrics to dictionary.

**Returns**: Dictionary representation

---

### CompletenessMetrics

```python
from am_qadf.quality import CompletenessMetrics

@dataclass
class CompletenessMetrics:
    completeness_ratio: float  # Overall completeness (0-1)
    spatial_coverage: float  # Spatial coverage ratio (0-1)
    temporal_coverage: float  # Temporal coverage ratio (0-1)
    missing_voxels_count: int  # Number of missing voxels
    missing_regions_count: int  # Number of missing regions
    gap_fillable_ratio: float  # Ratio of gaps that can be filled (0-1)
    missing_voxel_indices: Optional[np.ndarray] = None  # Indices of missing voxels
    missing_regions: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None  # Missing region bboxes
```

#### Methods

##### `to_dict() -> Dict[str, Any]`

Convert metrics to dictionary.

**Returns**: Dictionary representation

---

## GapFillingStrategy

Enumeration of gap filling strategies.

```python
from am_qadf.quality import GapFillingStrategy

# Available strategies:
GapFillingStrategy.NONE     # Don't fill gaps
GapFillingStrategy.ZERO     # Fill with zeros
GapFillingStrategy.NEAREST  # Nearest neighbor interpolation
GapFillingStrategy.LINEAR   # Linear interpolation
GapFillingStrategy.MEAN     # Fill with mean value
GapFillingStrategy.MEDIAN   # Fill with median value
```

---

## Related

- [Quality Module Documentation](../05-modules/quality.md) - Module overview
- [All API References](README.md) - Other API references

---

**Parent**: [API Reference](README.md)
