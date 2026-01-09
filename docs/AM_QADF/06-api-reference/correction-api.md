# Correction Module API Reference

## Overview

The Correction module provides geometric distortion correction and calibration capabilities.

## Distortion Models

### DistortionModel

Abstract base class for geometric distortion models.

```python
from am_qadf.correction import DistortionModel

# This is an abstract class - use subclasses instead
```

#### Abstract Methods

- `apply(points: np.ndarray) -> np.ndarray`: Apply distortion to points
- `correct(points: np.ndarray) -> np.ndarray`: Correct distortion from points
- `get_parameters() -> Dict[str, Any]`: Get distortion parameters

---

### ScalingModel

Scaling distortion correction.

```python
from am_qadf.correction import ScalingModel

model = ScalingModel(
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    scale_z: float = 1.0,
    center: Optional[Tuple[float, float, float]] = None
)
```

#### Methods

##### `apply(points: np.ndarray) -> np.ndarray`

Apply scaling distortion.

**Parameters**:
- `points` (np.ndarray): Array of points (N, 3)

**Returns**: Distorted points (N, 3)

##### `correct(points: np.ndarray) -> np.ndarray`

Apply scaling correction.

**Parameters**:
- `points` (np.ndarray): Array of points (N, 3)

**Returns**: Corrected points (N, 3)

##### `get_parameters() -> Dict[str, Any]`

Get scaling parameters.

**Returns**: Dictionary with 'type', 'scale_x', 'scale_y', 'scale_z', 'center'

### RotationModel

Rotation distortion correction.

```python
from am_qadf.correction import RotationModel

model = RotationModel(
    axis: str = 'z',
    angle: float = 0.0,  # radians
    center: Optional[Tuple[float, float, float]] = None
)
```

#### Methods

##### `apply(points: np.ndarray) -> np.ndarray`

Apply rotation distortion.

**Parameters**:
- `points` (np.ndarray): Array of points (N, 3)

**Returns**: Distorted points (N, 3)

##### `correct(points: np.ndarray) -> np.ndarray`

Apply rotation correction.

**Parameters**:
- `points` (np.ndarray): Array of points (N, 3)

**Returns**: Corrected points (N, 3)

##### `get_parameters() -> Dict[str, Any]`

Get rotation parameters.

**Returns**: Dictionary with 'type', 'axis', 'angle', 'angle_degrees', 'center'

### WarpingModel

Warping distortion correction.

```python
from am_qadf.correction import WarpingModel

model = WarpingModel(
    displacement_field: Optional[np.ndarray] = None,
    reference_points: Optional[np.ndarray] = None,
    displacement_vectors: Optional[np.ndarray] = None
)
```

#### Methods

##### `apply(points: np.ndarray) -> np.ndarray`

Apply warping distortion.

**Parameters**:
- `points` (np.ndarray): Array of points (N, 3)

**Returns**: Distorted points (N, 3)

##### `correct(points: np.ndarray) -> np.ndarray`

Apply warping correction.

**Parameters**:
- `points` (np.ndarray): Array of points (N, 3)

**Returns**: Corrected points (N, 3)

##### `get_parameters() -> Dict[str, Any]`

Get warping parameters.

**Returns**: Dictionary with 'type', 'has_field', 'num_reference_points'

##### `estimate_from_correspondences(source_points: np.ndarray, target_points: np.ndarray) -> None`

Estimate warping model from point correspondences.

**Parameters**:
- `source_points` (np.ndarray): Source (distorted) points (N, 3)
- `target_points` (np.ndarray): Target (correct) points (N, 3)

### CombinedDistortionModel

Combined distortion correction.

```python
from am_qadf.correction import CombinedDistortionModel, DistortionModel

model = CombinedDistortionModel(
    models: List[DistortionModel]
)
```

#### Methods

##### `apply(points: np.ndarray) -> np.ndarray`

Apply all distortions in sequence.

**Parameters**:
- `points` (np.ndarray): Array of points (N, 3)

**Returns**: Distorted points (N, 3)

##### `correct(points: np.ndarray) -> np.ndarray`

Apply all corrections in reverse order.

**Parameters**:
- `points` (np.ndarray): Array of points (N, 3)

**Returns**: Corrected points (N, 3)

##### `get_parameters() -> Dict[str, Any]`

Get parameters from all models.

**Returns**: Dictionary with 'type', 'num_models', 'models'

---

## Calibration

### ReferenceMeasurement

Represents a reference measurement for calibration.

```python
from am_qadf.correction import ReferenceMeasurement
from datetime import datetime

measurement = ReferenceMeasurement(
    point: Tuple[float, float, float],
    expected_point: Tuple[float, float, float],
    timestamp: Optional[datetime] = None,
    measurement_type: str = "manual",
    uncertainty: float = 0.0,
    metadata: Dict[str, Any] = {}
)
```

### CalibrationData

Calibration data for a coordinate system or sensor.

```python
from am_qadf.correction import CalibrationData
from datetime import datetime

calibration = CalibrationData(
    name: str,
    calibration_date: datetime,
    reference_measurements: List[ReferenceMeasurement] = [],
    transformation_matrix: Optional[np.ndarray] = None,
    distortion_parameters: Dict[str, Any] = {},
    uncertainty: float = 0.0,
    metadata: Dict[str, Any] = {}
)
```

#### Methods

##### `add_measurement(measurement: ReferenceMeasurement) -> None`

Add a reference measurement.

**Parameters**:
- `measurement` (ReferenceMeasurement): Reference measurement

##### `compute_error() -> Dict[str, float]`

Compute calibration error from reference measurements.

**Returns**: Dictionary with 'mean_error', 'max_error', 'rms_error', 'std_error', 'num_measurements'

---

## CalibrationManager

Manages calibration data and corrections.

```python
from am_qadf.correction import CalibrationManager

manager = CalibrationManager()
```

### Methods

#### `register_calibration(name: str, calibration: CalibrationData) -> None`

Register calibration data.

**Parameters**:
- `name` (str): Calibration name/identifier
- `calibration` (CalibrationData): CalibrationData object

#### `get_calibration(name: str) -> Optional[CalibrationData]`

Get calibration data.

**Parameters**:
- `name` (str): Calibration name

**Returns**: CalibrationData or None

#### `list_calibrations() -> List[str]`

List all registered calibrations.

**Returns**: List of calibration names

#### `estimate_transformation(calibration_name: str) -> Optional[np.ndarray]`

Estimate transformation matrix from calibration measurements.

**Parameters**:
- `calibration_name` (str): Name of calibration

**Returns**: 4x4 transformation matrix or None

#### `validate_calibration(calibration_name: str, threshold: float = 0.1) -> Dict[str, Any]`

Validate calibration quality.

**Parameters**:
- `calibration_name` (str): Name of calibration
- `threshold` (float): Maximum acceptable error (mm)

**Returns**: Dictionary with 'valid', 'error_metrics', 'threshold', 'within_threshold'

#### `apply_calibration_correction(points: np.ndarray, calibration_name: str) -> np.ndarray`

Apply calibration correction to points.

**Parameters**:
- `points` (np.ndarray): Points to correct (N, 3)
- `calibration_name` (str): Name of calibration to use

**Returns**: Corrected points (N, 3)

---

## Validation

### AlignmentQuality

Quality levels for alignment.

```python
from am_qadf.correction import AlignmentQuality

# Available quality levels:
AlignmentQuality.EXCELLENT   # < 0.05 mm error
AlignmentQuality.GOOD        # < 0.1 mm error
AlignmentQuality.ACCEPTABLE   # < 0.2 mm error
AlignmentQuality.POOR         # >= 0.2 mm error
```

### ValidationMetrics

Validation metrics for correction quality.

```python
from am_qadf.correction import ValidationMetrics

metrics = ValidationMetrics(
    mean_error: float,
    max_error: float,
    rms_error: float,
    std_error: float,
    median_error: float,
    num_points: int,
    quality: AlignmentQuality
)
```

#### Methods

##### `to_dict() -> Dict[str, Any]`

Convert to dictionary.

**Returns**: Dictionary representation

---

## CorrectionValidator

Validates correction accuracy.

```python
from am_qadf.correction import CorrectionValidator

validator = CorrectionValidator(
    excellent_threshold: float = 0.05,  # mm
    good_threshold: float = 0.1,  # mm
    acceptable_threshold: float = 0.2  # mm
)
```

### Methods

#### `compute_alignment_error(corrected_points: np.ndarray, reference_points: np.ndarray) -> np.ndarray`

Compute alignment error between corrected and reference points.

**Parameters**:
- `corrected_points` (np.ndarray): Corrected points (N, 3)
- `reference_points` (np.ndarray): Reference (ground truth) points (N, 3)

**Returns**: Array of errors (N,)

#### `assess_quality(mean_error: float) -> AlignmentQuality`

Assess alignment quality based on mean error.

**Parameters**:
- `mean_error` (float): Mean alignment error (mm)

**Returns**: AlignmentQuality enum

#### `validate_correction(corrected_points: np.ndarray, reference_points: np.ndarray) -> ValidationMetrics`

Validate correction quality.

**Parameters**:
- `corrected_points` (np.ndarray): Corrected points (N, 3)
- `reference_points` (np.ndarray): Reference (ground truth) points (N, 3)

**Returns**: ValidationMetrics object

#### `compare_corrections(corrections: Dict[str, Tuple[np.ndarray, np.ndarray]], reference_points: np.ndarray) -> Dict[str, ValidationMetrics]`

Compare multiple correction methods.

**Parameters**:
- `corrections` (Dict[str, Tuple[np.ndarray, np.ndarray]]): Dictionary mapping method names to (corrected_points, description)
- `reference_points` (np.ndarray): Reference (ground truth) points (N, 3)

**Returns**: Dictionary mapping method names to ValidationMetrics

#### `generate_validation_report(metrics: ValidationMetrics, include_details: bool = True) -> str`

Generate human-readable validation report.

**Parameters**:
- `metrics` (ValidationMetrics): ValidationMetrics object
- `include_details` (bool): Whether to include detailed statistics

**Returns**: Formatted report string

#### `validate_distortion_correction(original_points: np.ndarray, corrected_points: np.ndarray, reference_points: np.ndarray) -> Dict[str, Any]`

Validate distortion correction improvement.

**Parameters**:
- `original_points` (np.ndarray): Original (distorted) points (N, 3)
- `corrected_points` (np.ndarray): Corrected points (N, 3)
- `reference_points` (np.ndarray): Reference (ground truth) points (N, 3)

**Returns**: Dictionary with 'before', 'after', and 'improvement' metrics

---

## Related

- [Correction Module Documentation](../05-modules/correction.md) - Module overview
- [All API References](README.md) - Other API references

---

**Parent**: [API Reference](README.md)

