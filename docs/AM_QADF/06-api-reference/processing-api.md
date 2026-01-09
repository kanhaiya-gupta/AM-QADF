# Processing Module API Reference

## Overview

The Processing module provides signal processing and noise reduction capabilities.

## Noise Reduction

### OutlierDetector

Detects and removes outliers from signal data.

```python
from am_qadf.processing import OutlierDetector

detector = OutlierDetector(
    method: str = 'zscore',
    threshold: float = 3.0,
    use_spatial: bool = True
)
```

#### Methods

##### `detect(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]`

Detect outliers using specified method.

**Parameters**:
- `signal` (np.ndarray): Signal array

**Returns**: Tuple of (outlier_mask, scores) where outlier_mask is boolean array

##### `detect_zscore(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]`

Detect outliers using Z-score method.

**Parameters**:
- `signal` (np.ndarray): Signal array

**Returns**: Tuple of (outlier_mask, z_scores)

##### `detect_iqr(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]`

Detect outliers using Interquartile Range (IQR) method.

**Parameters**:
- `signal` (np.ndarray): Signal array

**Returns**: Tuple of (outlier_mask, iqr_scores)

##### `detect_spatial(signal: np.ndarray, kernel_size: int = 3) -> Tuple[np.ndarray, np.ndarray]`

Detect outliers using spatial context.

**Parameters**:
- `signal` (np.ndarray): Signal array
- `kernel_size` (int): Size of neighborhood kernel

**Returns**: Tuple of (outlier_mask, spatial_scores)

##### `remove_outliers(signal: np.ndarray, fill_method: str = 'median') -> np.ndarray`

Remove outliers and fill with replacement values.

**Parameters**:
- `signal` (np.ndarray): Signal array
- `fill_method` (str): Method to fill outliers ('median', 'mean', 'interpolate', 'zero')

**Returns**: Cleaned signal array

---

### SignalSmoother

Smooths signals using various filtering techniques.

```python
from am_qadf.processing import SignalSmoother

smoother = SignalSmoother(
    method: str = 'gaussian',
    kernel_size: float = 1.0
)
```

#### Methods

##### `smooth(signal: np.ndarray) -> np.ndarray`

Apply smoothing using specified method.

**Parameters**:
- `signal` (np.ndarray): Signal array

**Returns**: Smoothed signal array

##### `gaussian_smooth(signal: np.ndarray, sigma: Optional[float] = None) -> np.ndarray`

Apply Gaussian smoothing.

**Parameters**:
- `signal` (np.ndarray): Signal array
- `sigma` (Optional[float]): Standard deviation of Gaussian kernel (default: kernel_size)

**Returns**: Smoothed signal array

##### `median_smooth(signal: np.ndarray, size: Optional[int] = None) -> np.ndarray`

Apply median filtering.

**Parameters**:
- `signal` (np.ndarray): Signal array
- `size` (Optional[int]): Size of median filter kernel (default: kernel_size)

**Returns**: Smoothed signal array

##### `savgol_smooth(signal: np.ndarray, window_length: Optional[int] = None, poly_order: int = 3) -> np.ndarray`

Apply Savitzky-Golay filtering (1D only, applied per axis).

**Parameters**:
- `signal` (np.ndarray): Signal array
- `window_length` (Optional[int]): Window length (default: kernel_size * 2 + 1)
- `poly_order` (int): Polynomial order

**Returns**: Smoothed signal array

---

### SignalQualityMetrics

Compute quality metrics for signals.

```python
from am_qadf.processing import SignalQualityMetrics

# This is a static class - use static methods directly
```

#### Static Methods

##### `compute_snr(signal: np.ndarray, noise_estimate: Optional[np.ndarray] = None) -> float`

Compute Signal-to-Noise Ratio (SNR).

**Parameters**:
- `signal` (np.ndarray): Signal array
- `noise_estimate` (Optional[np.ndarray]): Optional noise estimate (if None, uses std of signal)

**Returns**: SNR in dB

##### `compute_coverage(signal: np.ndarray, threshold: float = 0.0) -> float`

Compute signal coverage (fraction of non-zero/non-nan voxels).

**Parameters**:
- `signal` (np.ndarray): Signal array
- `threshold` (float): Threshold for valid signal

**Returns**: Coverage fraction (0.0 to 1.0)

##### `compute_uniformity(signal: np.ndarray) -> float`

Compute signal uniformity (coefficient of variation).

**Parameters**:
- `signal` (np.ndarray): Signal array

**Returns**: Coefficient of variation

##### `compute_statistics(signal: np.ndarray) -> Dict[str, float]`

Compute comprehensive signal statistics.

**Parameters**:
- `signal` (np.ndarray): Signal array

**Returns**: Dictionary with 'mean', 'std', 'min', 'max', 'median', 'q25', 'q75', 'snr', 'coverage', 'uniformity'

---

### NoiseReductionPipeline

Complete noise reduction pipeline.

```python
from am_qadf.processing import NoiseReductionPipeline

pipeline = NoiseReductionPipeline(
    outlier_method: str = 'zscore',
    outlier_threshold: float = 3.0,
    smoothing_method: str = 'gaussian',
    smoothing_kernel: float = 1.0,
    use_spatial: bool = True
)
```

#### Methods

##### `process(signal: np.ndarray, remove_outliers: bool = True, apply_smoothing: bool = True, compute_metrics: bool = True) -> Dict[str, Any]`

Process signal through noise reduction pipeline.

**Parameters**:
- `signal` (np.ndarray): Input signal array
- `remove_outliers` (bool): Whether to remove outliers
- `apply_smoothing` (bool): Whether to apply smoothing
- `compute_metrics` (bool): Whether to compute quality metrics

**Returns**: Dictionary with:
- `'original'`: Original signal array
- `'cleaned'`: Cleaned signal array
- `'outlier_mask'`: Outlier detection mask
- `'outlier_scores'`: Outlier scores
- `'metrics'`: Quality metrics (if compute_metrics=True) with 'original' and 'cleaned' statistics

---

## Signal Generation

### ThermalFieldGenerator

Generate thermal fields from energy density using heat diffusion models.

```python
from am_qadf.processing import ThermalFieldGenerator

generator = ThermalFieldGenerator(
    thermal_diffusivity: float = 8.0e-6,  # m²/s
    heat_capacity: float = 500.0,  # J/(kg·K)
    density: float = 8000.0,  # kg/m³
    ambient_temp: float = 25.0,  # °C
    cooling_rate: float = 0.1  # 1/s
)
```

#### Methods

##### `energy_to_temperature(energy_density: np.ndarray, time_step: float = 0.001, voxel_size: float = 0.001) -> np.ndarray`

Convert energy density to temperature using simple heat model.

**Parameters**:
- `energy_density` (np.ndarray): Energy density array (J/m³)
- `time_step` (float): Time step for heat diffusion (seconds)
- `voxel_size` (float): Size of each voxel (meters)

**Returns**: Temperature array (°C)

##### `apply_heat_diffusion(temperature_field: np.ndarray, time_step: float = 0.001, voxel_size: float = 0.001, num_iterations: int = 1) -> np.ndarray`

Apply heat diffusion to temperature field.

**Parameters**:
- `temperature_field` (np.ndarray): Initial temperature field (°C)
- `time_step` (float): Time step for diffusion (seconds)
- `voxel_size` (float): Size of each voxel (meters)
- `num_iterations` (int): Number of diffusion iterations

**Returns**: Diffused temperature field (°C)

##### `apply_cooling(temperature_field: np.ndarray, time_step: float = 0.001) -> np.ndarray`

Apply cooling to temperature field (exponential decay to ambient).

**Parameters**:
- `temperature_field` (np.ndarray): Temperature field (°C)
- `time_step` (float): Time step (seconds)

**Returns**: Cooled temperature field (°C)

##### `generate_thermal_field(energy_density: np.ndarray, voxel_size: float = 0.001, apply_diffusion: bool = True, apply_cooling: bool = True, time_steps: int = 1) -> np.ndarray`

Generate complete thermal field from energy density.

**Parameters**:
- `energy_density` (np.ndarray): Energy density array (J/m³)
- `voxel_size` (float): Size of each voxel (meters)
- `apply_diffusion` (bool): Whether to apply heat diffusion
- `apply_cooling` (bool): Whether to apply cooling
- `time_steps` (int): Number of time steps to simulate

**Returns**: Thermal field (temperature in °C)

---

### DensityFieldEstimator

Estimate density field from process parameters.

```python
from am_qadf.processing import DensityFieldEstimator

estimator = DensityFieldEstimator(
    base_density: float = 8000.0,  # kg/m³ (fully dense)
    min_density: float = 6000.0,  # kg/m³ (porous)
    optimal_energy: float = 50.0  # J/mm³
)
```

#### Methods

##### `estimate_from_energy(energy_density: np.ndarray) -> np.ndarray`

Estimate density from energy density.

**Parameters**:
- `energy_density` (np.ndarray): Energy density array (J/mm³ or J/m³)

**Returns**: Estimated density array (kg/m³)

##### `estimate_from_thermal_history(temperature_field: np.ndarray, melting_temp: float = 1500.0) -> np.ndarray`

Estimate density from thermal history (temperature field).

**Parameters**:
- `temperature_field` (np.ndarray): Temperature field (°C)
- `melting_temp` (float): Melting temperature (°C)

**Returns**: Estimated density array (kg/m³)

---

### StressFieldGenerator

Generate stress fields from thermal gradients.

```python
from am_qadf.processing import StressFieldGenerator

generator = StressFieldGenerator(
    thermal_expansion_coeff: float = 12.0e-6,  # 1/K
    youngs_modulus: float = 200.0e9,  # Pa
    poissons_ratio: float = 0.3
)
```

#### Methods

##### `compute_thermal_strain(temperature_field: np.ndarray, reference_temp: float = 25.0) -> np.ndarray`

Compute thermal strain from temperature field.

**Parameters**:
- `temperature_field` (np.ndarray): Temperature field (°C)
- `reference_temp` (float): Reference temperature (°C)

**Returns**: Thermal strain (dimensionless)

##### `compute_thermal_stress(temperature_field: np.ndarray, reference_temp: float = 25.0) -> np.ndarray`

Compute thermal stress from temperature field.

**Parameters**:
- `temperature_field` (np.ndarray): Temperature field (°C)
- `reference_temp` (float): Reference temperature (°C)

**Returns**: Thermal stress (Pa)

##### `compute_stress_from_gradient(temperature_field: np.ndarray, voxel_size: float = 0.001) -> Dict[str, np.ndarray]`

Compute stress components from temperature gradients.

**Parameters**:
- `temperature_field` (np.ndarray): Temperature field (°C)
- `voxel_size` (float): Size of each voxel (meters)

**Returns**: Dictionary with stress components:
- `'von_mises'`: Von Mises equivalent stress (Pa)
- `'max_principal'`: Maximum principal stress (Pa)
- `'min_principal'`: Minimum principal stress (Pa)

---

## Related

- [Processing Module Documentation](../05-modules/processing.md) - Module overview
- [All API References](README.md) - Other API references

---

**Parent**: [API Reference](README.md)

