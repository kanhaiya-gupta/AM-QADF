# Analytics Module API Reference

## Overview

The Analytics module provides statistical analysis, sensitivity analysis, and process analysis capabilities.

## AnalyticsClient

Main client for analytics operations.

```python
from am_qadf.analytics import AnalyticsClient

client = AnalyticsClient(mongo_client: Optional[MongoDBClient] = None)
```

### Methods

#### `get_statistical_analysis_client() -> StatisticalAnalysisClient`

Get statistical analysis client.

**Returns**: `StatisticalAnalysisClient` instance

#### `get_sensitivity_analysis_client() -> SensitivityAnalysisClient`

Get sensitivity analysis client.

**Returns**: `SensitivityAnalysisClient` instance

#### `get_quality_assessment_client() -> QualityAssessmentClient`

Get quality assessment client.

**Returns**: `QualityAssessmentClient` instance

#### `get_process_analysis_client() -> ProcessAnalysisClient`

Get process analysis client.

**Returns**: `ProcessAnalysisClient` instance

#### `get_virtual_experiments_client() -> VirtualExperimentsClient`

Get virtual experiments client.

**Returns**: `VirtualExperimentsClient` instance

---

## StatisticalAnalysisClient

Statistical analysis operations.

```python
from am_qadf.analytics import StatisticalAnalysisClient

client = StatisticalAnalysisClient(mongo_client: Optional[MongoDBClient] = None)
```

### Methods

#### `compute_descriptive_statistics(signal_array: np.ndarray) -> Dict[str, float]`

Compute descriptive statistics.

**Parameters**:
- `signal_array` (np.ndarray): Signal array

**Returns**: Dictionary of statistics (mean, std, min, max, median, etc.)

#### `compute_correlation(signal1: np.ndarray, signal2: np.ndarray) -> float`

Compute correlation coefficient.

**Parameters**:
- `signal1` (np.ndarray): First signal array
- `signal2` (np.ndarray): Second signal array

**Returns**: Correlation coefficient (-1 to 1)

#### `compute_correlation_matrix(signals: Dict[str, np.ndarray]) -> np.ndarray`

Compute correlation matrix for multiple signals.

**Parameters**:
- `signals` (Dict[str, np.ndarray]): Dictionary of signal arrays

**Returns**: Correlation matrix

#### `perform_regression(x: np.ndarray, y: np.ndarray, method: str = 'linear') -> Dict[str, Any]`

Perform regression analysis.

**Parameters**:
- `x` (np.ndarray): Independent variable
- `y` (np.ndarray): Dependent variable
- `method` (str): Regression method ('linear', 'polynomial', 'nonlinear')

**Returns**: Dictionary with regression results

---

## SensitivityAnalysisClient

Main client for sensitivity analysis operations with warehouse data integration.

```python
from am_qadf.analytics.sensitivity_analysis import SensitivityAnalysisClient, SensitivityAnalysisConfig

client = SensitivityAnalysisClient(
    unified_query_client: UnifiedQueryClient,
    voxel_domain_client: Optional[VoxelDomainClient] = None
)
```

### Attributes

- `unified_client` (UnifiedQueryClient): Unified query client for warehouse data
- `voxel_client` (Optional[VoxelDomainClient]): Optional voxel domain client
- `global_analyzer` (Optional[GlobalSensitivityAnalyzer]): Global sensitivity analyzer
- `local_analyzer` (Optional[LocalSensitivityAnalyzer]): Local sensitivity analyzer
- `doe_designer` (Optional[ExperimentalDesigner]): Experimental designer
- `uncertainty_quantifier` (Optional[UncertaintyQuantifier]): Uncertainty quantifier

### Methods

#### `query_process_variables(model_id: str, variables: List[str], spatial_region: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None, layer_range: Optional[Tuple[int, int]] = None, time_range: Optional[Tuple[datetime, datetime]] = None) -> pd.DataFrame`

Query process variables from warehouse.

**Parameters**:
- `model_id` (str): Model ID
- `variables` (List[str]): List of variable names (e.g., ['laser_power', 'scan_speed'])
- `spatial_region` (Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]): Optional bounding box
- `layer_range` (Optional[Tuple[int, int]]): Optional layer range
- `time_range` (Optional[Tuple[datetime, datetime]]): Optional time range

**Returns**: DataFrame with process variables

#### `query_measurement_data(model_id: str, sources: List[str] = ['ispm', 'ct'], spatial_region: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None, layer_range: Optional[Tuple[int, int]] = None, time_range: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, pd.DataFrame]`

Query measurement data from warehouse.

**Parameters**:
- `model_id` (str): Model ID
- `sources` (List[str]): List of sources (e.g., ['ispm', 'ct'])
- `spatial_region` (Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]): Optional bounding box
- `layer_range` (Optional[Tuple[int, int]]): Optional layer range
- `time_range` (Optional[Tuple[datetime, datetime]]): Optional time range

**Returns**: Dictionary mapping source names to DataFrames

#### `perform_sensitivity_analysis(model_id: str, config: SensitivityAnalysisConfig) -> Dict[str, Any]`

Perform sensitivity analysis using warehouse data.

**Parameters**:
- `model_id` (str): Model ID
- `config` (SensitivityAnalysisConfig): Sensitivity analysis configuration

**Returns**: Dictionary with sensitivity analysis results containing:
- `model_id` (str): Model ID
- `method` (str): Analysis method used
- `result` (SensitivityResult): Analysis result object
- `parameter_bounds` (Dict[str, Tuple[float, float]]): Parameter bounds
- `sample_size` (int): Number of samples used
- `timestamp` (datetime): Analysis timestamp

---

## SensitivityAnalysisConfig

Configuration dataclass for sensitivity analysis.

```python
from am_qadf.analytics.sensitivity_analysis import SensitivityAnalysisConfig

config = SensitivityAnalysisConfig(
    method: str = "sobol",  # "sobol", "morris", "local", "doe"
    process_variables: List[str] = None,
    measurement_variables: List[str] = None,
    spatial_region: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None,
    layer_range: Optional[Tuple[int, int]] = None,
    time_range: Optional[Tuple[datetime, datetime]] = None,
    sample_size: int = 1000,
    confidence_level: float = 0.95,
    use_voxel_domain: bool = False,
    voxel_resolution: float = 0.5
)
```

### Attributes

- `method` (str): Analysis method ("sobol", "morris", "local", "doe")
- `process_variables` (List[str]): Variables to analyze (e.g., ['laser_power', 'scan_speed'])
- `measurement_variables` (List[str]): Measurement outputs (e.g., ['density', 'temperature'])
- `spatial_region` (Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]): Bounding box
- `layer_range` (Optional[Tuple[int, int]]): Layer range
- `time_range` (Optional[Tuple[datetime, datetime]]): Time range
- `sample_size` (int): Number of samples
- `confidence_level` (float): Confidence level (0-1)
- `use_voxel_domain` (bool): Whether to use voxel domain
- `voxel_resolution` (float): Voxel resolution

---

## GlobalSensitivityAnalyzer

Global sensitivity analyzer for process parameters.

```python
from am_qadf.analytics.sensitivity_analysis import GlobalSensitivityAnalyzer, SensitivityConfig

analyzer = GlobalSensitivityAnalyzer(config: Optional[SensitivityConfig] = None)
```

### Methods

#### `analyze_sobol(model_function: Callable, parameter_bounds: Dict[str, Tuple[float, float]], parameter_names: List[str] = None) -> SensitivityResult`

Perform Sobol sensitivity analysis.

**Parameters**:
- `model_function` (Callable): Function that takes parameter array and returns output
- `parameter_bounds` (Dict[str, Tuple[float, float]]): Dictionary of parameter bounds {name: (min, max)}
- `parameter_names` (List[str]): List of parameter names (optional)

**Returns**: `SensitivityResult` with:
- `success` (bool): Whether analysis succeeded
- `method` (str): Analysis method ("Sobol")
- `parameter_names` (List[str]): Parameter names
- `sensitivity_indices` (Dict[str, float]): Sensitivity indices (S1, ST, S2)
- `confidence_intervals` (Dict[str, Tuple[float, float]]): Confidence intervals
- `analysis_time` (float): Analysis time in seconds
- `sample_size` (int): Number of samples used

#### `analyze_morris(model_function: Callable, parameter_bounds: Dict[str, Tuple[float, float]], parameter_names: List[str] = None) -> SensitivityResult`

Perform Morris screening analysis.

**Parameters**:
- `model_function` (Callable): Function that takes parameter array and returns output
- `parameter_bounds` (Dict[str, Tuple[float, float]]): Dictionary of parameter bounds {name: (min, max)}
- `parameter_names` (List[str]): List of parameter names (optional)

**Returns**: `SensitivityResult` with:
- `success` (bool): Whether analysis succeeded
- `method` (str): Analysis method ("Morris")
- `parameter_names` (List[str]): Parameter names
- `sensitivity_indices` (Dict[str, float]): Sensitivity indices (mu, sigma, mu_star)
- `confidence_intervals` (Dict[str, Tuple[float, float]]): Confidence intervals
- `analysis_time` (float): Analysis time in seconds
- `sample_size` (int): Number of samples used

#### `analyze_variance_based(model_function: Callable, parameter_bounds: Dict[str, Tuple[float, float]], parameter_names: List[str] = None) -> SensitivityResult`

Perform variance-based sensitivity analysis using random sampling.

**Parameters**:
- `model_function` (Callable): Function that takes parameter array and returns output
- `parameter_bounds` (Dict[str, Tuple[float, float]]): Dictionary of parameter bounds {name: (min, max)}
- `parameter_names` (List[str]): List of parameter names (optional)

**Returns**: `SensitivityResult` with variance-based sensitivity indices

#### `get_cached_result(method: str, parameter_names: List[str]) -> Optional[SensitivityResult]`

Get cached analysis result.

**Parameters**:
- `method` (str): Analysis method
- `parameter_names` (List[str]): Parameter names

**Returns**: Cached result or None

#### `clear_cache()`

Clear analysis cache.

#### `get_analysis_statistics() -> Dict[str, Any]`

Get analysis statistics.

**Returns**: Dictionary with cache size, SALib availability, and configuration

---

## SobolAnalyzer

Specialized Sobol sensitivity analyzer.

```python
from am_qadf.analytics.sensitivity_analysis import SobolAnalyzer

analyzer = SobolAnalyzer(config: Optional[SensitivityConfig] = None)
```

### Methods

#### `analyze(model_function: Callable, parameter_bounds: Dict[str, Tuple[float, float]], parameter_names: List[str] = None) -> SensitivityResult`

Perform Sobol analysis (alias for `analyze_sobol`).

---

## MorrisAnalyzer

Specialized Morris sensitivity analyzer.

```python
from am_qadf.analytics.sensitivity_analysis import MorrisAnalyzer

analyzer = MorrisAnalyzer(config: Optional[SensitivityConfig] = None)
```

### Methods

#### `analyze(model_function: Callable, parameter_bounds: Dict[str, Tuple[float, float]], parameter_names: List[str] = None) -> SensitivityResult`

Perform Morris analysis (alias for `analyze_morris`).

---

## LocalSensitivityAnalyzer

Local sensitivity analyzer for process parameters.

```python
from am_qadf.analytics.sensitivity_analysis import LocalSensitivityAnalyzer, LocalSensitivityConfig

analyzer = LocalSensitivityAnalyzer(config: Optional[LocalSensitivityConfig] = None)
```

### Methods

#### `analyze_derivatives(model_function: Callable, nominal_point: Dict[str, float], parameter_names: List[str] = None) -> LocalSensitivityResult`

Perform derivative-based local sensitivity analysis.

**Parameters**:
- `model_function` (Callable): Function that takes parameter array and returns output
- `nominal_point` (Dict[str, float]): Dictionary of nominal parameter values {name: value}
- `parameter_names` (List[str]): List of parameter names (optional)

**Returns**: `LocalSensitivityResult` with:
- `success` (bool): Whether analysis succeeded
- `method` (str): Analysis method ("Derivatives")
- `parameter_names` (List[str]): Parameter names
- `nominal_point` (Dict[str, float]): Nominal point
- `sensitivity_gradients` (Dict[str, float]): Sensitivity gradients
- `sensitivity_elasticities` (Dict[str, float]): Sensitivity elasticities
- `analysis_time` (float): Analysis time in seconds

#### `analyze_perturbation(model_function: Callable, nominal_point: Dict[str, float], parameter_names: List[str] = None, perturbation_size: float = None) -> LocalSensitivityResult`

Perform perturbation-based local sensitivity analysis.

**Parameters**:
- `model_function` (Callable): Function that takes parameter array and returns output
- `nominal_point` (Dict[str, float]): Dictionary of nominal parameter values {name: value}
- `parameter_names` (List[str]): List of parameter names (optional)
- `perturbation_size` (float): Size of parameter perturbation (optional)

**Returns**: `LocalSensitivityResult` with perturbation-based sensitivities

#### `analyze_central_differences(model_function: Callable, nominal_point: Dict[str, float], parameter_names: List[str] = None, step_size: float = None) -> LocalSensitivityResult`

Perform central difference local sensitivity analysis.

**Parameters**:
- `model_function` (Callable): Function that takes parameter array and returns output
- `nominal_point` (Dict[str, float]): Dictionary of nominal parameter values {name: value}
- `parameter_names` (List[str]): List of parameter names (optional)
- `step_size` (float): Step size for central differences (optional)

**Returns**: `LocalSensitivityResult` with central difference sensitivities

#### `analyze_automatic_differentiation(model_function: Callable, nominal_point: Dict[str, float], parameter_names: List[str] = None) -> LocalSensitivityResult`

Perform automatic differentiation local sensitivity analysis.

**Parameters**:
- `model_function` (Callable): Function that takes parameter array and returns output
- `nominal_point` (Dict[str, float]): Dictionary of nominal parameter values {name: value}
- `parameter_names` (List[str]): List of parameter names (optional)

**Returns**: `LocalSensitivityResult` with automatic differentiation sensitivities

---

## DerivativeAnalyzer

Specialized derivative-based sensitivity analyzer.

```python
from am_qadf.analytics.sensitivity_analysis import DerivativeAnalyzer

analyzer = DerivativeAnalyzer(config: Optional[LocalSensitivityConfig] = None)
```

### Methods

#### `analyze(model_function: Callable, nominal_point: Dict[str, float], parameter_names: List[str] = None) -> LocalSensitivityResult`

Perform derivative analysis (alias for `analyze_derivatives`).

---

## ExperimentalDesigner

Experimental designer for process analysis.

```python
from am_qadf.analytics.sensitivity_analysis import ExperimentalDesigner, DOEConfig

designer = ExperimentalDesigner(config: Optional[DOEConfig] = None)
```

### Methods

#### `create_factorial_design(parameter_bounds: Dict[str, Tuple[float, float]], parameter_names: List[str] = None, levels: int = None) -> ExperimentalDesign`

Create factorial experimental design.

**Parameters**:
- `parameter_bounds` (Dict[str, Tuple[float, float]]): Dictionary of parameter bounds {name: (min, max)}
- `parameter_names` (List[str]): List of parameter names (optional)
- `levels` (int): Number of levels (2 or 3)

**Returns**: `ExperimentalDesign` with:
- `design_type` (str): Design type (e.g., "2^k_factorial")
- `design_matrix` (pd.DataFrame): Design matrix
- `parameter_names` (List[str]): Parameter names
- `parameter_bounds` (Dict[str, Tuple[float, float]]): Parameter bounds
- `design_points` (int): Number of design points
- `design_quality` (Dict[str, float]): Design quality metrics

#### `create_response_surface_design(parameter_bounds: Dict[str, Tuple[float, float]], parameter_names: List[str] = None, design_type: str = None) -> ExperimentalDesign`

Create response surface experimental design.

**Parameters**:
- `parameter_bounds` (Dict[str, Tuple[float, float]]): Dictionary of parameter bounds {name: (min, max)}
- `parameter_names` (List[str]): List of parameter names (optional)
- `design_type` (str): Type of response surface design ("ccd", "bbd", "d_optimal")

**Returns**: `ExperimentalDesign` with response surface design

#### `create_optimal_design(parameter_bounds: Dict[str, Tuple[float, float]], parameter_names: List[str] = None, n_points: int = None, criterion: str = None) -> ExperimentalDesign`

Create optimal experimental design.

**Parameters**:
- `parameter_bounds` (Dict[str, Tuple[float, float]]): Dictionary of parameter bounds {name: (min, max)}
- `parameter_names` (List[str]): List of parameter names (optional)
- `n_points` (int): Number of design points
- `criterion` (str): Optimality criterion ("d_optimal", "a_optimal", "g_optimal")

**Returns**: `ExperimentalDesign` with optimal design

---

## UncertaintyQuantifier

Uncertainty quantifier for process analysis.

```python
from am_qadf.analytics.sensitivity_analysis import UncertaintyQuantifier, UncertaintyConfig

quantifier = UncertaintyQuantifier(config: Optional[UncertaintyConfig] = None)
```

### Methods

#### `analyze_monte_carlo(model_function: Callable, parameter_distributions: Dict[str, Dict[str, Any]], parameter_names: List[str] = None, n_samples: int = None) -> UncertaintyResult`

Perform Monte Carlo uncertainty analysis.

**Parameters**:
- `model_function` (Callable): Function that takes parameter array and returns output
- `parameter_distributions` (Dict[str, Dict[str, Any]]): Dictionary of parameter distributions {name: {type: str, params: dict}}
- `parameter_names` (List[str]): List of parameter names (optional)
- `n_samples` (int): Number of Monte Carlo samples (optional)

**Returns**: `UncertaintyResult` with:
- `success` (bool): Whether analysis succeeded
- `method` (str): Analysis method ("MonteCarlo")
- `parameter_names` (List[str]): Parameter names
- `parameter_distributions` (Dict[str, Dict[str, Any]]): Parameter distributions
- `output_statistics` (Dict[str, float]): Output statistics (mean, std, var, min, max, etc.)
- `confidence_intervals` (Dict[str, Tuple[float, float]]): Confidence intervals
- `sensitivity_analysis` (Dict[str, float]): Sensitivity analysis results
- `analysis_time` (float): Analysis time in seconds
- `sample_size` (int): Number of samples used

#### `analyze_bayesian(model_function: Callable, parameter_distributions: Dict[str, Dict[str, Any]], parameter_names: List[str] = None, observed_data: Optional[np.ndarray] = None) -> UncertaintyResult`

Perform Bayesian uncertainty analysis.

**Parameters**:
- `model_function` (Callable): Function that takes parameter array and returns output
- `parameter_distributions` (Dict[str, Dict[str, Any]]): Dictionary of parameter distributions {name: {type: str, params: dict}}
- `parameter_names` (List[str]): List of parameter names (optional)
- `observed_data` (Optional[np.ndarray]): Observed data for Bayesian inference (optional)

**Returns**: `UncertaintyResult` with Bayesian analysis results

#### `analyze_uncertainty_propagation(model_function: Callable, parameter_distributions: Dict[str, Dict[str, Any]], parameter_names: List[str] = None, method: str = None) -> UncertaintyResult`

Perform uncertainty propagation analysis.

**Parameters**:
- `model_function` (Callable): Function that takes parameter array and returns output
- `parameter_distributions` (Dict[str, Dict[str, Any]]): Dictionary of parameter distributions {name: {type: str, params: dict}}
- `parameter_names` (List[str]): List of parameter names (optional)
- `method` (str): Propagation method ("monte_carlo", "taylor", "polynomial_chaos")

**Returns**: `UncertaintyResult` with uncertainty propagation results

---

## SensitivityQuery

Query client for sensitivity analysis results.

```python
from am_qadf.analytics.sensitivity_analysis import SensitivityQuery

query = SensitivityQuery(mongo_client: MongoDBClient)
```

### Methods

#### `query_sensitivity_results(model_id: Optional[str] = None, method: Optional[str] = None, variable: Optional[str] = None, analysis_id: Optional[str] = None) -> List[Dict[str, Any]]`

Query sensitivity analysis results.

**Parameters**:
- `model_id` (Optional[str]): Model ID (optional)
- `method` (Optional[str]): Analysis method (optional)
- `variable` (Optional[str]): Variable name (optional)
- `analysis_id` (Optional[str]): Analysis ID (optional)

**Returns**: List of sensitivity result documents

#### `compare_sensitivity(model_ids: List[str], method: str = "sobol") -> pd.DataFrame`

Compare sensitivity across multiple models.

**Parameters**:
- `model_ids` (List[str]): List of model IDs
- `method` (str): Analysis method

**Returns**: DataFrame with comparison results

#### `analyze_sensitivity_trends(model_id: str, time_range: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, Any]`

Analyze sensitivity trends over time.

**Parameters**:
- `model_id` (str): Model ID
- `time_range` (Optional[Tuple[datetime, datetime]]): Optional time range

**Returns**: Dictionary with trend analysis

---

## SensitivityStorage

Storage client for sensitivity analysis results.

```python
from am_qadf.analytics.sensitivity_analysis import SensitivityStorage

storage = SensitivityStorage(mongo_client: MongoDBClient)
```

### Methods

#### `store_sensitivity_result(result: SensitivityResult) -> str`

Store sensitivity analysis result.

**Parameters**:
- `result` (SensitivityResult): SensitivityResult object

**Returns**: Document ID

#### `store_doe_design(model_id: str, design_id: str, design_data: Dict[str, Any]) -> str`

Store Design of Experiments (DoE) design.

**Parameters**:
- `model_id` (str): Model ID
- `design_id` (str): Design ID
- `design_data` (Dict[str, Any]): Design data dictionary

**Returns**: Document ID

#### `store_influence_rankings(model_id: str, ranking_id: str, rankings: Dict[str, float]) -> str`

Store process variable influence rankings.

**Parameters**:
- `model_id` (str): Model ID
- `ranking_id` (str): Ranking ID
- `rankings` (Dict[str, float]): Dictionary mapping variable names to influence scores

**Returns**: Document ID

---

## ProcessAnalysisClient

Process analysis operations. Note: The process analysis module consists of four main components: Parameter Analysis, Quality Analysis, Sensor Analysis, and Process Optimization. Each component has its own classes and methods documented below.

---

## ParameterAnalyzer

Process parameter analyzer for PBF-LB/M systems.

```python
from am_qadf.analytics.process_analysis import ParameterAnalyzer, ParameterAnalysisConfig

analyzer = ParameterAnalyzer(config: Optional[ParameterAnalysisConfig] = None)
```

### Methods

#### `analyze_parameter_optimization(objective_function: Callable, parameter_bounds: Dict[str, Tuple[float, float]], parameter_names: List[str] = None, optimization_method: str = None) -> ParameterAnalysisResult`

Perform parameter optimization analysis.

**Parameters**:
- `objective_function` (Callable): Function to optimize (should return scalar value)
- `parameter_bounds` (Dict[str, Tuple[float, float]]): Dictionary of parameter bounds {name: (min, max)}
- `parameter_names` (List[str]): List of parameter names (optional)
- `optimization_method` (str): Optimization method ("minimize", "differential_evolution") (optional)

**Returns**: `ParameterAnalysisResult` with:
- `success` (bool): Whether analysis succeeded
- `method` (str): Analysis method
- `parameter_names` (List[str]): Parameter names
- `optimal_parameters` (Dict[str, float]): Optimal parameter values
- `optimal_value` (float): Optimal objective value
- `parameter_interactions` (Dict[str, Dict[str, float]]): Parameter interactions
- `parameter_importance` (Dict[str, float]): Parameter importance scores
- `analysis_time` (float): Analysis time in seconds

#### `analyze_parameter_interactions(process_data: pd.DataFrame, parameter_names: List[str] = None, target_variable: str = None) -> ParameterAnalysisResult`

Analyze parameter interactions in process data.

**Parameters**:
- `process_data` (pd.DataFrame): DataFrame containing process data
- `parameter_names` (List[str]): List of parameter names (optional)
- `target_variable` (str): Target variable name (optional)

**Returns**: `ParameterAnalysisResult` with parameter interaction analysis

#### `analyze_parameter_sensitivity(objective_function: Callable, nominal_parameters: Dict[str, float], parameter_bounds: Dict[str, Tuple[float, float]], parameter_names: List[str] = None) -> ParameterAnalysisResult`

Analyze parameter sensitivity around nominal point.

**Parameters**:
- `objective_function` (Callable): Function to analyze
- `nominal_parameters` (Dict[str, float]): Dictionary of nominal parameter values
- `parameter_bounds` (Dict[str, Tuple[float, float]]): Dictionary of parameter bounds
- `parameter_names` (List[str]): List of parameter names (optional)

**Returns**: `ParameterAnalysisResult` with parameter sensitivity analysis

#### `get_cached_result(method: str, parameter_names: List[str]) -> Optional[ParameterAnalysisResult]`

Get cached analysis result.

#### `clear_cache()`

Clear analysis cache.

#### `get_analysis_statistics() -> Dict[str, Any]`

Get analysis statistics.

---

## ParameterAnalysisConfig

Configuration dataclass for parameter analysis.

```python
from am_qadf.analytics.process_analysis import ParameterAnalysisConfig

config = ParameterAnalysisConfig(
    optimization_method: str = "differential_evolution",
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
    interaction_threshold: float = 0.3,
    correlation_method: str = "pearson",
    confidence_level: float = 0.95,
    significance_level: float = 0.05,
    random_seed: Optional[int] = None
)
```

---

## ProcessParameterOptimizer

Specialized process parameter optimizer.

```python
from am_qadf.analytics.process_analysis import ProcessParameterOptimizer

optimizer = ProcessParameterOptimizer(config: Optional[ParameterAnalysisConfig] = None)
```

### Methods

#### `optimize(objective_function: Callable, parameter_bounds: Dict[str, Tuple[float, float]], parameter_names: List[str] = None) -> ParameterAnalysisResult`

Optimize process parameters (alias for `analyze_parameter_optimization`).

---

## QualityAnalyzer

Quality analyzer for PBF-LB/M systems.

```python
from am_qadf.analytics.process_analysis import QualityAnalyzer, QualityAnalysisConfig

analyzer = QualityAnalyzer(config: Optional[QualityAnalysisConfig] = None)
```

### Methods

#### `analyze_quality_prediction(process_data: pd.DataFrame, quality_target: str, feature_names: List[str] = None) -> QualityAnalysisResult`

Perform quality prediction analysis.

**Parameters**:
- `process_data` (pd.DataFrame): DataFrame containing process data
- `quality_target` (str): Name of quality target variable
- `feature_names` (List[str]): List of feature names (optional)

**Returns**: `QualityAnalysisResult` with:
- `success` (bool): Whether analysis succeeded
- `method` (str): Analysis method ("QualityPrediction")
- `quality_metrics` (Dict[str, float]): Quality metrics (mean, std, min, max, range)
- `quality_predictions` (np.ndarray): Quality predictions
- `quality_classifications` (np.ndarray): Quality classifications (0=low, 1=medium, 2=high)
- `model_performance` (Dict[str, float]): Model performance (r2_score, mse, rmse)
- `analysis_time` (float): Analysis time in seconds

#### `get_cached_result(method: str) -> Optional[QualityAnalysisResult]`

Get cached analysis result.

#### `clear_cache()`

Clear analysis cache.

#### `get_analysis_statistics() -> Dict[str, Any]`

Get analysis statistics.

---

## QualityAnalysisConfig

Configuration dataclass for quality analysis.

```python
from am_qadf.analytics.process_analysis import QualityAnalysisConfig

config = QualityAnalysisConfig(
    model_type: str = "random_forest",  # "random_forest", "gradient_boosting"
    test_size: float = 0.2,
    random_seed: Optional[int] = None,
    quality_threshold: float = 0.8,
    defect_threshold: float = 0.1,
    confidence_level: float = 0.95
)
```

---

## QualityPredictor

Specialized quality predictor.

```python
from am_qadf.analytics.process_analysis import QualityPredictor

predictor = QualityPredictor(config: Optional[QualityAnalysisConfig] = None)
```

### Methods

#### `predict(process_data: pd.DataFrame, quality_target: str, feature_names: List[str] = None) -> QualityAnalysisResult`

Predict quality from process data (alias for `analyze_quality_prediction`).

---

## SensorAnalyzer

Sensor analyzer for PBF-LB/M systems.

```python
from am_qadf.analytics.process_analysis import SensorAnalyzer, SensorAnalysisConfig

analyzer = SensorAnalyzer(config: Optional[SensorAnalysisConfig] = None)
```

### Methods

#### `analyze_sensor_data(sensor_data: pd.DataFrame, sensor_columns: List[str] = None) -> SensorAnalysisResult`

Perform comprehensive sensor data analysis.

**Parameters**:
- `sensor_data` (pd.DataFrame): DataFrame containing sensor data
- `sensor_columns` (List[str]): List of sensor column names (optional)

**Returns**: `SensorAnalysisResult` with:
- `success` (bool): Whether analysis succeeded
- `method` (str): Analysis method ("SensorAnalysis")
- `sensor_data` (pd.DataFrame): Original sensor data
- `processed_data` (pd.DataFrame): Processed sensor data (filtered, normalized)
- `anomaly_detection` (Dict[str, Any]): Anomaly detection results
- `signal_statistics` (Dict[str, float]): Signal statistics
- `analysis_time` (float): Analysis time in seconds

#### `get_cached_result(method: str, sensor_columns: List[str]) -> Optional[SensorAnalysisResult]`

Get cached analysis result.

#### `clear_cache()`

Clear analysis cache.

#### `get_analysis_statistics() -> Dict[str, Any]`

Get analysis statistics.

---

## SensorAnalysisConfig

Configuration dataclass for sensor analysis.

```python
from am_qadf.analytics.process_analysis import SensorAnalysisConfig

config = SensorAnalysisConfig(
    sampling_rate: float = 1000.0,  # Hz
    filter_type: str = "butterworth",  # "butterworth", "chebyshev", "ellip"
    filter_order: int = 4,
    cutoff_frequency: float = 100.0,  # Hz
    anomaly_threshold: float = 3.0,  # Standard deviations
    window_size: int = 100,
    confidence_level: float = 0.95
)
```

---

## ISPMAnalyzer

Specialized ISPM sensor analyzer.

```python
from am_qadf.analytics.process_analysis import ISPMAnalyzer

analyzer = ISPMAnalyzer(config: Optional[SensorAnalysisConfig] = None)
```

### Methods

#### `analyze_ispm_data(ispm_data: pd.DataFrame) -> SensorAnalysisResult`

Analyze ISPM sensor data.

**Parameters**:
- `ispm_data` (pd.DataFrame): ISPM sensor data

**Returns**: `SensorAnalysisResult` with ISPM-specific analysis

---

## CTSensorAnalyzer

Specialized CT sensor analyzer.

```python
from am_qadf.analytics.process_analysis import CTSensorAnalyzer

analyzer = CTSensorAnalyzer(config: Optional[SensorAnalysisConfig] = None)
```

### Methods

#### `analyze_ct_data(ct_data: pd.DataFrame) -> SensorAnalysisResult`

Analyze CT sensor data.

**Parameters**:
- `ct_data` (pd.DataFrame): CT sensor data

**Returns**: `SensorAnalysisResult` with CT-specific analysis

---

## ProcessOptimizer

Process optimizer for PBF-LB/M systems.

```python
from am_qadf.analytics.process_analysis import ProcessOptimizer, OptimizationConfig

optimizer = ProcessOptimizer(config: Optional[OptimizationConfig] = None)
```

### Methods

#### `optimize_single_objective(objective_function: Callable, parameter_bounds: Dict[str, Tuple[float, float]], parameter_names: List[str] = None, optimization_method: str = None) -> OptimizationResult`

Perform single-objective optimization.

**Parameters**:
- `objective_function` (Callable): Function to optimize (should return scalar value)
- `parameter_bounds` (Dict[str, Tuple[float, float]]): Dictionary of parameter bounds {name: (min, max)}
- `parameter_names` (List[str]): List of parameter names (optional)
- `optimization_method` (str): Optimization method ("minimize", "differential_evolution") (optional)

**Returns**: `OptimizationResult` with:
- `success` (bool): Whether optimization succeeded
- `method` (str): Optimization method
- `parameter_names` (List[str]): Parameter names
- `optimal_parameters` (Dict[str, float]): Optimal parameter values
- `optimal_values` (float): Optimal objective value
- `analysis_time` (float): Optimization time in seconds

#### `optimize_multi_objective(objective_functions: List[Callable], parameter_bounds: Dict[str, Tuple[float, float]], parameter_names: List[str] = None, optimization_method: str = None) -> OptimizationResult`

Perform multi-objective optimization.

**Parameters**:
- `objective_functions` (List[Callable]): List of functions to optimize
- `parameter_bounds` (Dict[str, Tuple[float, float]]): Dictionary of parameter bounds {name: (min, max)}
- `parameter_names` (List[str]): List of parameter names (optional)
- `optimization_method` (str): Optimization method ("nsga2", "weighted_sum") (optional)

**Returns**: `OptimizationResult` with:
- `success` (bool): Whether optimization succeeded
- `method` (str): Optimization method
- `parameter_names` (List[str]): Parameter names
- `optimal_parameters` (Dict[str, float]): Optimal parameter values (may be empty for multi-objective)
- `optimal_values` (List[float]): Multiple optimal objective values
- `pareto_front` (pd.DataFrame): Pareto front solutions
- `analysis_time` (float): Optimization time in seconds

#### `get_cached_result(method: str, parameter_names: List[str]) -> Optional[OptimizationResult]`

Get cached optimization result.

#### `clear_cache()`

Clear optimization cache.

#### `get_optimization_statistics() -> Dict[str, Any]`

Get optimization statistics.

---

## OptimizationConfig

Configuration dataclass for process optimization.

```python
from am_qadf.analytics.process_analysis import OptimizationConfig

config = OptimizationConfig(
    optimization_method: str = "differential_evolution",  # "minimize", "differential_evolution", "nsga2"
    max_iterations: int = 1000,
    population_size: int = 50,
    tolerance: float = 1e-6,
    n_objectives: int = 2,
    pareto_front_size: int = 100,
    random_seed: Optional[int] = None
)
```

---

## MultiObjectiveOptimizer

Specialized multi-objective optimizer.

```python
from am_qadf.analytics.process_analysis import MultiObjectiveOptimizer

optimizer = MultiObjectiveOptimizer(config: Optional[OptimizationConfig] = None)
```

### Methods

#### `optimize(objective_functions: List[Callable], parameter_bounds: Dict[str, Tuple[float, float]], parameter_names: List[str] = None) -> OptimizationResult`

Optimize multiple objectives (alias for `optimize_multi_objective`).

---

## VirtualExperimentClient

Main client for virtual experiment operations with warehouse data integration.

```python
from am_qadf.analytics.virtual_experiments import VirtualExperimentClient, VirtualExperimentConfig

client = VirtualExperimentClient(
    unified_query_client: UnifiedQueryClient,
    voxel_domain_client: Optional[VoxelDomainClient] = None
)
```

### Attributes

- `unified_client` (UnifiedQueryClient): Unified query client for warehouse data
- `voxel_client` (Optional[VoxelDomainClient]): Optional voxel domain client
- `experiment_designer` (Optional[VirtualExperimentDesigner]): Virtual experiment designer

### Methods

#### `query_historical_builds(model_type: Optional[str] = None, process_conditions: Optional[List[str]] = None, limit: int = 100) -> pd.DataFrame`

Query historical build data from warehouse.

**Parameters**:
- `model_type` (Optional[str]): Optional model type filter
- `process_conditions` (Optional[List[str]]): Optional list of process condition filters (e.g., ['laser_power > 200'])
- `limit` (int): Maximum number of builds to return

**Returns**: DataFrame with historical build data

#### `get_parameter_ranges_from_warehouse(model_ids: Optional[List[str]] = None, variables: List[str] = None) -> Dict[str, Tuple[float, float]]`

Get parameter ranges from warehouse data.

**Parameters**:
- `model_ids` (Optional[List[str]]): Optional list of model IDs to consider
- `variables` (List[str]): List of variable names

**Returns**: Dictionary mapping variable names to (min, max) ranges

#### `design_experiment(base_model_id: str, parameter_ranges: Optional[Dict[str, Tuple[float, float]]] = None, config: VirtualExperimentConfig = None) -> Dict[str, Any]`

Design virtual experiment based on warehouse data.

**Parameters**:
- `base_model_id` (str): Base model ID
- `parameter_ranges` (Optional[Dict[str, Tuple[float, float]]]): Optional parameter ranges (if None, will query from warehouse)
- `config` (VirtualExperimentConfig): Experiment configuration

**Returns**: Dictionary with experiment design containing:
- `experiment_id` (str): Experiment ID
- `base_model_id` (str): Base model ID
- `parameter_ranges` (Dict[str, Tuple[float, float]]): Parameter ranges
- `design_points` (List[Dict[str, float]]): Design points
- `design_type` (str): Design type
- `num_samples` (int): Number of samples
- `timestamp` (datetime): Design timestamp

#### `compare_with_warehouse(experiment_id: str, model_id: str, comparison_metrics: List[str] = None) -> Dict[str, Any]`

Compare experiment results with warehouse data.

**Parameters**:
- `experiment_id` (str): Experiment ID
- `model_id` (str): Model ID to compare with
- `comparison_metrics` (List[str]): List of metrics to compare

**Returns**: Dictionary with comparison results

---

## VirtualExperimentConfig

Configuration dataclass for virtual experiments.

```python
from am_qadf.analytics.virtual_experiments import VirtualExperimentConfig

config = VirtualExperimentConfig(
    experiment_type: str = "parameter_sweep",  # "parameter_sweep", "optimization", "validation"
    base_model_id: str = None,
    parameter_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    use_warehouse_ranges: bool = True,
    design_type: str = "factorial",  # "factorial", "lhs", "random"
    num_samples: int = 100,
    compare_with_warehouse: bool = True,
    comparison_metrics: List[str] = None,
    spatial_region: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None,
    layer_range: Optional[Tuple[int, int]] = None
)
```

### Attributes

- `experiment_type` (str): Experiment type ("parameter_sweep", "optimization", "validation")
- `base_model_id` (str): Base model ID
- `parameter_ranges` (Optional[Dict[str, Tuple[float, float]]]): Parameter ranges
- `use_warehouse_ranges` (bool): Whether to use parameter ranges from warehouse
- `design_type` (str): Design type ("factorial", "lhs", "random")
- `num_samples` (int): Number of samples
- `compare_with_warehouse` (bool): Whether to compare with warehouse data
- `comparison_metrics` (List[str]): Metrics to compare
- `spatial_region` (Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]): Spatial region
- `layer_range` (Optional[Tuple[int, int]]): Layer range

---

## VirtualExperimentResultAnalyzer

Analyzer for virtual experiment results.

```python
from am_qadf.analytics.virtual_experiments import VirtualExperimentResultAnalyzer

analyzer = VirtualExperimentResultAnalyzer()
```

### Methods

#### `analyze_results(experiment_results: List[Dict[str, Any]], parameter_names: List[str], response_names: List[str]) -> AnalysisResult`

Analyze virtual experiment results.

**Parameters**:
- `experiment_results` (List[Dict[str, Any]]): List of experiment result dictionaries
- `parameter_names` (List[str]): List of parameter names
- `response_names` (List[str]): List of response variable names

**Returns**: `AnalysisResult` with:
- `experiment_id` (str): Experiment ID
- `analysis_type` (str): Analysis type ("comprehensive")
- `parameter_names` (List[str]): Parameter names
- `response_names` (List[str]): Response names
- `response_statistics` (Dict[str, Dict[str, float]]): Response statistics (mean, std, min, max, median, q25, q75, skewness, kurtosis)
- `correlations` (Dict[str, Dict[str, float]]): Parameter-response correlations (Pearson correlation, p-value, significance)
- `parameter_interactions` (Dict[str, float]): Parameter interactions
- `analysis_time` (float): Analysis time in seconds
- `sample_size` (int): Number of samples
- `success` (bool): Whether analysis succeeded

#### `compare_with_sensitivity_analysis(virtual_results: AnalysisResult, sensitivity_results: Dict[str, Any]) -> Dict[str, Any]`

Compare virtual experiment results with sensitivity analysis.

**Parameters**:
- `virtual_results` (AnalysisResult): Analysis results from virtual experiments
- `sensitivity_results` (Dict[str, Any]): Sensitivity analysis results

**Returns**: Dictionary with comparison results

---

## ComparisonAnalyzer

Analyzer for comparing virtual experiment results with sensitivity analysis.

```python
from am_qadf.analytics.virtual_experiments import ComparisonAnalyzer

analyzer = ComparisonAnalyzer()
```

### Methods

#### `compare_parameter_importance(virtual_results: Dict[str, Any], sensitivity_results: Dict[str, Any], response_name: str = 'quality') -> ComparisonResult`

Compare parameter importance rankings between virtual experiments and sensitivity analysis.

**Parameters**:
- `virtual_results` (Dict[str, Any]): Virtual experiment analysis results
- `sensitivity_results` (Dict[str, Any]): Sensitivity analysis results
- `response_name` (str): Response variable name to compare

**Returns**: `ComparisonResult` with:
- `success` (bool): Whether comparison succeeded
- `parameter_rankings_virtual` (Dict[str, float]): Parameter rankings from virtual experiments
- `parameter_rankings_sensitivity` (Dict[str, float]): Parameter rankings from sensitivity analysis
- `ranking_correlation` (float): Correlation between rankings
- `agreement_metrics` (Dict[str, float]): Agreement metrics (agreement, top3_agreement, common_parameters)
- `discrepancies` (List[str]): List of significant discrepancies

---

## ParameterOptimizer

Parameter optimizer for virtual experiment results.

```python
from am_qadf.analytics.virtual_experiments import ParameterOptimizer

optimizer = ParameterOptimizer()
```

### Methods

#### `optimize_single_objective(experiment_results: List[Dict[str, Any]], parameter_names: List[str], objective_name: str, maximize: bool = True, constraints: Dict[str, Tuple[float, float]] = None) -> OptimizationResult`

Optimize parameters for a single objective.

**Parameters**:
- `experiment_results` (List[Dict[str, Any]]): List of experiment result dictionaries
- `parameter_names` (List[str]): List of parameter names
- `objective_name` (str): Name of objective to optimize
- `maximize` (bool): Whether to maximize (True) or minimize (False)
- `constraints` (Dict[str, Tuple[float, float]]): Parameter constraints {name: (min, max)}

**Returns**: `OptimizationResult` with:
- `success` (bool): Whether optimization succeeded
- `optimal_parameters` (Dict[str, float]): Optimal parameter values
- `optimal_objectives` (Dict[str, float]): Optimal objective values
- `optimization_method` (str): Optimization method used
- `iterations` (int): Number of iterations
- `convergence_info` (Dict[str, Any]): Convergence information

#### `optimize_multi_objective(experiment_results: List[Dict[str, Any]], parameter_names: List[str], objective_names: List[str], objective_directions: List[str] = None, constraints: Dict[str, Tuple[float, float]] = None) -> OptimizationResult`

Optimize parameters for multiple objectives (Pareto optimization).

**Parameters**:
- `experiment_results` (List[Dict[str, Any]]): List of experiment result dictionaries
- `parameter_names` (List[str]): List of parameter names
- `objective_names` (List[str]): List of objective names
- `objective_directions` (List[str]): List of 'maximize' or 'minimize' for each objective
- `constraints` (Dict[str, Tuple[float, float]]): Parameter constraints

**Returns**: `OptimizationResult` with:
- `success` (bool): Whether optimization succeeded
- `optimal_parameters` (Dict[str, float]): Representative optimal parameter values
- `optimal_objectives` (Dict[str, float]): Representative optimal objective values
- `optimization_method` (str): Optimization method ("pareto")
- `iterations` (int): Number of Pareto solutions
- `convergence_info` (Dict[str, Any]): Convergence information
- `pareto_front` (List[Dict[str, Any]]): Pareto front solutions

---

## ExperimentQuery

Query client for virtual experiment results.

```python
from am_qadf.analytics.virtual_experiments import ExperimentQuery

query = ExperimentQuery(mongo_client: MongoDBClient)
```

### Methods

#### `query_experiment_results(experiment_id: Optional[str] = None, model_id: Optional[str] = None, design_type: Optional[str] = None) -> List[Dict[str, Any]]`

Query virtual experiment results.

**Parameters**:
- `experiment_id` (Optional[str]): Experiment ID (optional)
- `model_id` (Optional[str]): Model ID (optional)
- `design_type` (Optional[str]): Design type (optional)

**Returns**: List of experiment result documents

#### `compare_experiments_with_warehouse(experiment_id: str, model_id: str) -> Dict[str, Any]`

Compare experiment results with warehouse data.

**Parameters**:
- `experiment_id` (str): Experiment ID
- `model_id` (str): Model ID to compare with

**Returns**: Dictionary with comparison results

#### `analyze_experiment_trends(model_id: Optional[str] = None, time_range: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, Any]`

Analyze experiment trends.

**Parameters**:
- `model_id` (Optional[str]): Optional model ID
- `time_range` (Optional[Tuple[datetime, datetime]]): Optional time range

**Returns**: Dictionary with trend analysis

---

## ExperimentStorage

Storage client for virtual experiment results.

```python
from am_qadf.analytics.virtual_experiments import ExperimentStorage

storage = ExperimentStorage(mongo_client: MongoDBClient)
```

### Methods

#### `store_experiment_result(result: ExperimentResult) -> str`

Store virtual experiment result.

**Parameters**:
- `result` (ExperimentResult): ExperimentResult object

**Returns**: Document ID

#### `store_experiment_design(experiment_id: str, design_data: Dict[str, Any]) -> str`

Store experiment design.

**Parameters**:
- `experiment_id` (str): Experiment ID
- `design_data` (Dict[str, Any]): Design data dictionary

**Returns**: Document ID

#### `store_comparison_results(experiment_id: str, model_id: str, comparison_data: Dict[str, Any]) -> str`

Store comparison results with warehouse data.

**Parameters**:
- `experiment_id` (str): Experiment ID
- `model_id` (str): Model ID
- `comparison_data` (Dict[str, Any]): Comparison data dictionary

**Returns**: Document ID

---

## ExperimentResult

Virtual experiment result data structure.

```python
from am_qadf.analytics.virtual_experiments import ExperimentResult

result = ExperimentResult(
    experiment_id: str,
    model_id: str,
    design_data: Dict[str, Any],
    results: Dict[str, Any],
    comparison_results: Optional[Dict[str, Any]] = None,
    validation_results: Optional[Dict[str, Any]] = None,
    timestamp: datetime = None
)
```

### Attributes

- `experiment_id` (str): Experiment ID
- `model_id` (str): Model ID
- `design_data` (Dict[str, Any]): Design data
- `results` (Dict[str, Any]): Experiment results
- `comparison_results` (Optional[Dict[str, Any]]): Comparison results
- `validation_results` (Optional[Dict[str, Any]]): Validation results
- `timestamp` (datetime): Timestamp

### Methods

#### `to_dict() -> Dict[str, Any]`

Convert to dictionary for storage.

---

## Related

- [Analytics Module Documentation](../05-modules/analytics.md) - Module overview
- [All API References](README.md) - Other API references

---

**Parent**: [API Reference](README.md)

