# Validation Module API Reference

## Overview

The Validation module provides comprehensive validation and benchmarking capabilities for the AM-QADF framework, including performance benchmarking, MPM system comparison, accuracy validation, and statistical significance testing.

## ValidationConfig

Configuration dataclass for validation operations.

```python
from am_qadf.validation import ValidationConfig

config = ValidationConfig(
    confidence_level: float = 0.95,
    significance_level: float = 0.05,
    max_acceptable_error: float = 0.1,
    correlation_threshold: float = 0.85,
    sample_size: Optional[int] = None,
    random_seed: Optional[int] = None,
    enable_benchmarking: bool = True,
    enable_mpm_comparison: bool = True,
    enable_accuracy_validation: bool = True,
    enable_statistical_validation: bool = True
)
```

### Fields

- `confidence_level` (float): Confidence level for statistical tests (default: 0.95)
- `significance_level` (float): Significance level (α) for hypothesis tests (default: 0.05)
- `max_acceptable_error` (float): Maximum acceptable error for accuracy validation (default: 0.1)
- `correlation_threshold` (float): Minimum correlation coefficient for MPM comparison (default: 0.85)
- `sample_size` (Optional[int]): Optional sample size for statistical tests
- `random_seed` (Optional[int]): Optional random seed for reproducibility
- `enable_benchmarking` (bool): Enable performance benchmarking (default: True)
- `enable_mpm_comparison` (bool): Enable MPM comparison (default: True)
- `enable_accuracy_validation` (bool): Enable accuracy validation (default: True)
- `enable_statistical_validation` (bool): Enable statistical validation (default: True)

### Methods

#### `to_dict() -> Dict[str, Any]`

Convert configuration to dictionary.

**Returns**: Dictionary representation of configuration

---

## ValidationClient

Main client for comprehensive validation operations.

```python
from am_qadf.validation import ValidationClient, ValidationConfig

client = ValidationClient(config: Optional[ValidationConfig] = None)
```

### Methods

#### `__init__(config: Optional[ValidationConfig] = None)`

Initialize the validation client.

**Parameters**:
- `config` (Optional[ValidationConfig]): Validation configuration. If None, uses default config.

#### `benchmark_operation(operation: Callable, *args, iterations: int = 1, **kwargs) -> Optional[BenchmarkResult]`

Benchmark a framework operation.

**Parameters**:
- `operation` (Callable): Function or method to benchmark
- `*args`: Positional arguments for the operation
- `iterations` (int): Number of iterations to run (for averaging, default: 1)
- `**kwargs`: Keyword arguments for the operation

**Returns**: `BenchmarkResult` if benchmarking is available, `None` otherwise

#### `compare_with_mpm(framework_data: Any, mpm_data: Any, metrics: Optional[List[str]] = None) -> Dict[str, Optional[MPMComparisonResult]]`

Compare framework outputs with MPM system outputs.

**Parameters**:
- `framework_data` (Any): Framework-generated data (dict, array, or object)
- `mpm_data` (Any): MPM system data (dict, array, or object)
- `metrics` (Optional[List[str]]): List of metric names to compare. If None, compares all available.

**Returns**: Dictionary mapping metric names to `MPMComparisonResult`

#### `validate_mpm_correlation(framework_values: np.ndarray, mpm_values: np.ndarray, method: str = "pearson") -> float`

Validate correlation between framework and MPM values.

**Parameters**:
- `framework_values` (np.ndarray): Framework values
- `mpm_values` (np.ndarray): MPM system values
- `method` (str): Correlation method - "pearson", "spearman", or "kendall" (default: "pearson")

**Returns**: Correlation coefficient (float)

#### `validate_accuracy(framework_data: Any, ground_truth: Any, validation_type: str = "signal") -> Optional[AccuracyValidationResult]`

Validate framework accuracy against ground truth.

**Parameters**:
- `framework_data` (Any): Framework-generated data
- `ground_truth` (Any): Ground truth data
- `validation_type` (str): Type of validation - "signal", "spatial", "temporal", or "quality" (default: "signal")

**Returns**: `AccuracyValidationResult` if validation is available, `None` otherwise

#### `perform_statistical_test(sample1: np.ndarray, sample2: Optional[np.ndarray] = None, test_type: str = "t_test", alternative: str = "two-sided", **kwargs) -> Optional[StatisticalValidationResult]`

Perform statistical hypothesis test.

**Parameters**:
- `sample1` (np.ndarray): First sample or single sample for normality test
- `sample2` (Optional[np.ndarray]): Second sample (required for comparison tests, None for normality test)
- `test_type` (str): Test type - "t_test", "mann_whitney", "correlation", "chi_square", "anova", or "normality" (default: "t_test")
- `alternative` (str): Alternative hypothesis - "two-sided", "less", or "greater" (default: "two-sided")
- `**kwargs`: Additional test-specific parameters

**Returns**: `StatisticalValidationResult` if validation is available, `None` otherwise

#### `validate_improvement(baseline: np.ndarray, improved: np.ndarray, test_type: str = "t_test", alternative: str = "greater") -> Optional[StatisticalValidationResult]`

Validate if improved version is significantly better than baseline.

**Parameters**:
- `baseline` (np.ndarray): Baseline sample values
- `improved` (np.ndarray): Improved sample values
- `test_type` (str): Test type - "t_test" or "mann_whitney" (default: "t_test")
- `alternative` (str): Alternative hypothesis - "greater" (improved > baseline), "less", or "two-sided" (default: "greater")

**Returns**: `StatisticalValidationResult` if validation is available, `None` otherwise

#### `comprehensive_validation(framework_output: Dict[str, Any], reference_data: Any, config: Optional[ValidationConfig] = None) -> Dict[str, Any]`

Run all applicable validation checks.

**Parameters**:
- `framework_output` (Dict[str, Any]): Framework output data
- `reference_data` (Any): Reference data (MPM, ground truth, or baseline)
- `config` (Optional[ValidationConfig]): Optional validation configuration

**Returns**: Dictionary containing all validation results with keys:
- `'benchmark'`: BenchmarkResult (if benchmarking enabled)
- `'mpm_comparison'`: Dict[str, MPMComparisonResult] (if MPM comparison enabled)
- `'accuracy'`: AccuracyValidationResult (if accuracy validation enabled)
- `'statistical'`: StatisticalValidationResult (if statistical validation enabled)

#### `validate_all(framework_output: Dict[str, Any], reference_data: Any, config: Optional[ValidationConfig] = None) -> Dict[str, Any]`

Alias for `comprehensive_validation()`.

#### `generate_validation_report(results: Dict[str, Any]) -> str`

Generate comprehensive validation report.

**Parameters**:
- `results` (Dict[str, Any]): Validation results dictionary (typically from `comprehensive_validation()`)

**Returns**: Human-readable validation report string

---

## BenchmarkResult

Result dataclass for benchmarking operations.

```python
from am_qadf.validation import BenchmarkResult

@dataclass
class BenchmarkResult:
    operation_name: str
    execution_time: float  # seconds
    memory_usage: float  # MB
    data_volume: int  # bytes or number of elements
    throughput: float  # operations per second
    iterations: int = 1
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### Fields

- `operation_name` (str): Name of the benchmarked operation
- `execution_time` (float): Average execution time in seconds
- `memory_usage` (float): Peak memory usage in MB
- `data_volume` (int): Data volume processed (bytes or element count)
- `throughput` (float): Operations per second
- `iterations` (int): Number of iterations run
- `timestamp` (datetime): Timestamp of benchmark
- `metadata` (Dict[str, Any]): Additional metadata

### Methods

#### `to_dict() -> Dict[str, Any]`

Convert result to dictionary.

**Returns**: Dictionary representation of result

---

## PerformanceBenchmarker

Performance benchmarking utility for framework operations.

```python
from am_qadf.validation import PerformanceBenchmarker

benchmarker = PerformanceBenchmarker()
```

### Methods

#### `__init__()`

Initialize the performance benchmarker.

#### `benchmark_operation(operation_name: str, operation: Callable, *args, iterations: int = 1, warmup_iterations: int = 0, **kwargs) -> BenchmarkResult`

Benchmark a single operation.

**Parameters**:
- `operation_name` (str): Name of the operation being benchmarked
- `operation` (Callable): Function or method to benchmark
- `*args`: Positional arguments for the operation
- `iterations` (int): Number of iterations to run (for averaging, default: 1)
- `warmup_iterations` (int): Number of warmup iterations (excluded from timing, default: 0)
- `**kwargs`: Keyword arguments for the operation

**Returns**: `BenchmarkResult` with performance metrics

#### `benchmark_function(func: Callable, function_name: str = None, num_runs: int = 5, warmup: int = 1, *args, **kwargs) -> BenchmarkResult`

Benchmark a function with arguments.

**Parameters**:
- `func` (Callable): Function to benchmark
- `function_name` (str): Optional function name (default: uses `func.__name__`)
- `num_runs` (int): Number of runs for benchmarking (default: 5)
- `warmup` (int): Number of warmup iterations (default: 1)
- `*args`: Positional arguments for the function
- `**kwargs`: Keyword arguments for the function

**Returns**: `BenchmarkResult` with performance metrics

#### `compare_operations(operations: Dict[str, Callable], *args, iterations: int = 5, **kwargs) -> Dict[str, BenchmarkResult]`

Compare multiple operations.

**Parameters**:
- `operations` (Dict[str, Callable]): Dictionary mapping operation names to callables
- `*args`: Positional arguments for all operations
- `iterations` (int): Number of iterations to run (default: 5)
- `**kwargs`: Keyword arguments for all operations

**Returns**: Dictionary mapping operation names to `BenchmarkResult`

#### `generate_report(results: Union[BenchmarkResult, Dict[str, BenchmarkResult]], output_file: Optional[str] = None) -> str`

Generate benchmarking report.

**Parameters**:
- `results` (Union[BenchmarkResult, Dict[str, BenchmarkResult]]): Single result or dictionary of results
- `output_file` (Optional[str]): Optional file path to save report

**Returns**: Human-readable benchmarking report string

---

## benchmark (Decorator)

Decorator for automatically benchmarking functions or methods.

```python
from am_qadf.validation.benchmarking import benchmark

@benchmark(num_runs=5, warmup=1)
def my_function(x, y):
    return x + y
```

### Parameters

- `num_runs` (int): Number of benchmark runs (default: 5)
- `warmup` (int): Number of warmup iterations (default: 1)
- `store_result` (bool): Whether to store benchmark result in function metadata (default: True)

### Usage

The decorator automatically benchmarks function execution. Results are available in `func.__benchmark_result__` if `store_result=True`.

---

## MPMComparisonResult

Result dataclass for MPM comparison operations.

```python
from am_qadf.validation import MPMComparisonResult

@dataclass
class MPMComparisonResult:
    metric_name: str
    framework_value: float
    mpm_value: float
    difference: float
    relative_error: float  # percentage
    correlation: float
    is_valid: bool
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### Fields

- `metric_name` (str): Name of the compared metric
- `framework_value` (float): Framework metric value
- `mpm_value` (float): MPM system metric value
- `difference` (float): Absolute difference
- `relative_error` (float): Relative error percentage
- `correlation` (float): Correlation coefficient (if applicable)
- `is_valid` (bool): Whether comparison is within acceptable thresholds
- `metadata` (Dict[str, Any]): Additional metadata

### Methods

#### `to_dict() -> Dict[str, Any]`

Convert result to dictionary.

**Returns**: Dictionary representation of result

---

## MPMComparisonEngine

Engine for comparing framework outputs with MPM system outputs.

```python
from am_qadf.validation import MPMComparisonEngine

engine = MPMComparisonEngine(
    correlation_threshold: float = 0.85,
    max_relative_error: float = 0.1
)
```

### Methods

#### `__init__(correlation_threshold: float = 0.85, max_relative_error: float = 0.1)`

Initialize MPM comparison engine.

**Parameters**:
- `correlation_threshold` (float): Minimum correlation coefficient for validity (default: 0.85)
- `max_relative_error` (float): Maximum relative error percentage for validity (default: 0.1 = 10%)

#### `compare_metric(metric_name: str, framework_value: float, mpm_value: float) -> MPMComparisonResult`

Compare a single metric.

**Parameters**:
- `metric_name` (str): Name of the metric
- `framework_value` (float): Framework metric value
- `mpm_value` (float): MPM system metric value

**Returns**: `MPMComparisonResult`

#### `compare_quality_metrics(framework_metrics: Dict[str, float], mpm_metrics: Dict[str, float]) -> Dict[str, MPMComparisonResult]`

Compare multiple quality metrics.

**Parameters**:
- `framework_metrics` (Dict[str, float]): Dictionary of framework metric names to values
- `mpm_metrics` (Dict[str, float]): Dictionary of MPM metric names to values

**Returns**: Dictionary mapping metric names to `MPMComparisonResult`

#### `compare_arrays(framework_array: np.ndarray, mpm_array: np.ndarray, metric_name: str = "signal") -> MPMComparisonResult`

Compare two arrays (e.g., signal arrays).

**Parameters**:
- `framework_array` (np.ndarray): Framework array
- `mpm_array` (np.ndarray): MPM system array
- `metric_name` (str): Name for the comparison (default: "signal")

**Returns**: `MPMComparisonResult`

#### `calculate_correlation(array1: np.ndarray, array2: np.ndarray, method: str = "pearson") -> float`

Calculate correlation between two arrays.

**Parameters**:
- `array1` (np.ndarray): First array
- `array2` (np.ndarray): Second array
- `method` (str): Correlation method - "pearson", "spearman", or "kendall" (default: "pearson")

**Returns**: Correlation coefficient (float)

#### `compare_all_metrics(framework_data: Any, mpm_data: Any, metrics: Optional[List[str]] = None) -> Dict[str, Optional[MPMComparisonResult]]`

Compare all available metrics between framework and MPM data.

**Parameters**:
- `framework_data` (Any): Framework data (dict, array, or object)
- `mpm_data` (Any): MPM system data (dict, array, or object)
- `metrics` (Optional[List[str]]): Optional list of specific metrics to compare. If None, compares all available.

**Returns**: Dictionary mapping metric names to `MPMComparisonResult` (or None if comparison failed)

---

## AccuracyValidationResult

Result dataclass for accuracy validation operations.

```python
from am_qadf.validation import AccuracyValidationResult

@dataclass
class AccuracyValidationResult:
    signal_name: str
    rmse: float  # Root Mean Square Error
    mae: float  # Mean Absolute Error
    r2_score: float
    max_error: float
    within_tolerance: bool
    ground_truth_size: int
    validated_points: int
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### Fields

- `signal_name` (str): Name of the validated signal/quantity
- `rmse` (float): Root Mean Square Error
- `mae` (float): Mean Absolute Error
- `r2_score` (float): Coefficient of determination (R²)
- `max_error` (float): Maximum absolute error
- `within_tolerance` (bool): Whether errors are within acceptable tolerance
- `ground_truth_size` (int): Size of ground truth data
- `validated_points` (int): Number of points successfully validated
- `metadata` (Dict[str, Any]): Additional metadata

### Methods

#### `to_dict() -> Dict[str, Any]`

Convert result to dictionary.

**Returns**: Dictionary representation of result

---

## AccuracyValidator

Validator for accuracy validation against ground truth data.

```python
from am_qadf.validation import AccuracyValidator

validator = AccuracyValidator(
    max_acceptable_error: float = 0.1,
    tolerance_percent: float = 5.0
)
```

### Methods

#### `__init__(max_acceptable_error: float = 0.1, tolerance_percent: float = 5.0)`

Initialize accuracy validator.

**Parameters**:
- `max_acceptable_error` (float): Maximum acceptable absolute error (default: 0.1)
- `tolerance_percent` (float): Relative tolerance percentage (default: 5.0%)

#### `validate_signal_mapping(framework_signal: np.ndarray, ground_truth: np.ndarray, signal_name: str) -> AccuracyValidationResult`

Validate signal mapping against ground truth.

**Parameters**:
- `framework_signal` (np.ndarray): Framework-generated signal array
- `ground_truth` (np.ndarray): Ground truth signal array
- `signal_name` (str): Name of the signal

**Returns**: `AccuracyValidationResult`

#### `validate_spatial_alignment(framework_coords: np.ndarray, ground_truth_coords: np.ndarray) -> AccuracyValidationResult`

Validate spatial alignment/coordinate transformation.

**Parameters**:
- `framework_coords` (np.ndarray): Framework coordinate array (N×3)
- `ground_truth_coords` (np.ndarray): Ground truth coordinate array (N×3)

**Returns**: `AccuracyValidationResult`

#### `validate_temporal_alignment(framework_times: np.ndarray, ground_truth_times: np.ndarray) -> AccuracyValidationResult`

Validate temporal alignment/time synchronization.

**Parameters**:
- `framework_times` (np.ndarray): Framework time array
- `ground_truth_times` (np.ndarray): Ground truth time array

**Returns**: `AccuracyValidationResult`

#### `validate_quality_metrics(framework_metrics: Dict[str, float], ground_truth_metrics: Dict[str, float]) -> AccuracyValidationResult`

Validate quality metrics against ground truth.

**Parameters**:
- `framework_metrics` (Dict[str, float]): Framework quality metrics
- `ground_truth_metrics` (Dict[str, float]): Ground truth quality metrics

**Returns**: `AccuracyValidationResult`

#### `calculate_rmse(predicted: np.ndarray, actual: np.ndarray) -> float`

Calculate Root Mean Square Error.

**Parameters**:
- `predicted` (np.ndarray): Predicted/framework values
- `actual` (np.ndarray): Actual/ground truth values

**Returns**: RMSE value (float)

#### `calculate_mae(predicted: np.ndarray, actual: np.ndarray) -> float`

Calculate Mean Absolute Error.

**Parameters**:
- `predicted` (np.ndarray): Predicted/framework values
- `actual` (np.ndarray): Actual/ground truth values

**Returns**: MAE value (float)

#### `calculate_r2_score(predicted: np.ndarray, actual: np.ndarray) -> float`

Calculate R² (coefficient of determination).

**Parameters**:
- `predicted` (np.ndarray): Predicted/framework values
- `actual` (np.ndarray): Actual/ground truth values

**Returns**: R² score (float, range: -∞ to 1)

#### `validate_within_tolerance(predicted: np.ndarray, actual: np.ndarray, tolerance: Optional[float] = None) -> bool`

Check if predictions are within tolerance.

**Parameters**:
- `predicted` (np.ndarray): Predicted/framework values
- `actual` (np.ndarray): Actual/ground truth values
- `tolerance` (Optional[float]): Optional tolerance value. If None, uses `max_acceptable_error` or `tolerance_percent`.

**Returns**: True if within tolerance, False otherwise

---

## StatisticalValidationResult

Result dataclass for statistical validation operations.

```python
from am_qadf.validation import StatisticalValidationResult

@dataclass
class StatisticalValidationResult:
    test_name: str
    null_hypothesis: str
    alternative_hypothesis: str
    test_statistic: float
    p_value: float
    significance_level: float
    is_significant: bool
    conclusion: str
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### Fields

- `test_name` (str): Name of the statistical test
- `null_hypothesis` (str): Description of null hypothesis
- `alternative_hypothesis` (str): Description of alternative hypothesis
- `test_statistic` (float): Calculated test statistic
- `p_value` (float): P-value of the test
- `significance_level` (float): Significance level (α) used
- `is_significant` (bool): Whether result is statistically significant
- `conclusion` (str): Human-readable conclusion
- `metadata` (Dict[str, Any]): Additional metadata (degrees of freedom, etc.)

### Methods

#### `to_dict() -> Dict[str, Any]`

Convert result to dictionary.

**Returns**: Dictionary representation of result

---

## StatisticalValidator

Validator for statistical hypothesis testing.

```python
from am_qadf.validation import StatisticalValidator

validator = StatisticalValidator(
    significance_level: float = 0.05,
    confidence_level: float = 0.95
)
```

### Methods

#### `__init__(significance_level: float = 0.05, confidence_level: float = 0.95)`

Initialize statistical validator.

**Parameters**:
- `significance_level` (float): Significance level (α) for hypothesis tests (default: 0.05)
- `confidence_level` (float): Confidence level for confidence intervals (default: 0.95)

#### `t_test(sample1: np.ndarray, sample2: np.ndarray, alternative: str = "two-sided") -> StatisticalValidationResult`

Perform independent samples t-test.

**Parameters**:
- `sample1` (np.ndarray): First sample
- `sample2` (np.ndarray): Second sample
- `alternative` (str): Alternative hypothesis - "two-sided", "less", or "greater" (default: "two-sided")

**Returns**: `StatisticalValidationResult`

#### `mann_whitney_u_test(sample1: np.ndarray, sample2: np.ndarray, alternative: str = "two-sided") -> StatisticalValidationResult`

Perform Mann-Whitney U test (non-parametric alternative to t-test).

**Parameters**:
- `sample1` (np.ndarray): First sample
- `sample2` (np.ndarray): Second sample
- `alternative` (str): Alternative hypothesis - "two-sided", "less", or "greater" (default: "two-sided")

**Returns**: `StatisticalValidationResult`

#### `correlation_test(x: np.ndarray, y: np.ndarray, method: str = "pearson") -> StatisticalValidationResult`

Test for significant correlation between two variables.

**Parameters**:
- `x` (np.ndarray): First variable
- `y` (np.ndarray): Second variable
- `method` (str): Correlation method - "pearson" or "spearman" (default: "pearson")

**Returns**: `StatisticalValidationResult`

#### `chi_square_test(observed: np.ndarray, expected: Optional[np.ndarray] = None, ddof: int = 0) -> StatisticalValidationResult`

Perform chi-square test (goodness-of-fit or independence).

**Parameters**:
- `observed` (np.ndarray): Observed frequencies
- `expected` (Optional[np.ndarray]): Expected frequencies (for goodness-of-fit). If None, assumes independence test.
- `ddof` (int): Degrees of freedom adjustment (default: 0)

**Returns**: `StatisticalValidationResult`

#### `anova_test(groups: List[np.ndarray]) -> StatisticalValidationResult`

Perform one-way ANOVA test.

**Parameters**:
- `groups` (List[np.ndarray]): List of arrays, one per group

**Returns**: `StatisticalValidationResult`

#### `normality_test(sample: np.ndarray, method: str = "shapiro") -> StatisticalValidationResult`

Test if sample follows normal distribution.

**Parameters**:
- `sample` (np.ndarray): Sample to test
- `method` (str): Test method - "shapiro", "normaltest", or "ks" (default: "shapiro")

**Returns**: `StatisticalValidationResult`

#### `validate_improvement(baseline: np.ndarray, improved: np.ndarray, test_type: str = "t_test", alternative: str = "greater") -> StatisticalValidationResult`

Validate if improved version is significantly better than baseline.

**Parameters**:
- `baseline` (np.ndarray): Baseline sample values
- `improved` (np.ndarray): Improved sample values
- `test_type` (str): Test type - "t_test" or "mann_whitney" (default: "t_test")
- `alternative` (str): Alternative hypothesis - "greater" (improved > baseline), "less", or "two-sided" (default: "greater")

**Returns**: `StatisticalValidationResult`

---

## Related

- [Validation Module](../05-modules/validation.md) - Module documentation
- [Quality Module](../05-modules/quality.md) - Quality assessment (integrated with validation)
- [Analytics Module](../05-modules/analytics.md) - Statistical analysis

---

**Parent**: [API Reference](README.md)
