# Statistical Process Control (SPC) Module API Reference

## Overview

The Statistical Process Control (SPC) module provides comprehensive statistical process control capabilities for monitoring and controlling manufacturing processes in AM-QADF. It includes control chart generation, process capability analysis, multivariate SPC monitoring, control rule violation detection, baseline calculation, and adaptive control limits.

## SPCConfig

Configuration dataclass for SPC operations.

```python
from am_qadf.analytics.spc import SPCConfig

config = SPCConfig(
    control_limit_sigma: float = 3.0,  # Standard deviations for control limits
    subgroup_size: int = 5,  # Subgroup size for X-bar charts
    baseline_sample_size: int = 100,  # Minimum samples for baseline
    adaptive_limits: bool = False,  # Enable adaptive control limits
    update_frequency: Optional[int] = None,  # Samples between limit updates
    specification_limits: Optional[Tuple[float, float]] = None,  # USL, LSL
    target_value: Optional[float] = None,  # Target/center value
    enable_warnings: bool = True,  # Enable warning limits (2-sigma)
    warning_sigma: float = 2.0  # Warning limit multiplier
)
```

### Fields

- `control_limit_sigma` (float): Standard deviations for control limits (default: 3.0)
- `subgroup_size` (int): Subgroup size for X-bar charts (default: 5)
- `baseline_sample_size` (int): Minimum samples for baseline (default: 100)
- `adaptive_limits` (bool): Enable adaptive control limits (default: False)
- `update_frequency` (Optional[int]): Samples between limit updates (default: None)
- `specification_limits` (Optional[Tuple[float, float]]): Specification limits (USL, LSL) (default: None)
- `target_value` (Optional[float]): Target/center value (default: None)
- `enable_warnings` (bool): Enable warning limits (2-sigma) (default: True)
- `warning_sigma` (float): Warning limit multiplier (default: 2.0)

---

## SPCClient

Main client interface for all SPC operations.

```python
from am_qadf.analytics.spc import SPCClient, SPCConfig

client = SPCClient(config: Optional[SPCConfig] = None, mongo_client: Optional[Any] = None)
```

### Methods

#### `__init__(config: Optional[SPCConfig] = None, mongo_client: Optional[Any] = None)`

Initialize the SPC client.

**Parameters**:
- `config` (Optional[SPCConfig]): SPC configuration. If None, uses default config.
- `mongo_client` (Optional[Any]): Optional MongoDB client for data storage.

---

#### `create_control_chart(data: np.ndarray, chart_type: str = 'xbar', baseline: Optional[BaselineStatistics] = None, subgroup_size: Optional[int] = None) -> ControlChartResult`

Create a control chart for the given data.

**Parameters**:
- `data` (np.ndarray): Process data (1D array for individual/moving range, 2D array for X-bar/R/S)
- `chart_type` (str): Chart type - 'xbar', 'r', 's', 'individual', 'moving_range', 'xbar_r', 'xbar_s' (default: 'xbar')
- `baseline` (Optional[BaselineStatistics]): Optional baseline statistics. If None, calculated from data.
- `subgroup_size` (Optional[int]): Subgroup size (for X-bar charts). If None, uses config default.

**Returns**: `ControlChartResult` containing chart statistics and out-of-control points.

**Example**:
```python
import numpy as np
data = np.random.normal(10.0, 1.0, (20, 5))  # 20 subgroups of size 5
result = client.create_control_chart(data, chart_type='xbar')
print(f"UCL: {result.upper_control_limit:.3f}")
print(f"OOC points: {len(result.out_of_control_points)}")
```

---

#### `establish_baseline(data: np.ndarray, subgroup_size: Optional[int] = None, signal_name: Optional[str] = None, model_id: Optional[str] = None) -> BaselineStatistics`

Establish baseline statistics from historical data.

**Parameters**:
- `data` (np.ndarray): Historical process data (1D or 2D array)
- `subgroup_size` (Optional[int]): Subgroup size (for subgrouped data). If None, inferred from data.
- `signal_name` (Optional[str]): Optional signal name for storage
- `model_id` (Optional[str]): Optional model ID for storage

**Returns**: `BaselineStatistics` containing baseline mean, std, and other statistics.

**Example**:
```python
baseline_data = np.random.normal(10.0, 1.0, 100)
baseline = client.establish_baseline(baseline_data, subgroup_size=1)
print(f"Baseline: {baseline.mean:.3f} ± {baseline.std:.3f}")
```

---

#### `update_baseline_adaptive(baseline: BaselineStatistics, new_data: np.ndarray, method: str = 'exponential_smoothing', alpha: float = 0.3, signal_name: Optional[str] = None, model_id: Optional[str] = None) -> BaselineStatistics`

Update baseline statistics with new data using adaptive methods.

**Parameters**:
- `baseline` (BaselineStatistics): Current baseline statistics
- `new_data` (np.ndarray): New process data
- `method` (str): Update method - 'exponential_smoothing' or 'cumulative' (default: 'exponential_smoothing')
- `alpha` (float): Smoothing parameter for exponential smoothing (default: 0.3)
- `signal_name` (Optional[str]): Optional signal name for storage
- `model_id` (Optional[str]): Optional model ID for storage

**Returns**: Updated `BaselineStatistics`.

**Example**:
```python
new_data = np.random.normal(10.1, 1.0, 50)
updated_baseline = client.update_baseline_adaptive(
    baseline, new_data, method='exponential_smoothing', alpha=0.3
)
```

---

#### `detect_rule_violations(chart_result: ControlChartResult, rules: str = 'both') -> List[ControlRuleViolation]`

Detect control rule violations in a control chart.

**Parameters**:
- `chart_result` (ControlChartResult): Control chart result to analyze
- `rules` (str): Rule set - 'western_electric', 'nelson', or 'both' (default: 'both')

**Returns**: List of `ControlRuleViolation` objects.

**Example**:
```python
violations = client.detect_rule_violations(chart_result, rules='both')
for violation in violations:
    print(f"{violation.rule_name}: {violation.description} (Severity: {violation.severity})")
```

---

#### `analyze_process_capability(data: np.ndarray, specification_limits: Tuple[float, float], target_value: Optional[float] = None, subgroup_size: Optional[int] = None) -> ProcessCapabilityResult`

Analyze process capability relative to specification limits.

**Parameters**:
- `data` (np.ndarray): Process data (1D or 2D array)
- `specification_limits` (Tuple[float, float]): Specification limits (LSL, USL)
- `target_value` (Optional[float]): Optional target value
- `subgroup_size` (Optional[int]): Optional subgroup size (for within-subgroup variation)

**Returns**: `ProcessCapabilityResult` containing Cp, Cpk, Pp, Ppk indices and capability rating.

**Example**:
```python
data = np.random.normal(10.0, 1.0, 200)
result = client.analyze_process_capability(
    data,
    specification_limits=(6.0, 14.0),
    target_value=10.0
)
print(f"Cp: {result.cp:.3f}, Cpk: {result.cpk:.3f}")
print(f"Rating: {result.capability_rating}")
```

---

#### `create_multivariate_chart(data: np.ndarray, method: str = 'hotelling_t2', baseline_mean: Optional[np.ndarray] = None, baseline_covariance: Optional[np.ndarray] = None, n_components: Optional[int] = None) -> MultivariateSPCResult`

Create a multivariate control chart for multiple correlated variables.

**Parameters**:
- `data` (np.ndarray): Multivariate process data (shape: [n_samples, n_variables])
- `method` (str): Method - 'hotelling_t2' or 'pca' (default: 'hotelling_t2')
- `baseline_mean` (Optional[np.ndarray]): Optional baseline mean vector. If None, calculated from data.
- `baseline_covariance` (Optional[np.ndarray]): Optional baseline covariance matrix. If None, calculated from data.
- `n_components` (Optional[int]): Number of PCA components (for PCA method). If None, selected automatically.

**Returns**: `MultivariateSPCResult` containing T² statistics, control limits, and out-of-control points.

**Example**:
```python
multivariate_data = np.random.multivariate_normal(
    mean=[0, 0, 0],
    cov=[[1, 0.7, 0.5], [0.7, 1, 0.6], [0.5, 0.6, 1]],
    size=100
)
result = client.create_multivariate_chart(multivariate_data, method='hotelling_t2')
print(f"T² UCL: {result.ucl_t2:.3f}")
print(f"OOC points: {len(result.out_of_control_points)}")
```

---

#### `comprehensive_spc_analysis(data: np.ndarray, specification_limits: Optional[Tuple[float, float]] = None, target_value: Optional[float] = None, chart_types: Optional[List[str]] = None, detect_rules: str = 'both', include_capability: bool = True, include_multivariate: bool = False) -> Dict[str, Any]`

Perform comprehensive SPC analysis including baseline, control charts, capability, and rules.

**Parameters**:
- `data` (np.ndarray): Process data
- `specification_limits` (Optional[Tuple[float, float]]): Optional specification limits (LSL, USL)
- `target_value` (Optional[float]): Optional target value
- `chart_types` (Optional[List[str]]): List of chart types to generate. If None, uses ['xbar_r'] or ['individual'].
- `detect_rules` (str): Rule set for detection - 'western_electric', 'nelson', 'both', or 'none' (default: 'both')
- `include_capability` (bool): Include capability analysis (default: True)
- `include_multivariate` (bool): Include multivariate analysis (default: False)

**Returns**: Dictionary containing:
- `'baseline'`: BaselineStatistics
- `'control_charts'`: Dict[str, ControlChartResult] (by chart type)
- `'capability'`: ProcessCapabilityResult (if include_capability=True)
- `'rule_violations'`: List[ControlRuleViolation] (if detect_rules != 'none')
- `'multivariate'`: MultivariateSPCResult (if include_multivariate=True)

**Example**:
```python
data = np.random.normal(10.0, 1.0, (25, 5))  # 25 subgroups of 5
results = client.comprehensive_spc_analysis(
    data,
    specification_limits=(6.0, 14.0),
    target_value=10.0,
    chart_types=['xbar_r'],
    detect_rules='both'
)
print(f"Baseline: {results['baseline'].mean:.3f}")
print(f"Capability Cpk: {results['capability'].cpk:.3f}")
print(f"Rule violations: {len(results['rule_violations'])}")
```

---

## ControlChartResult

Result dataclass for control chart analysis.

```python
from am_qadf.analytics.spc import ControlChartResult

@dataclass
class ControlChartResult:
    chart_type: str  # 'xbar', 'r', 's', 'individual', 'moving_range'
    center_line: float  # Center line (CL)
    upper_control_limit: float  # UCL
    lower_control_limit: float  # LCL
    upper_warning_limit: Optional[float] = None  # UWL
    lower_warning_limit: Optional[float] = None  # LWL
    sample_values: np.ndarray  # Sample values
    sample_indices: np.ndarray  # Sample indices/timestamps
    out_of_control_points: List[int]  # Indices of OOC points
    rule_violations: Dict[str, List[int]]  # Rule violations by rule name
    baseline_stats: Dict[str, float]  # Baseline statistics
    metadata: Dict[str, Any]  # Additional metadata
```

### Fields

- `chart_type` (str): Chart type identifier
- `center_line` (float): Center line value
- `upper_control_limit` (float): Upper control limit (UCL)
- `lower_control_limit` (float): Lower control limit (LCL)
- `upper_warning_limit` (Optional[float]): Upper warning limit (UWL) if enabled
- `lower_warning_limit` (Optional[float]): Lower warning limit (LWL) if enabled
- `sample_values` (np.ndarray): Sample values plotted on chart
- `sample_indices` (np.ndarray): Sample indices or timestamps
- `out_of_control_points` (List[int]): Indices of out-of-control points
- `rule_violations` (Dict[str, List[int]]): Rule violations grouped by rule name
- `baseline_stats` (Dict[str, float]): Baseline statistics used for chart
- `metadata` (Dict[str, Any]): Additional metadata

---

## ProcessCapabilityResult

Result dataclass for process capability analysis.

```python
from am_qadf.analytics.spc import ProcessCapabilityResult

@dataclass
class ProcessCapabilityResult:
    cp: float  # Process capability index
    cpk: float  # Process capability index (centered)
    pp: float  # Process performance index
    ppk: float  # Process performance index (centered)
    cpu: float  # Upper capability index
    cpl: float  # Lower capability index
    specification_limits: Tuple[float, float]  # USL, LSL
    target_value: Optional[float]  # Target value
    process_mean: float  # Actual process mean
    process_std: float  # Actual process standard deviation
    within_subgroup_std: Optional[float] = None  # Within-subgroup std (for Cp)
    overall_std: Optional[float] = None  # Overall std (for Pp)
    capability_rating: str = "Unknown"  # 'Excellent', 'Adequate', 'Marginal', 'Inadequate'
    metadata: Dict[str, Any]  # Additional metadata
```

### Fields

- `cp` (float): Process capability index (short-term, within-subgroup variation)
- `cpk` (float): Process capability index accounting for centering
- `pp` (float): Process performance index (long-term, overall variation)
- `ppk` (float): Process performance index accounting for centering
- `cpu` (float): Upper capability index: (USL - mean) / (3σ)
- `cpl` (float): Lower capability index: (mean - LSL) / (3σ)
- `specification_limits` (Tuple[float, float]): Specification limits (USL, LSL)
- `target_value` (Optional[float]): Target value if specified
- `process_mean` (float): Actual process mean
- `process_std` (float): Actual process standard deviation
- `within_subgroup_std` (Optional[float]): Within-subgroup standard deviation (for Cp)
- `overall_std` (Optional[float]): Overall standard deviation (for Pp)
- `capability_rating` (str): Capability rating - 'Excellent' (≥1.67), 'Adequate' (≥1.33), 'Marginal' (≥1.0), 'Inadequate' (<1.0)
- `metadata` (Dict[str, Any]): Additional metadata

---

## MultivariateSPCResult

Result dataclass for multivariate SPC analysis.

```python
from am_qadf.analytics.spc import MultivariateSPCResult

@dataclass
class MultivariateSPCResult:
    hotelling_t2: np.ndarray  # Hotelling T² statistics
    ucl_t2: float  # Upper control limit for T²
    control_limits: Dict[str, float]  # Control limits for each component
    out_of_control_points: List[int]  # Indices of OOC points
    baseline_mean: np.ndarray  # Baseline mean vector
    baseline_covariance: np.ndarray  # Baseline covariance matrix
    principal_components: Optional[np.ndarray] = None  # PCA components
    explained_variance: Optional[np.ndarray] = None  # Explained variance ratio
    contribution_analysis: Optional[Dict[int, List[str]]] = None  # Variable contributions for OOC points
    metadata: Dict[str, Any]  # Additional metadata
```

### Fields

- `hotelling_t2` (np.ndarray): Hotelling T² statistics for each sample
- `ucl_t2` (float): Upper control limit for T² statistic
- `control_limits` (Dict[str, float]): Control limits for each component/variable
- `out_of_control_points` (List[int]): Indices of out-of-control points
- `baseline_mean` (np.ndarray): Baseline mean vector
- `baseline_covariance` (np.ndarray): Baseline covariance matrix
- `principal_components` (Optional[np.ndarray]): PCA principal components (for PCA method)
- `explained_variance` (Optional[np.ndarray]): Explained variance ratio for each component (for PCA method)
- `contribution_analysis` (Optional[Dict[int, List[str]]]): Variable contributions for OOC points
- `metadata` (Dict[str, Any]): Additional metadata

---

## BaselineStatistics

Baseline statistics dataclass for SPC.

```python
from am_qadf.analytics.spc import BaselineStatistics

@dataclass
class BaselineStatistics:
    mean: float
    std: float
    median: float
    min: float
    max: float
    range: float
    sample_size: int
    subgroup_size: int
    within_subgroup_std: Optional[float] = None
    between_subgroup_std: Optional[float] = None
    overall_std: Optional[float] = None
    calculated_at: datetime
    metadata: Dict[str, Any]
```

### Fields

- `mean` (float): Baseline mean
- `std` (float): Baseline standard deviation
- `median` (float): Baseline median
- `min` (float): Minimum value
- `max` (float): Maximum value
- `range` (float): Range (max - min)
- `sample_size` (int): Number of samples used
- `subgroup_size` (int): Subgroup size (1 for individual charts)
- `within_subgroup_std` (Optional[float]): Within-subgroup standard deviation (for subgrouped data)
- `between_subgroup_std` (Optional[float]): Between-subgroup standard deviation (for subgrouped data)
- `overall_std` (Optional[float]): Overall standard deviation
- `calculated_at` (datetime): Timestamp when baseline was calculated
- `metadata` (Dict[str, Any]): Additional metadata

---

## ControlRuleViolation

Control rule violation dataclass.

```python
from am_qadf.analytics.spc import ControlRuleViolation

@dataclass
class ControlRuleViolation:
    rule_name: str  # 'western_electric_1', 'nelson_1', etc.
    violation_type: str  # 'out_of_control', 'trend', 'pattern', etc.
    affected_points: List[int]  # Indices of violating points
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str  # Human-readable description
    metadata: Dict[str, Any]  # Additional metadata
```

### Fields

- `rule_name` (str): Rule identifier (e.g., 'western_electric_1', 'nelson_1')
- `violation_type` (str): Type of violation ('out_of_control', 'trend', 'pattern', etc.)
- `affected_points` (List[int]): Indices of points violating the rule
- `severity` (str): Severity level ('low', 'medium', 'high', 'critical')
- `description` (str): Human-readable description of the violation
- `metadata` (Dict[str, Any]): Additional metadata

---

## ControlChartGenerator

Standalone control chart generator class.

```python
from am_qadf.analytics.spc import ControlChartGenerator

generator = ControlChartGenerator()
```

### Methods

#### `generate_xbar_chart(subgroups: np.ndarray, sigma: float = 3.0, baseline: Optional[BaselineStatistics] = None) -> ControlChartResult`

Generate X-bar control chart.

**Parameters**:
- `subgroups` (np.ndarray): Subgrouped data (shape: [n_subgroups, subgroup_size])
- `sigma` (float): Control limit multiplier (default: 3.0)
- `baseline` (Optional[BaselineStatistics]): Optional baseline statistics

**Returns**: `ControlChartResult`

---

#### `generate_r_chart(subgroups: np.ndarray, sigma: float = 3.0, baseline: Optional[BaselineStatistics] = None) -> ControlChartResult`

Generate R (range) control chart.

**Parameters**:
- `subgroups` (np.ndarray): Subgrouped data (shape: [n_subgroups, subgroup_size])
- `sigma` (float): Control limit multiplier (default: 3.0)
- `baseline` (Optional[BaselineStatistics]): Optional baseline statistics

**Returns**: `ControlChartResult`

---

#### `generate_individual_chart(data: np.ndarray, sigma: float = 3.0, baseline: Optional[BaselineStatistics] = None) -> ControlChartResult`

Generate Individual control chart.

**Parameters**:
- `data` (np.ndarray): Individual measurements (1D array)
- `sigma` (float): Control limit multiplier (default: 3.0)
- `baseline` (Optional[BaselineStatistics]): Optional baseline statistics

**Returns**: `ControlChartResult`

---

## ProcessCapabilityAnalyzer

Standalone process capability analyzer class.

```python
from am_qadf.analytics.spc import ProcessCapabilityAnalyzer

analyzer = ProcessCapabilityAnalyzer()
```

### Methods

#### `calculate_capability(data: np.ndarray, specification_limits: Tuple[float, float], target_value: Optional[float] = None, subgroup_size: Optional[int] = None) -> ProcessCapabilityResult`

Calculate process capability indices.

**Parameters**:
- `data` (np.ndarray): Process data (1D or 2D array)
- `specification_limits` (Tuple[float, float]): Specification limits (LSL, USL)
- `target_value` (Optional[float]): Optional target value
- `subgroup_size` (Optional[int]): Optional subgroup size

**Returns**: `ProcessCapabilityResult`

---

## MultivariateSPCAnalyzer

Standalone multivariate SPC analyzer class.

```python
from am_qadf.analytics.spc import MultivariateSPCAnalyzer

analyzer = MultivariateSPCAnalyzer()
```

### Methods

#### `hotelling_t2_chart(data: np.ndarray, baseline_mean: Optional[np.ndarray] = None, baseline_covariance: Optional[np.ndarray] = None, alpha: float = 0.05) -> MultivariateSPCResult`

Create Hotelling T² control chart.

**Parameters**:
- `data` (np.ndarray): Multivariate data (shape: [n_samples, n_variables])
- `baseline_mean` (Optional[np.ndarray]): Optional baseline mean vector
- `baseline_covariance` (Optional[np.ndarray]): Optional baseline covariance matrix
- `alpha` (float): Significance level for control limit (default: 0.05)

**Returns**: `MultivariateSPCResult`

---

#### `pca_based_spc(data: np.ndarray, n_components: Optional[int] = None, explained_variance_threshold: float = 0.95) -> MultivariateSPCResult`

Create PCA-based multivariate SPC chart.

**Parameters**:
- `data` (np.ndarray): Multivariate data (shape: [n_samples, n_variables])
- `n_components` (Optional[int]): Number of PCA components. If None, selected based on explained variance.
- `explained_variance_threshold` (float): Minimum explained variance threshold (default: 0.95)

**Returns**: `MultivariateSPCResult`

**Note**: Requires scikit-learn for PCA functionality.

---

## ControlRuleDetector

Standalone control rule detector class.

```python
from am_qadf.analytics.spc import ControlRuleDetector

detector = ControlRuleDetector()
```

### Methods

#### `detect_western_electric_rules(chart_result: ControlChartResult) -> List[ControlRuleViolation]`

Detect Western Electric rule violations.

**Parameters**:
- `chart_result` (ControlChartResult): Control chart result to analyze

**Returns**: List of `ControlRuleViolation` objects

---

#### `detect_nelson_rules(chart_result: ControlChartResult) -> List[ControlRuleViolation]`

Detect Nelson rule violations.

**Parameters**:
- `chart_result` (ControlChartResult): Control chart result to analyze

**Returns**: List of `ControlRuleViolation` objects

---

## BaselineCalculator

Standalone baseline calculator class.

```python
from am_qadf.analytics.spc import BaselineCalculator

calculator = BaselineCalculator()
```

### Methods

#### `calculate_baseline(data: np.ndarray, subgroup_size: Optional[int] = None) -> BaselineStatistics`

Calculate baseline statistics from data.

**Parameters**:
- `data` (np.ndarray): Process data (1D or 2D array)
- `subgroup_size` (Optional[int]): Optional subgroup size

**Returns**: `BaselineStatistics`

---

## AdaptiveLimitsCalculator

Standalone adaptive limits calculator class.

```python
from am_qadf.analytics.spc import AdaptiveLimitsCalculator

calculator = AdaptiveLimitsCalculator()
```

### Methods

#### `calculate_adaptive_limits(baseline: BaselineStatistics, new_data: np.ndarray, method: str = 'exponential_smoothing', alpha: float = 0.3) -> BaselineStatistics`

Calculate adaptive control limits.

**Parameters**:
- `baseline` (BaselineStatistics): Current baseline statistics
- `new_data` (np.ndarray): New process data
- `method` (str): Update method - 'exponential_smoothing' or 'cumulative' (default: 'exponential_smoothing')
- `alpha` (float): Smoothing parameter for exponential smoothing (default: 0.3)

**Returns**: Updated `BaselineStatistics`

---

## SPCStorage

Storage interface for SPC data in MongoDB.

```python
from am_qadf.analytics.spc import SPCStorage

storage = SPCStorage(mongo_client: Any)
```

### Methods

#### `save_baseline(model_id: str, signal_name: str, baseline: BaselineStatistics, metadata: Optional[Dict] = None) -> str`

Save baseline statistics to MongoDB.

**Parameters**:
- `model_id` (str): Model ID
- `signal_name` (str): Signal name
- `baseline` (BaselineStatistics): Baseline statistics to save
- `metadata` (Optional[Dict]): Optional additional metadata

**Returns**: Document ID (str)

---

#### `load_baseline(model_id: str, signal_name: str, baseline_id: Optional[str] = None) -> Optional[BaselineStatistics]`

Load baseline statistics from MongoDB.

**Parameters**:
- `model_id` (str): Model ID
- `signal_name` (str): Signal name
- `baseline_id` (Optional[str]): Optional specific baseline ID. If None, loads most recent.

**Returns**: `BaselineStatistics` if found, None otherwise

---

#### `query_spc_history(model_id: Optional[str] = None, signal_name: Optional[str] = None, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None, spc_type: Optional[str] = None) -> List[Dict[str, Any]]`

Query SPC history from MongoDB.

**Parameters**:
- `model_id` (Optional[str]): Optional model ID filter
- `signal_name` (Optional[str]): Optional signal name filter
- `start_time` (Optional[datetime]): Optional start time filter
- `end_time` (Optional[datetime]): Optional end time filter
- `spc_type` (Optional[str]): Optional SPC type filter ('baseline', 'control_chart', 'capability', 'multivariate')

**Returns**: List of SPC history documents

---

## Related Documentation

- **[SPC Module Documentation](../05-modules/spc.md)** - Complete module documentation
- **[Notebook 25: Statistical Process Control](../../Notebook/04-notebooks/25-statistical-process-control.md)** - Interactive tutorial
- **[Implementation Plan](../../../implementation_plans/SPC_MODULE_IMPLEMENTATION.md)** - Implementation details

---

**Related**: [Quality Assessment API](quality-api.md) | [Analytics API](analytics-api.md) | [Anomaly Detection API](anomaly-detection-api.md)