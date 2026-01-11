# Notebook 25: Statistical Process Control

**File**: `25_Statistical_Process_Control.ipynb`  
**Category**: Advanced Topics / Quality Control  
**Duration**: 90-120 minutes

## Purpose

This notebook teaches you how to apply Statistical Process Control (SPC) methods to monitor and control manufacturing processes in AM-QADF. You'll learn SPC fundamentals, control chart generation, process capability analysis, multivariate SPC, control rule detection, and baseline calculation using a unified interactive interface with real-time progress tracking and detailed logging.

## Learning Objectives

By the end of this notebook, you will:

- ✅ Understand SPC concepts and their importance in quality control
- ✅ Generate various control charts (X-bar, R, S, Individual, Moving Range)
- ✅ Analyze process capability using indices (Cp, Cpk, Pp, Ppk)
- ✅ Perform multivariate SPC analysis (Hotelling T², PCA-based)
- ✅ Detect control rule violations (Western Electric, Nelson rules)
- ✅ Establish and update process baselines with adaptive limits
- ✅ Monitor quality metrics using SPC control charts
- ✅ Generate comprehensive SPC reports with visualizations
- ✅ Interpret SPC results and make process improvement decisions
- ✅ Monitor SPC analysis progress with real-time status and logs

## Topics Covered

### Control Charts

- **X-bar Charts**: Monitor process mean with subgrouped data
- **R Charts**: Monitor process variability (range)
- **S Charts**: Monitor process variability (standard deviation)
- **Individual Charts**: Monitor individual measurements
- **Moving Range Charts**: Monitor variability between consecutive measurements
- **Combined Charts**: X-bar & R, X-bar & S pairs

### Process Capability

- **Cp**: Process capability index (short-term, within-subgroup variation)
- **Cpk**: Process capability index accounting for centering
- **Pp**: Process performance index (long-term, overall variation)
- **Ppk**: Process performance index accounting for centering
- **Capability Rating**: Automatic classification (Excellent ≥ 1.67, Adequate ≥ 1.33, Marginal ≥ 1.0, Inadequate < 1.0)
- **PPM Estimation**: Estimate parts per million outside specifications

### Multivariate SPC

- **Hotelling T²**: Multivariate control chart for correlated variables
- **PCA-based SPC**: Dimensionality reduction for high-dimensional monitoring
- **Contribution Analysis**: Identify variables contributing to out-of-control conditions
- **Outlier Detection**: Detect multivariate outliers

### Control Rules

**Western Electric Rules**:
1. One point beyond 3σ control limits
2. Two of three consecutive points beyond 2σ warning limits (same side)
3. Four of five consecutive points beyond 1σ (same side)
4. Eight consecutive points on same side of center line
5. Six consecutive points increasing or decreasing
6. Fourteen consecutive points alternating up and down
7. Fifteen consecutive points within 1σ (both sides of center)
8. Eight consecutive points beyond 1σ (either side of center)

**Nelson Rules**:
1. One point beyond 3σ
2. Nine points in a row on same side of center line
3. Six points in a row steadily increasing or decreasing
4. Fourteen points in a row alternating up and down
5. Two of three points beyond 2σ (same side)
6. Four of five points beyond 1σ (same side)
7. Fifteen points in a row within 1σ (both sides)
8. Eight points in a row beyond 1σ (either side)

### Baseline Management

- **Baseline Calculation**: Calculate from historical data (minimum 30-100 samples recommended)
- **Control Limit Calculation**: Calculate limits for all chart types using appropriate constants
- **Baseline Updates**: Update baselines with new data using:
  - Exponential smoothing (weights recent data more)
  - Cumulative updates (weights all data equally)
- **Adaptive Limits**: Dynamically update limits when process changes are detected

### Storage and History

- **Baseline Storage**: Save baselines with model/signal identification
- **Chart History**: Store control chart results with timestamps
- **Capability History**: Track capability over time
- **Query Interface**: Query SPC history by time range, type, model, signal

## Interactive Widgets

### Top Panel

- **SPC Analysis Type**: Radio buttons
  - Control Charts
  - Process Capability
  - Multivariate SPC
  - Control Rules
  - Baseline Calculation
  - Comprehensive Analysis
- **Data Source**: Radio buttons (Demo Data / MongoDB)
- **Execute SPC Analysis**: Button to execute analysis
- **Export Report**: Button to export results

### Left Panel (Configuration Accordion)

#### Control Charts Configuration
- **Chart Type**: Dropdown (X-bar, R, S, Individual, Moving Range, X-bar & R, X-bar & S)
- **Subgroup Size**: IntSlider (2-20, default: 5)
- **Control Limit (σ)**: FloatSlider (2.0-4.0, default: 3.0)
- **Enable Warning Limits**: Checkbox (default: True)
- **Sample Size**: IntSlider (20-500, default: 100)

#### Process Capability Configuration
- **USL**: FloatText (default: 14.0)
- **LSL**: FloatText (default: 6.0)
- **Target**: FloatText (default: 10.0)
- **Sample Size**: IntSlider (30-1000, default: 200)

#### Multivariate SPC Configuration
- **Method**: Radio buttons (Hotelling T², PCA-based)
- **Number of Variables**: IntSlider (2-10, default: 3)
- **Sample Size**: IntSlider (30-500, default: 100)
- **PCA Components**: IntSlider (1-5, default: 2)

#### Control Rules Configuration
- **Rule Set**: Radio buttons (Western Electric, Nelson, Both)
- **Enable All Rules**: Checkbox (default: True)

#### Baseline Configuration
- **Baseline Sample Size**: IntSlider (30-500, default: 100)
- **Enable Adaptive Limits**: Checkbox (default: False)
- **Update Frequency**: IntSlider (10-200, default: 50)
- **Update Method**: Radio buttons (Exponential Smoothing, Cumulative)

### Center Panel

- **View Mode**: Radio buttons
  - Control Charts
  - Process Capability
  - Multivariate SPC
  - Rule Violations
  - Baseline Statistics
  - Comprehensive Report
- **Main Output**: Output widget for plots/text (height: 600px)

### Right Panel

- **Status Display**: Current operation status
- **Progress Bar**: Progress indicator (0-100%)
- **Results Summary**: Summary of analysis results
- **Key Metrics**: Key SPC metrics display
- **SPC Status**: Process control status (In Control / Out of Control)

### Bottom Panel

- **Status**: Overall status with elapsed time
- **SPC Analysis Logs**: Output widget for detailed execution logs (height: 200px)
  - Timestamped logs with emoji indicators:
    - ℹ️ Information messages
    - ✅ Success messages
    - ⚠️ Warning messages
    - ❌ Error messages (with full tracebacks)
- **Overall Progress Bar**: Progress indicator for comprehensive workflows
- **Warning Display**: Warning/error messages

## Key Features

### Real-Time Progress Tracking

- **Progress Bars**: Visual progress indicators (0-100%)
- **Status Updates**: Real-time status updates with elapsed time
- **Time Tracking**: Automatic tracking of execution time for all SPC operations

### Detailed Logging

- **Timestamped Logs**: All operations logged with timestamps
- **Log Levels**: Info, success, warning, and error messages
- **Error Tracebacks**: Full error tracebacks in logs for debugging
- **Log Retention**: Logs accumulated during session

### Comprehensive Analysis

- **Comprehensive Workflow**: All SPC steps in sequence:
  1. Baseline Calculation
  2. Control Charts
  3. Process Capability
  4. Control Rules
  5. Multivariate SPC
- **Integrated Results**: All results accessible from single interface
- **Report Generation**: Export comprehensive SPC reports

## Usage Examples

### Control Chart Generation

```python
# Select "Control Charts" in SPC Analysis Type
# Select "X-bar" in Chart Type
# Set Subgroup Size: 5
# Set Control Limit (σ): 3.0
# Set Sample Size: 100
# Click "Execute SPC Analysis"

# Results displayed:
# - Control chart plot with UCL, LCL, center line
# - Out-of-control points highlighted
# - Chart statistics in Results Summary
```

### Process Capability Analysis

```python
# Select "Process Capability" in SPC Analysis Type
# Set USL: 14.0
# Set LSL: 6.0
# Set Target: 10.0
# Set Sample Size: 200
# Click "Execute SPC Analysis"

# Results displayed:
# - Capability indices (Cp, Cpk, Pp, Ppk)
# - Capability rating (Excellent/Adequate/Marginal/Inadequate)
# - Process statistics (mean, std)
```

### Multivariate SPC

```python
# Select "Multivariate SPC" in SPC Analysis Type
# Select "Hotelling T²" in Method
# Set Number of Variables: 3
# Set Sample Size: 100
# Click "Execute SPC Analysis"

# Results displayed:
# - Hotelling T² chart
# - Out-of-control points
# - Contribution analysis (if applicable)
```

### Control Rule Detection

```python
# Select "Control Rules" in SPC Analysis Type
# Select "Both" in Rule Set
# Click "Execute SPC Analysis"

# Results displayed:
# - List of rule violations
# - Severity classification
# - Affected points
# - Human-readable descriptions
```

### Baseline Calculation

```python
# Select "Baseline Calculation" in SPC Analysis Type
# Set Baseline Sample Size: 100
# Select "Exponential Smoothing" in Update Method
# Click "Execute SPC Analysis"

# Results displayed:
# - Baseline statistics (mean, std, median, min, max)
# - Sample size
# - Calculated timestamp
```

### Comprehensive Analysis

```python
# Select "Comprehensive Analysis" in SPC Analysis Type
# Configure all sections as needed
# Click "Execute SPC Analysis"

# Results displayed:
# - All SPC steps executed in sequence
# - Progress tracked for each step
# - Comprehensive report generated
# - All results accessible from interface
```

## Best Practices

1. **Establish Baseline First**: Always establish a baseline from sufficient historical data (recommended: 30-100+ samples) before monitoring
2. **Choose Appropriate Chart Types**: 
   - Use X-bar/R or X-bar/S for subgrouped data (subgroup size ≥ 2)
   - Use Individual/Moving Range for individual measurements
3. **Monitor Capability Regularly**: Track Cpk and Ppk indices to ensure process meets specifications
4. **Investigate All Rule Violations**: All control rule violations should be investigated, even minor ones
5. **Use Adaptive Limits Carefully**: Enable adaptive limits only after understanding process stability
6. **Set Appropriate Limits**: Use 3-sigma limits for control, 2-sigma for warnings
7. **Track History**: Store SPC results in MongoDB for historical analysis and trending
8. **Monitor Logs**: Check the logs section for detailed execution information and any warnings
9. **Review Progress**: Use the status bar to monitor long-running SPC operations
10. **Use Comprehensive Analysis**: Use the comprehensive workflow for thorough process analysis

## Common Use Cases

### Process Monitoring

Monitor manufacturing processes in real-time using control charts:
- Generate X-bar/R charts for subgrouped data
- Monitor process mean and variability
- Detect out-of-control conditions
- Track process stability over time

### Capability Assessment

Assess process capability relative to specifications:
- Calculate Cp, Cpk, Pp, Ppk indices
- Rate process capability (Excellent/Adequate/Marginal/Inadequate)
- Identify capability improvement opportunities
- Track capability trends over time

### Multivariate Monitoring

Monitor multiple correlated process variables:
- Use Hotelling T² for correlated variables
- Apply PCA-based SPC for high-dimensional data
- Identify variables contributing to out-of-control signals
- Detect multivariate outliers

### Rule Violation Detection

Detect process issues through pattern recognition:
- Apply Western Electric rules for pattern detection
- Apply Nelson rules for trend detection
- Classify violations by severity
- Prioritize violations for investigation

### Baseline Management

Establish and maintain process baselines:
- Calculate baseline statistics from historical data
- Update baselines with new data using adaptive methods
- Track baseline changes over time
- Store baselines in MongoDB for persistence

## Troubleshooting

### Control Chart Not Displaying

- **Check Data Format**: Ensure data is properly formatted (1D for individual charts, 2D for subgrouped charts)
- **Check Sample Size**: Ensure sufficient samples (minimum 20-30)
- **Check Configuration**: Verify chart type matches data format

### Capability Analysis Fails

- **Check Specification Limits**: Ensure USL > LSL
- **Check Data Range**: Ensure data values are within reasonable range
- **Check Sample Size**: Ensure sufficient samples (minimum 30-50)

### Multivariate SPC Fails

- **Check Data Dimensions**: Ensure multivariate data has shape [n_samples, n_variables]
- **Check Correlation**: Ensure variables are correlated (for Hotelling T²)
- **Check PCA Availability**: Ensure scikit-learn is installed (for PCA method)

### Rule Violations Not Detected

- **Check Chart Type**: Ensure appropriate chart type for rule detection
- **Check Sample Size**: Ensure sufficient samples for pattern detection
- **Check Configuration**: Verify rule set is selected correctly

### Baseline Calculation Issues

- **Check Sample Size**: Ensure minimum sample size (30-100 recommended)
- **Check Data Quality**: Ensure data is free of outliers or anomalies
- **Check Subgroup Size**: Verify subgroup size matches data structure

## Related Notebooks

- **[07: Quality Assessment](07-quality.md)** - Quality assessment fundamentals
- **[08: Quality Dashboard](08-quality-dashboard.md)** - Quality monitoring dashboard
- **[09: Statistical Analysis](09-statistical.md)** - Statistical analysis methods
- **[14: Anomaly Detection Workflow](14-anomaly-workflow.md)** - Anomaly detection using SPC

## Related Documentation

- **[SPC Module Documentation](../../AM_QADF/05-modules/spc.md)** - Complete module documentation
- **[SPC API Reference](../../AM_QADF/06-api-reference/spc-api.md)** - API documentation
- **[Implementation Plan](../../../implementation_plans/SPC_MODULE_IMPLEMENTATION.md)** - Implementation details

---

**Next**: Explore the SPC module API for advanced use cases, integrate SPC analysis into your quality assessment workflows, and customize control limits and rules for your specific manufacturing processes.

**Previous**: [24: Validation and Benchmarking](24-validation-and-benchmarking.md) (if exists) or [23: Data Generation Workbench](23-data-generation.md) (if exists)