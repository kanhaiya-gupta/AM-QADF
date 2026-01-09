# Analytics Module - Testing Guide

## Test Structure

```
tests/unit/analytics/
├── sensitivity_analysis/
│   ├── test_client.py
│   ├── test_global_analysis.py
│   ├── test_local_analysis.py
│   ├── test_doe.py
│   ├── test_uncertainty.py
│   ├── test_query.py
│   └── test_storage.py
├── statistical_analysis/
│   ├── test_client.py
│   ├── test_descriptive_stats.py
│   ├── test_correlation.py
│   ├── test_trends.py
│   ├── test_patterns.py
│   ├── test_multivariate.py
│   ├── test_time_series.py
│   ├── test_regression.py
│   └── test_nonparametric.py
├── quality_assessment/
│   ├── test_client.py
│   ├── test_completeness.py
│   ├── test_signal_quality.py
│   ├── test_alignment_accuracy.py
│   └── test_data_quality.py
├── process_analysis/
│   ├── test_sensor_analysis.py
│   ├── test_parameter_analysis.py
│   ├── test_quality_analysis.py
│   └── test_optimization.py
├── virtual_experiments/
│   ├── test_client.py
│   ├── test_parameter_optimizer.py
│   ├── test_result_analyzer.py
│   ├── test_comparison_analyzer.py
│   ├── test_storage.py
│   └── test_query.py
└── reporting/
    ├── test_report_generators.py
    ├── test_visualization.py
    └── test_documentation.py
```

## Key Tests

### Sensitivity Analysis
- Analysis algorithm correctness (Sobol, Morris, etc.)
- Result storage and retrieval
- Query interface functionality
- Performance with large datasets

### Statistical Analysis
- Descriptive statistics accuracy
- Correlation calculations
- Trend analysis
- Pattern detection

### Quality Assessment
- Quality metric calculations
- Assessment workflows
- Result aggregation

### Virtual Experiments
- Parameter optimization
- Result analysis
- Comparison analysis

## Coverage Target

**80%+**

## Example Tests

```python
def test_sensitivity_analysis_sobol():
    """Test Sobol sensitivity analysis."""
    
def test_statistical_correlation():
    """Test correlation calculations."""
    
def test_virtual_experiment_optimization():
    """Test parameter optimization."""
```

## Running Analytics Module Tests

```bash
# Run all analytics tests
pytest tests/unit/analytics/ -m unit

# Run by submodule
pytest tests/unit/analytics/sensitivity_analysis/     # Sensitivity analysis
pytest tests/unit/analytics/statistical_analysis/      # Statistical analysis
pytest tests/unit/analytics/quality_assessment/         # Quality assessment
pytest tests/unit/analytics/process_analysis/         # Process analysis
pytest tests/unit/analytics/virtual_experiments/       # Virtual experiments
pytest tests/unit/analytics/reporting/                 # Reporting

# Run specific analysis type
pytest tests/unit/analytics/sensitivity_analysis/test_global_analysis.py
pytest tests/unit/analytics/statistical_analysis/test_descriptive_stats.py

# Run with coverage
pytest tests/unit/analytics/ --cov=am_qadf.analytics --cov-report=term-missing

# Run integration tests for analytics
pytest tests/integration/test_analytics_workflow.py -m integration

# Run E2E tests for analytics pipeline
pytest tests/e2e/test_analytics_pipeline.py -m e2e
```

---

**Parent**: [Module Testing Guides](README.md)

