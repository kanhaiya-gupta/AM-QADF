# Analytics Module

## Overview

The Analytics module provides comprehensive analysis capabilities including statistical analysis, sensitivity analysis, SPC, process analysis, and virtual experiments.

## Features

- **Statistical Analysis**: Descriptive statistics, correlation, trends, patterns
- **Sensitivity Analysis**: Sobol, Morris, and other sensitivity methods
- **SPC Analysis**: Statistical Process Control charts and analysis
- **Process Analysis**: Process optimization and prediction
- **Virtual Experiments**: Parameter optimization and design of experiments
- **Reporting**: Generate analysis reports

## Sub-Modules

### Statistical Analysis

- Descriptive statistics
- Correlation analysis
- Trend analysis
- Pattern recognition

### Sensitivity Analysis

- Sobol indices
- Morris screening
- Parameter sensitivity ranking

### SPC Analysis

- Control charts (X-bar, R, S, Individual, Moving Range)
- Process capability analysis
- Multivariate SPC
- Control rule detection

### Process Analysis

- Early defect prediction
- Time-series forecasting
- Model tracking
- Process optimization

### Virtual Experiments

- Parameter optimization
- Design of experiments
- Scenario analysis

## Components

### Routes (`routes.py`)

- `GET /analytics/` - Analytics main page
- `GET /analytics/statistical` - Statistical analysis page
- `GET /analytics/sensitivity` - Sensitivity analysis page
- `GET /analytics/spc` - SPC analysis page
- `GET /analytics/process_analysis` - Process analysis page
- `GET /analytics/virtual_experiments` - Virtual experiments page
- `POST /api/analytics/statistical` - Execute statistical analysis
- `POST /api/analytics/sensitivity` - Execute sensitivity analysis
- `POST /api/analytics/spc` - Execute SPC analysis
- `POST /api/analytics/process_analysis` - Execute process analysis
- `POST /api/analytics/virtual_experiments` - Execute virtual experiment

### Services

- **StatisticalAnalysisService**: Statistical analysis operations
- **SensitivityAnalysisService**: Sensitivity analysis operations
- **SPCAnalysisService**: SPC analysis operations
- **ProcessAnalysisService**: Process analysis operations
- **VirtualExperimentsService**: Virtual experiment operations
- **ReportingService**: Report generation

### Templates

- `templates/analytics/index.html` - Main analytics page
- `templates/analytics/statistical/index.html` - Statistical analysis page
- `templates/analytics/sensitivity/index.html` - Sensitivity analysis page
- `templates/analytics/spc/index.html` - SPC page
- `templates/analytics/process_analysis/index.html` - Process analysis page
- `templates/analytics/virtual_experiments/index.html` - Virtual experiments page

## Usage

### Statistical Analysis

```javascript
const request = {
  model_id: "my_model",
  signals: ["temperature", "power"],
  analysis_type: "descriptive"
};

const result = await API.post('/analytics/statistical', request);
console.log(result.data);
```

### Sensitivity Analysis

```javascript
const request = {
  model_id: "my_model",
  method: "sobol",
  parameters: ["laser_power", "scan_speed"]
};

const result = await API.post('/analytics/sensitivity', request);
console.log(result.data);
```

## Related Documentation

- [Analytics API Reference](../06-api-reference/analytics-api.md)
- [Architecture](../02-architecture.md)

---

**Parent**: [Modules](README.md)
