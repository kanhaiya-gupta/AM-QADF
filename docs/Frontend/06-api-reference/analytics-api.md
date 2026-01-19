# Analytics API Reference

## Overview

API endpoints for analytics operations including statistical analysis, sensitivity analysis, SPC, process analysis, and virtual experiments.

## Endpoints

### GET /analytics/

Serve the analytics main page.

**Response**: HTML page

---

### POST /api/analytics/statistical

Execute statistical analysis.

**Request Body**:
```json
{
  "model_id": "string",
  "signals": ["temperature", "power"],
  "analysis_type": "descriptive",
  "options": {}
}
```

**Response**:
```json
{
  "status": "success",
  "data": {
    "analysis_id": "analysis_123",
    "results": {
      "mean": {...},
      "std": {...},
      "correlation": {...}
    }
  }
}
```

---

### POST /api/analytics/sensitivity

Execute sensitivity analysis.

**Request Body**:
```json
{
  "model_id": "string",
  "method": "sobol",
  "parameters": ["laser_power", "scan_speed"],
  "options": {}
}
```

**Response**:
```json
{
  "status": "success",
  "data": {
    "analysis_id": "analysis_123",
    "sensitivity_indices": {...}
  }
}
```

---

### POST /api/analytics/spc

Execute SPC analysis.

**Request Body**:
```json
{
  "model_id": "string",
  "signal": "temperature",
  "chart_type": "xbar",
  "options": {}
}
```

**Response**:
```json
{
  "status": "success",
  "data": {
    "analysis_id": "analysis_123",
    "chart_data": {...},
    "control_limits": {...}
  }
}
```

---

### POST /api/analytics/process_analysis

Execute process analysis.

**Request Body**:
```json
{
  "model_id": "string",
  "analysis_type": "prediction",
  "options": {}
}
```

**Response**:
```json
{
  "status": "success",
  "data": {
    "analysis_id": "analysis_123",
    "results": {...}
  }
}
```

---

### POST /api/analytics/virtual_experiments

Execute virtual experiment.

**Request Body**:
```json
{
  "model_id": "string",
  "experiment_type": "optimization",
  "parameters": {...},
  "options": {}
}
```

**Response**:
```json
{
  "status": "success",
  "data": {
    "experiment_id": "exp_123",
    "results": {...}
  }
}
```

## Related Documentation

- [Analytics Module](../05-modules/analytics.md)
- [API Reference](README.md)

---

**Parent**: [API Reference](README.md)
