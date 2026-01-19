# Anomaly Detection Module

## Overview

The Anomaly Detection module provides interfaces for detecting anomalies in voxel data.

## Features

- **Multiple Algorithms**: Various anomaly detection algorithms
- **Ensemble Methods**: Combine multiple detectors
- **Anomaly Visualization**: Visualize detected anomalies
- **Anomaly Reports**: Generate anomaly reports

## Components

### Routes (`routes.py`)

- `GET /anomaly-detection/` - Anomaly detection page
- `POST /api/anomaly-detection/detect` - Execute anomaly detection
- `GET /api/anomaly-detection/algorithms` - List available algorithms
- `GET /api/anomaly-detection/results/{detection_id}` - Get detection results

### Services

- **AnomalyDetectionService**: Core anomaly detection operations
- **ConnectionService**: AM-QADF framework connection

### Templates

- `templates/anomaly_detection/index.html` - Main anomaly detection page

## Usage

```javascript
const detectionRequest = {
  model_id: "my_model",
  grid_id: "grid_123",
  algorithm: "statistical",
  signals: ["temperature", "power"]
};

const response = await API.post('/anomaly-detection/detect', detectionRequest);
console.log(response.data);
```

## Related Documentation

- [Anomaly Detection API Reference](../06-api-reference/anomaly-detection-api.md)
- [Architecture](../02-architecture.md)

---

**Parent**: [Modules](README.md)
