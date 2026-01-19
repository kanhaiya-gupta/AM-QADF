# Correction Module

## Overview

The Correction module provides interfaces for geometric distortion correction and calibration.

## Features

- **Geometric Correction**: Correct geometric distortions
- **Calibration**: Calibrate coordinate systems
- **Distortion Analysis**: Analyze distortion patterns
- **Correction Validation**: Validate correction results

## Components

### Routes (`routes.py`)

- `GET /correction/` - Correction page
- `POST /api/correction/correct` - Execute correction
- `POST /api/correction/calibrate` - Calibrate system
- `GET /api/correction/distortion` - Analyze distortion

### Services

- **CorrectionService**: Core correction operations
- **DistortionService**: Distortion analysis
- **ConnectionService**: AM-QADF framework connection

### Templates

- `templates/correction/index.html` - Main correction page

## Usage

```javascript
const correctionRequest = {
  model_id: "my_model",
  grid_id: "grid_123",
  correction_type: "geometric",
  parameters: {}
};

const response = await API.post('/correction/correct', correctionRequest);
console.log(response.data);
```

## Related Documentation

- [Correction API Reference](../06-api-reference/correction-api.md)
- [Architecture](../02-architecture.md)

---

**Parent**: [Modules](README.md)
