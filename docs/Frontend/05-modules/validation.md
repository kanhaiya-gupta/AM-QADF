# Validation Module

## Overview

The Validation module provides interfaces for validation and benchmarking.

## Features

- **Accuracy Validation**: Validate processing accuracy
- **Benchmarking**: Performance benchmarking
- **MPM Comparison**: Compare with MPM systems
- **Validation Reports**: Generate validation reports

## Components

### Routes (`routes.py`)

- `GET /validation/` - Validation page
- `POST /api/validation/validate` - Execute validation
- `POST /api/validation/benchmark` - Run benchmark
- `GET /api/validation/metrics` - Get validation metrics

### Services

- **ValidationService**: Core validation operations
- **ValidationMetricsService**: Validation metrics calculation
- **ConnectionService**: AM-QADF framework connection

### Templates

- `templates/validation/index.html` - Main validation page

## Usage

```javascript
const validationRequest = {
  model_id: "my_model",
  grid_id: "grid_123",
  validation_type: "accuracy"
};

const response = await API.post('/validation/validate', validationRequest);
console.log(response.data);
```

## Related Documentation

- [Validation API Reference](../06-api-reference/validation-api.md)
- [Architecture](../02-architecture.md)

---

**Parent**: [Modules](README.md)
