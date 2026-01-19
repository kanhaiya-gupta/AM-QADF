# Quality Assessment Module

## Overview

The Quality Assessment module provides interfaces for assessing data quality and completeness.

## Features

- **Completeness Analysis**: Assess data coverage and gaps
- **Signal Quality**: Evaluate signal-to-noise ratios
- **Quality Metrics**: Calculate quality metrics
- **Quality Reports**: Generate quality assessment reports

## Components

### Routes (`routes.py`)

- `GET /quality/` - Quality assessment page
- `POST /api/quality/assess` - Execute quality assessment
- `GET /api/quality/metrics` - Get quality metrics
- `POST /api/quality/report` - Generate quality report

### Services

- **QualityService**: Core quality assessment operations
- **MetricsService**: Quality metrics calculation
- **ConnectionService**: AM-QADF framework connection

### Templates

- `templates/quality/index.html` - Main quality assessment page

## Usage

```javascript
const assessmentRequest = {
  model_id: "my_model",
  grid_id: "grid_123",
  assessment_type: "completeness"
};

const response = await API.post('/quality/assess', assessmentRequest);
console.log(response.data);
```

## Related Documentation

- [Quality API Reference](../06-api-reference/quality-api.md)
- [Architecture](../02-architecture.md)

---

**Parent**: [Modules](README.md)
