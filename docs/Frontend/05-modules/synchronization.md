# Synchronization Module

## Overview

The Synchronization module provides interfaces for temporal and spatial alignment of multi-source data.

## Features

- **Temporal Alignment**: Align data by time and layers
- **Spatial Alignment**: Align data by coordinate systems
- **Alignment Storage**: Store and retrieve alignments
- **Validation**: Validate alignment parameters

## Components

### Routes (`routes.py`)

- `GET /synchronization/` - Synchronization page
- `POST /api/synchronization/align` - Execute alignment
- `GET /api/synchronization/alignments` - List alignments
- `POST /api/synchronization/validate` - Validate alignment

### Services

- **SynchronizationService**: Core synchronization operations
- **AlignmentService**: Alignment operations
- **ConnectionService**: AM-QADF framework connection

### Templates

- `templates/synchronization/index.html` - Main synchronization page

## Usage

```javascript
const alignmentRequest = {
  model_id: "my_model",
  sources: ["hatching", "laser"],
  alignment_type: "temporal"
};

const response = await API.post('/synchronization/align', alignmentRequest);
console.log(response.data);
```

## Related Documentation

- [Synchronization API Reference](../06-api-reference/synchronization-api.md)
- [Architecture](../02-architecture.md)

---

**Parent**: [Modules](README.md)
