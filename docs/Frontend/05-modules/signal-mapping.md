# Signal Mapping Module

## Overview

The Signal Mapping module provides interfaces for mapping signals from data sources to voxel grids.

## Features

- **Signal Interpolation**: Map signals to voxel grids using various methods
- **Method Selection**: Choose interpolation method (nearest, linear, IDW, KDE)
- **Grid Integration**: Integrate with existing voxel grids
- **Validation**: Validate mapping parameters and results

## Components

### Routes (`routes.py`)

- `GET /signal-mapping/` - Signal mapping page
- `POST /api/signal-mapping/map` - Execute signal mapping
- `GET /api/signal-mapping/methods` - List available methods
- `POST /api/signal-mapping/validate` - Validate mapping parameters

### Services

- **SignalMappingService**: Core signal mapping operations
- **DataQueryService**: Data querying for mapping
- **GridStorageService**: Grid storage integration
- **ValidationService**: Mapping validation

### Templates

- `templates/signal_mapping/index.html` - Main signal mapping page

## Usage

```javascript
const mappingRequest = {
  model_id: "my_model",
  grid_id: "grid_123",
  sources: ["hatching", "laser"],
  method: "nearest",
  signals: ["temperature", "power"]
};

const response = await API.post('/signal-mapping/map', mappingRequest);
console.log(response.data);
```

## Related Documentation

- [Signal Mapping API Reference](../06-api-reference/signal-mapping-api.md)
- [Architecture](../02-architecture.md)

---

**Parent**: [Modules](README.md)
