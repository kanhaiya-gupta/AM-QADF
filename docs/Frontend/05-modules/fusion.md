# Fusion Module

## Overview

The Fusion module provides interfaces for multi-source data fusion.

## Features

- **Multi-Source Fusion**: Fuse data from multiple sources
- **Fusion Strategies**: Various fusion methods (weighted average, etc.)
- **Fusion Quality**: Assess fusion quality
- **Grid Fusion**: Fuse multiple voxel grids

## Components

### Routes (`routes.py`)

- `GET /fusion/` - Fusion page
- `POST /api/fusion/fuse` - Execute fusion
- `GET /api/fusion/strategies` - List fusion strategies
- `POST /api/fusion/quality` - Assess fusion quality

### Services

- **FusionService**: Core fusion operations
- **StrategyService**: Fusion strategy management
- **ConnectionService**: AM-QADF framework connection

### Templates

- `templates/fusion/index.html` - Main fusion page

## Usage

```javascript
const fusionRequest = {
  model_id: "my_model",
  grid_ids: ["grid_1", "grid_2"],
  strategy: "weighted_average",
  weights: [0.5, 0.5]
};

const response = await API.post('/fusion/fuse', fusionRequest);
console.log(response.data);
```

## Related Documentation

- [Fusion API Reference](../06-api-reference/fusion-api.md)
- [Architecture](../02-architecture.md)

---

**Parent**: [Modules](README.md)
