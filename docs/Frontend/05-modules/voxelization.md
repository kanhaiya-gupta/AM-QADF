# Voxelization Module

## Overview

The Voxelization module provides interfaces for creating and managing voxel grids from multi-source data.

## Features

- **Grid Creation**: Create uniform and adaptive resolution voxel grids
- **Grid Storage**: Store and retrieve voxel grids
- **Grid History**: Track grid creation history
- **Grid Validation**: Validate grid parameters and data
- **Visualization**: Preview grids using PyVista
- **Model Integration**: Create grids for specific models

## Components

### Routes (`routes.py`)

- `GET /voxelization/` - Voxelization page
- `POST /api/voxelization/create` - Create voxel grid
- `GET /api/voxelization/grids` - List grids
- `GET /api/voxelization/grids/{grid_id}` - Get grid details
- `GET /api/voxelization/history` - Get grid creation history
- `POST /api/voxelization/validate` - Validate grid parameters

### Services

- **VoxelizationService**: Core voxelization operations
- **GridStorageService**: Grid storage and retrieval
- **GridHistoryService**: Grid history management
- **GridValidationService**: Grid validation
- **PyVistaVisualizationService**: Grid visualization
- **ModelService**: Model integration

### Templates

- `templates/voxelization/index.html` - Main voxelization page

## Usage

### Create Voxel Grid

```javascript
const gridRequest = {
  model_id: "my_model",
  resolution: 0.5,
  bbox_min: [-50, -50, -50],
  bbox_max: [50, 50, 50],
  adaptive: false
};

const response = await API.post('/voxelization/create', gridRequest);
console.log(response.data.grid_id);
```

### List Grids

```javascript
const grids = await API.get('/voxelization/grids');
console.log(grids.data);
```

## Related Documentation

- [Voxelization API Reference](../06-api-reference/voxelization-api.md)
- [Architecture](../02-architecture.md)

---

**Parent**: [Modules](README.md)
