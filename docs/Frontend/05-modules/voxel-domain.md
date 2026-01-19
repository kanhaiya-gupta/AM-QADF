# Voxel Domain Module

## Overview

The Voxel Domain module provides orchestration for complete voxel domain workflows.

## Features

- **Workflow Orchestration**: Orchestrate complete workflows
- **Domain Creation**: Create unified voxel domains
- **Multi-Step Operations**: Execute multi-step operations
- **Domain Management**: Manage voxel domains

## Components

### Routes (`routes.py`)

- `GET /voxel-domain/` - Voxel domain page
- `POST /api/voxel-domain/create` - Create voxel domain
- `POST /api/voxel-domain/orchestrate` - Orchestrate workflow
- `GET /api/voxel-domain/domains` - List domains

### Services

- **VoxelDomainService**: Core voxel domain operations
- **OrchestrationService**: Workflow orchestration
- **ConnectionService**: AM-QADF framework connection

### Templates

- `templates/voxel_domain/index.html` - Main voxel domain page

## Usage

```javascript
const domainRequest = {
  model_id: "my_model",
  resolution: 0.5,
  sources: ["hatching", "laser"],
  steps: ["query", "voxelize", "map_signals", "fuse"]
};

const response = await API.post('/voxel-domain/orchestrate', domainRequest);
console.log(response.data);
```

## Related Documentation

- [Voxel Domain API Reference](../06-api-reference/voxel-domain-api.md)
- [Architecture](../02-architecture.md)

---

**Parent**: [Modules](README.md)
