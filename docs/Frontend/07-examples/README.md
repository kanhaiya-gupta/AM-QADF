# Frontend Examples

This directory contains usage examples and tutorials for the AM-QADF Frontend Client.

## Examples

### Basic Examples

1. **[Data Query Example](data-query-example.md)** - Basic data querying
2. **[Voxelization Example](voxelization-example.md)** - Creating voxel grids
3. **[Visualization Example](visualization-example.md)** - 2D and 3D visualization

### Advanced Examples

1. **[Complete Workflow Example](complete-workflow-example.md)** - End-to-end workflow
2. **[Analytics Workflow Example](analytics-workflow-example.md)** - Analytics pipeline
3. **[Quality Assessment Example](quality-assessment-example.md)** - Quality analysis

## Quick Examples

### Query Data

```javascript
// Execute a query
const queryRequest = {
  model_id: "my_model",
  sources: ["hatching", "laser"],
  spatial_bbox: {
    min: [-50, -50, -50],
    max: [50, 50, 50]
  }
};

const response = await API.post('/data-query/query', queryRequest);
console.log('Query results:', response.data);
```

### Create Voxel Grid

```javascript
// Create a voxel grid
const gridRequest = {
  model_id: "my_model",
  resolution: 0.5,
  bbox_min: [-50, -50, -50],
  bbox_max: [50, 50, 50]
};

const response = await API.post('/voxelization/create', gridRequest);
console.log('Grid ID:', response.data.grid_id);
```

### Visualize Data

```javascript
// Get 3D visualization data
const vizRequest = {
  model_id: "my_model",
  grid_id: "grid_123",
  signal: "temperature"
};

const data = await API.post('/visualization/3d', vizRequest);
// Render with Three.js
```

## Related Documentation

- [Quick Start](../04-quick-start.md) - Getting started guide
- [Modules](../05-modules/README.md) - Module documentation
- [API Reference](../06-api-reference/README.md) - API documentation

---

**Parent**: [Frontend Documentation](../README.md)
