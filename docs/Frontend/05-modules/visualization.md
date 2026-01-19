# Visualization Module

## Overview

The Visualization module provides 2D and 3D visualization capabilities for voxel data, signals, and analysis results.

## Features

- **3D Visualization**: Interactive 3D rendering of voxel grids
- **2D Visualization**: 2D charts and plots (Plotly.js)
- **Slice Views**: Cross-sectional views of 3D data
- **Interactive Controls**: Zoom, pan, rotate, and filter
- **Export**: Export visualizations as images or HTML

## Components

### Routes (`routes.py`)

- `GET /visualization/` - Visualization main page
- `GET /visualization/3d` - 3D visualization page
- `POST /api/visualization/3d` - Get 3D visualization data
- `POST /api/visualization/2d` - Get 2D plot data
- `POST /api/visualization/slice` - Get slice view data

### Services

- **VisualizationService**: Core visualization operations
- **PlotService**: 2D plot generation
- **ConnectionService**: AM-QADF framework connection

### Templates

- `templates/visualization/index.html` - Main visualization page
- `templates/visualization/3d/index.html` - 3D visualization page

### Static Files

- `static/js/visualization/` - Visualization JavaScript
- Uses Three.js for 3D rendering
- Uses Plotly.js for 2D charts

## Usage

### 3D Visualization

```javascript
const request = {
  model_id: "my_model",
  grid_id: "grid_123",
  signal: "temperature",
  colormap: "viridis"
};

const data = await API.post('/visualization/3d', request);
// Render with Three.js
```

### 2D Plot

```javascript
const request = {
  model_id: "my_model",
  signal: "temperature",
  plot_type: "line"
};

const plotData = await API.post('/visualization/2d', request);
// Render with Plotly.js
```

## Related Documentation

- [Visualization API Reference](../06-api-reference/visualization-api.md)
- [Architecture](../02-architecture.md)

---

**Parent**: [Modules](README.md)
