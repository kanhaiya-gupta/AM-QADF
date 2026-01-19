# Complete Workflow Example

## Overview

This example demonstrates a complete workflow from data querying to visualization.

## Workflow Steps

1. Query data
2. Create voxel grid
3. Map signals to grid
4. Visualize results

## Complete Example

```javascript
async function completeWorkflow() {
    try {
        // Step 1: Query data
        const queryRequest = {
            model_id: "my_model",
            sources: ["hatching", "laser"],
            spatial_bbox: {
                min: [-50, -50, -50],
                max: [50, 50, 50]
            }
        };
        
        const queryResponse = await API.post('/data-query/query', queryRequest);
        console.log('Query completed:', queryResponse.data.query_id);

        // Step 2: Create voxel grid
        const gridRequest = {
            model_id: "my_model",
            resolution: 0.5,
            bbox_min: [-50, -50, -50],
            bbox_max: [50, 50, 50]
        };
        
        const gridResponse = await API.post('/voxelization/create', gridRequest);
        const gridId = gridResponse.data.grid_id;
        console.log('Grid created:', gridId);

        // Step 3: Map signals (if needed)
        // This would typically be done through signal mapping module

        // Step 4: Visualize
        const vizRequest = {
            model_id: "my_model",
            grid_id: gridId,
            signal: "temperature",
            colormap: "viridis"
        };
        
        const vizData = await API.post('/visualization/3d', vizRequest);
        console.log('Visualization data ready');
        
        // Render visualization with Three.js
        render3DVisualization(vizData.data);
        
    } catch (error) {
        console.error('Workflow failed:', error);
    }
}
```

## Related Documentation

- [Quick Start](../04-quick-start.md)
- [Modules](../05-modules/README.md)

---

**Parent**: [Examples](README.md)
