# Quick Start Guide

## Basic Workflow

```mermaid
flowchart LR
    A([Start]) --> B["1️⃣ Query Data<br/>🔍 Unified Query Client"]
    B --> C["2️⃣ Create Voxel Grid<br/>🧊 VoxelGrid"]
    C --> D["3️⃣ Map Signals<br/>🎯 Interpolation"]
    D --> E["4️⃣ Assess Quality<br/>✅ Quality Metrics"]
    E --> F["5️⃣ Visualize<br/>🧊 3D Rendering"]
    F --> G([Complete])
    
    %% Styling
    classDef step fill:#e3f2fd,stroke:#0277bd,stroke-width:2px
    classDef start fill:#c8e6c9,stroke:#2e7d32,stroke-width:3px
    classDef end fill:#ffccbc,stroke:#e65100,stroke-width:3px

    class A start
    class B,C,D,E,F step
    class G end
```

### 1. Query Data

```python
from am_qadf.query import UnifiedQueryClient
from am_qadf.query.mongodb_client import MongoDBClient

# Initialize query client
mongodb_client = MongoDBClient(connection_string="mongodb://localhost:27017")
query_client = UnifiedQueryClient(mongodb_client)

# Query data
result = query_client.query(
    model_id="my_model",
    sources=['hatching', 'laser'],
    spatial_bbox=((-50, -50, -50), (50, 50, 50))
)
```

### 2. Create Voxel Grid

```python
from am_qadf.voxelization import VoxelGrid

# Create voxel grid
grid = VoxelGrid(
    bbox_min=(-50, -50, -50),
    bbox_max=(50, 50, 50),
    resolution=1.0
)
```

### 3. Map Signals to Voxels

```python
from am_qadf.signal_mapping.execution.sequential import interpolate_to_voxels

# Map signals
voxel_grid = interpolate_to_voxels(
    points=result.points,
    signals=result.signals,
    voxel_grid=grid,
    method='nearest'
)
```

### 4. Assess Quality

```python
from am_qadf.quality import QualityAssessmentClient

quality_client = QualityAssessmentClient(mongodb_client)
quality_result = quality_client.assess_completeness(
    model_id="my_model",
    voxel_grid=voxel_grid
)
```

### 5. Visualize

```python
from am_qadf.visualization import VoxelRenderer

renderer = VoxelRenderer()
renderer.render(voxel_grid, signal_name='power')
```

## Complete Example

```python
from am_qadf.voxel_domain import VoxelDomainClient
from am_qadf.query import UnifiedQueryClient
from am_qadf.query.mongodb_client import MongoDBClient

# Initialize
mongodb_client = MongoDBClient("mongodb://localhost:27017")
query_client = UnifiedQueryClient(mongodb_client)
voxel_client = VoxelDomainClient(query_client, base_resolution=1.0)

# Map signals to voxels
voxel_grid = voxel_client.map_signals_to_voxels(
    model_id="my_model",
    sources=['hatching', 'laser', 'ct'],
    interpolation_method='linear'
)

# Save to MongoDB
voxel_client.save_voxel_grid(
    model_id="my_model",
    grid_name="fused_grid",
    voxel_grid=voxel_grid,
    mongo_client=mongodb_client
)
```

## Next Steps

- **[Modules](05-modules/README.md)** - Explore individual modules
- **[Examples](07-examples/README.md)** - More examples
- **[API Reference](06-api-reference/README.md)** - Complete API docs

---

**Related**: [Overview](01-overview.md) | [Modules](05-modules/README.md)

