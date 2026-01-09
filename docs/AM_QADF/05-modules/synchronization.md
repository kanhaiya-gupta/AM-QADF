# Synchronization Module

## Overview

The Synchronization module handles temporal and spatial alignment of multi-source data, enabling accurate fusion of data from different sources with different coordinate systems and time references.

## Architecture

```mermaid
graph TB
    subgraph Temporal["⏰ Temporal Alignment"]
        TimePoint["Time Point<br/>📅 Time + Data"]
        LayerMapper["Layer Time Mapper<br/>🔄 Time ↔ Layer"]
        TemporalAligner["Temporal Aligner<br/>⏰ Align Timestamps"]
    end

    subgraph Spatial["📍 Spatial Transformation"]
        TransformMatrix["Transformation Matrix<br/>🔄 4x4 Matrix"]
        SpatialTransformer["Spatial Transformer<br/>📍 Transform Points"]
        TransformManager["Transformation Manager<br/>🗂️ Manage Transforms"]
    end

    subgraph Fusion["🔀 Data Fusion"]
        FusionStrategy["Fusion Strategy<br/>📊 Strategy Enum"]
        DataFusion["Data Fusion<br/>🔀 Combine Signals"]
    end

    subgraph Input["📥 Input Data"]
        Source1["Source 1<br/>🛤️ Hatching"]
        Source2["Source 2<br/>⚡ Laser"]
        Source3["Source 3<br/>🔬 CT"]
    end

    subgraph Output["📤 Output"]
        Aligned["Aligned Data<br/>✅ Synchronized"]
    end

    Source1 --> TemporalAligner
    Source2 --> TemporalAligner
    Source3 --> TemporalAligner

    TemporalAligner --> LayerMapper
    LayerMapper --> TimePoint

    Source1 --> SpatialTransformer
    Source2 --> SpatialTransformer
    Source3 --> SpatialTransformer

    SpatialTransformer --> TransformMatrix
    TransformManager --> TransformMatrix

    TimePoint --> DataFusion
    TransformMatrix --> DataFusion

    DataFusion --> FusionStrategy
    FusionStrategy --> Aligned

    %% Styling
    classDef temporal fill:#e3f2fd,stroke:#0277bd,stroke-width:2px
    classDef spatial fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef fusion fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef input fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef output fill:#ffccbc,stroke:#d84315,stroke-width:2px

    class TimePoint,LayerMapper,TemporalAligner temporal
    class TransformMatrix,SpatialTransformer,TransformManager spatial
    class FusionStrategy,DataFusion fusion
    class Source1,Source2,Source3 input
    class Aligned output
```

## Synchronization Workflow

```mermaid
flowchart TB
    Start([Multi-Source Data]) --> TemporalAlign["Temporal Alignment<br/>⏰ Align Timestamps"]
    
    TemporalAlign --> MapLayers["Map Time to Layers<br/>🔄 Layer Mapping"]
    
    MapLayers --> SpatialAlign["Spatial Alignment<br/>📍 Transform Coordinates"]
    
    SpatialAlign --> Register["Register Coordinate Systems<br/>🗂️ Register Transforms"]
    
    Register --> Transform["Transform Points<br/>🔄 Apply Transforms"]
    
    Transform --> PrepareFusion["Prepare for Fusion<br/>🔀 Align Data"]
    
    PrepareFusion --> SelectStrategy["Select Fusion Strategy<br/>📊 Choose Method"]
    
    SelectStrategy --> Fuse["Fuse Signals<br/>🔀 Combine"]
    
    Fuse --> Validate["Validate Alignment<br/>✅ Check Accuracy"]
    
    Validate --> Use([Use Aligned Data])
    
    %% Styling
    classDef step fill:#e3f2fd,stroke:#0277bd,stroke-width:2px
    classDef start fill:#c8e6c9,stroke:#2e7d32,stroke-width:3px
    classDef end fill:#ffccbc,stroke:#d84315,stroke-width:3px

    class TemporalAlign,MapLayers,SpatialAlign,Register,Transform,PrepareFusion,SelectStrategy,Fuse,Validate step
    class Start start
    class Use end
```

## Key Components

### Temporal Alignment

- **TimePoint**: Represents a point in time with associated data
- **LayerTimeMapper**: Maps between time and layer numbers
- **TemporalAligner**: Aligns timestamps from different sources

### Spatial Transformation

- **TransformationMatrix**: 4x4 transformation matrix (rotation, translation, scale)
- **SpatialTransformer**: Applies transformations to points
- **TransformationManager**: Manages multiple coordinate system transformations

### Data Fusion

- **FusionStrategy**: Enum of fusion strategies (average, weighted, median, etc.)
- **DataFusion**: Combines signals from multiple sources

## Usage Examples

### Temporal Alignment

```python
from am_qadf.synchronization import TemporalAligner, LayerTimeMapper

# Initialize aligner
aligner = TemporalAligner()

# Create layer-time mapper
mapper = LayerTimeMapper(
    layer_times={0: 0.0, 1: 10.0, 2: 20.0, ...}  # layer -> time mapping
)

# Align timestamps
aligned_data = aligner.align(
    data_sources=[hatching_data, laser_data, ct_data],
    reference_time=0.0
)

# Map time to layer
layer = mapper.time_to_layer(time=15.0)  # Returns layer 1
time = mapper.layer_to_time(layer=2)     # Returns 20.0
```

### Spatial Transformation

```python
from am_qadf.synchronization import (
    TransformationMatrix, 
    SpatialTransformer,
    TransformationManager
)

# Create transformation matrix (rotation + translation)
transform = TransformationMatrix.translation(10, 20, 30)
rotation = TransformationMatrix.rotation('z', 45)  # 45° around Z-axis
combined = TransformationMatrix.combine([rotation, transform])

# Initialize transformer
transformer = SpatialTransformer()
transformer.register_transformation('ct_to_build', combined)

# Transform points
ct_points = np.array([[0, 0, 0], [1, 1, 1]])
build_points = transformer.transform_points(
    points=ct_points,
    transformation_name='ct_to_build'
)

# Use transformation manager
manager = TransformationManager()
manager.register_coordinate_system('build_platform', origin=(0, 0, 0))
manager.register_coordinate_system('ct_scan', origin=(10, 20, 30))

# Get transformation
transform = manager.get_transformation('ct_scan', 'build_platform')
```

### Data Fusion

```python
from am_qadf.synchronization import DataFusion, FusionStrategy

# Initialize fusion
fusion = DataFusion(
    default_strategy=FusionStrategy.WEIGHTED_AVERAGE
)

# Register source quality
fusion.register_source_quality('hatching', 0.9)
fusion.register_source_quality('laser', 0.8)
fusion.register_source_quality('ct', 0.7)

# Fuse signals
signals = {
    'hatching': hatching_signal,
    'laser': laser_signal,
    'ct': ct_signal
}

fused = fusion.fuse_signals(
    signals=signals,
    strategy=FusionStrategy.WEIGHTED_AVERAGE,
    use_quality=True
)
```

## Transformation Types

```mermaid
graph LR
    subgraph Transforms["🔄 Transformation Types"]
        Translation["Translation<br/>📍 Move"]
        Rotation["Rotation<br/>🔄 Rotate"]
        Scale["Scale<br/>📏 Resize"]
        Combined["Combined<br/>🔗 All"]
    end

    subgraph Operations["⚙️ Operations"]
        Apply["Apply Transform<br/>📍 Transform Points"]
        Inverse["Inverse Transform<br/>↩️ Reverse"]
        Compose["Compose Transforms<br/>🔗 Chain"]
    end

    Translation --> Apply
    Rotation --> Apply
    Scale --> Apply
    Combined --> Apply

    Apply --> Inverse
    Apply --> Compose

    %% Styling
    classDef transform fill:#e3f2fd,stroke:#0277bd,stroke-width:2px
    classDef operation fill:#fff3e0,stroke:#e65100,stroke-width:2px

    class Translation,Rotation,Scale,Combined transform
    class Apply,Inverse,Compose operation
```

## Related

- [Signal Mapping Module](signal-mapping.md) - Uses synchronized data
- [Fusion Module](fusion.md) - Uses fusion strategies
- [Correction Module](correction.md) - Geometric correction

---

**Parent**: [Module Documentation](README.md)

