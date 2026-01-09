# Coordinate System Information for 3D Printing Data

## Overview

Coordinate system information is **critical** for merging 3D model data (STL, hatching paths) with other data sources (ISPM monitoring, CT scans, sensor data). This document explains what coordinate system information is stored and how to use it.

## What Information is Stored

For each STL model and its associated hatching layers, we store the following coordinate system information:

### 1. Coordinate System Type
- **Type**: `build_platform` (machine coordinate system)
- **Description**: Coordinates are in the build platform/machine space after the part has been positioned using `dropToPlatform()`

### 2. Origin (Translation)
- **Location**: `coordinate_system.origin`
- **Format**: `{x: float, y: float, z: float}`
- **Description**: The position of the part on the build platform (in mm). This is the translation offset from the machine origin (0,0,0).

### 3. Rotation
- **Location**: `coordinate_system.rotation`
- **Format**: `{x_deg: float, y_deg: float, z_deg: float}`
- **Description**: Rotation angles in degrees about X, Y, Z axes. Applied sequentially in order: X, then Y, then Z.

### 4. Scale Factor
- **Location**: `coordinate_system.scale_factor`
- **Format**: `{x: float, y: float, z: float}`
- **Description**: Scale factors for each axis. Typically `(1.0, 1.0, 1.0)` unless the part is scaled.

### 5. Bounding Box
- **Location**: `coordinate_system.bounding_box`
- **Format**: `{min: [x, y, z], max: [x, y, z]}`
- **Description**: The bounding box of the part in the build platform coordinate system.

## ISPM (In-Situ Process Monitoring) Coordinate System

ISPM sensors have their own coordinate system information, which is **critical** for merging ISPM monitoring data with STL/hatching/CT scan data:

### ISPM Coordinate System Fields

1. **Type**: `ispm_sensor` (ISPM sensor coordinate system)
2. **Origin**: Typically aligned with build platform origin (0, 0, 0)
3. **Rotation**: Typically (0, 0, 0) - aligned with build platform
4. **Sensor Position**: Fixed locations on the machine
5. **Measurement Coordinates**: Spatial coordinates (x, y, z) where measurements were taken
6. **Bounding Box**: Spatial extent of ISPM measurements

### Key Characteristics

- **Sensor Position**: ISPM sensors are typically positioned at fixed locations on the machine
- **Measurement Space**: Measurement coordinates are in build platform space
- **Alignment**: Usually aligned with build platform (rotation = 0,0,0)
- **Spatial Range**: Defined by the bounding box of actual measurements

## CT Scan Coordinate System

CT scans have their own coordinate system information, which is **critical** for merging CT scan data with STL/hatching data:

### CT Scan Coordinate System Fields

1. **Type**: `ct_scan` (CT scan coordinate system)
2. **Origin**: Corner of the voxel grid (x0, y0, z0) in mm
3. **Rotation**: Optional rotation angles (x_deg, y_deg, z_deg) - typically (0, 0, 0) if aligned with build platform
4. **Voxel Spacing**: Physical size of each voxel (dx, dy, dz) in mm
5. **Grid Dimensions**: Number of voxels (nx, ny, nz)
6. **Bounding Box**: Spatial extent of the CT scan volume

### Key Differences from STL Coordinate System

- **Voxel Spacing**: CT scans have discrete voxel spacing (e.g., 0.1mm per voxel)
- **Grid-Based**: CT scans are organized as discrete voxel grids, not continuous coordinates
- **Origin**: Typically the corner of the voxel grid, not the part center
- **Alignment**: Usually aligned with build platform (rotation = 0,0,0), but can be rotated if needed

## Where It's Stored

### STL Models Collection
```json
{
  "model_id": "part_001",
  "metadata": {
    "coordinate_system": {
      "type": "build_platform",
      "origin": {"x": 0.0, "y": 0.0, "z": 0.0},
      "rotation": {"x_deg": 0.0, "y_deg": 0.0, "z_deg": 0.0},
      "scale_factor": {"x": 1.0, "y": 1.0, "z": 1.0},
      "bounding_box": {
        "min": [0.0, 0.0, 0.0],
        "max": [50.0, 50.0, 30.0]
      }
    }
  }
}
```

### CT Scan Data Collection
```json
{
  "model_id": "part_001",
  "scan_id": "scan_part_001_20250101_120000",
  "coordinate_system": {
    "type": "ct_scan",
    "origin": {"x": 0.0, "y": 0.0, "z": 0.0},
    "rotation": {"x_deg": 0.0, "y_deg": 0.0, "z_deg": 0.0},
    "scale_factor": {"x": 1.0, "y": 1.0, "z": 1.0},
    "voxel_spacing": {"x": 0.1, "y": 0.1, "z": 0.1},
    "grid_dimensions": {"nx": 200, "ny": 200, "nz": 200},
    "bounding_box": {
      "min": [0.0, 0.0, 0.0],
      "max": [20.0, 20.0, 20.0]
    }
  },
  "voxel_grid": {
    "dimensions": [200, 200, 200],
    "spacing": [0.1, 0.1, 0.1],
    "origin": [0.0, 0.0, 0.0]
  },
  "density_values": [...],
  "porosity_map": [...],
  "defect_locations": [[x1, y1, z1], [x2, y2, z2], ...]
}
```

### ISPM Monitoring Data Collection
```json
{
  "model_id": "part_001",
  "layer_index": 0,
  "timestamp": "2025-01-01T12:00:00",
  "spatial_coordinates": [25.5, 30.2, 0.05],
  "melt_pool_temperature": 1500.0,
  "melt_pool_size": {"width": 0.5, "length": 0.8, "depth": 0.3},
  "peak_temperature": 1700.0,
  "cooling_rate": 100.0,
  "temperature_gradient": 50.0,
  "process_event": null,
  "coordinate_system": {
    "type": "ispm_sensor",
    "origin": {"x": 0.0, "y": 0.0, "z": 0.0},
    "rotation": {"x_deg": 0.0, "y_deg": 0.0, "z_deg": 0.0},
    "scale_factor": {"x": 1.0, "y": 1.0, "z": 1.0},
    "sensor_position": {
      "description": "ISPM sensors are typically positioned at fixed locations on the machine",
      "coordinate_space": "build_platform"
    },
    "measurement_coordinates": {
      "description": "Measurement coordinates (x, y, z) are in build platform space",
      "spatial_range": {
        "x": [0.0, 50.0],
        "y": [0.0, 50.0],
        "z": [0.0, 30.0]
      }
    },
    "bounding_box": {
      "min": [0.0, 0.0, 0.0],
      "max": [50.0, 50.0, 30.0]
    }
  }
}
```

### Hatching Layers Collection
```json
{
  "model_id": "part_001",
  "layer_index": 0,
  "z_position": 0.05,
  "coordinate_system": {
    "type": "build_platform",
    "origin": {"x": 0.0, "y": 0.0, "z": 0.0},
    "rotation": {"x_deg": 0.0, "y_deg": 0.0, "z_deg": 0.0},
    "scale_factor": {"x": 1.0, "y": 1.0, "z": 1.0},
    "bounding_box": {
      "min": [0.0, 0.0, 0.0],
      "max": [50.0, 50.0, 30.0]
    }
  },
  "hatches": [
    {
      "points": [[x1, y1, z1], [x2, y2, z2], ...],
      ...
    }
  ]
}
```

## How to Use This Information

### 1. Transform Coordinates Between Systems

When merging STL/hatching data with ISPM or CT scan data, you may need to transform coordinates:

```python
from src.data_pipeline.data_warehouse_clients.synchronization.spatial_transformation import (
    TransformationManager, TransformationMatrix
)

# Get coordinate system from STL model
stl_doc = mongo_client.find_one('stl_models', {'model_id': 'part_001'})
coord_sys = stl_doc['metadata']['coordinate_system']

# Create transformation matrix
# Origin translation
origin = np.array([
    coord_sys['origin']['x'],
    coord_sys['origin']['y'],
    coord_sys['origin']['z']
])

# Rotation (convert degrees to radians)
rotation = np.array([
    np.radians(coord_sys['rotation']['x_deg']),
    np.radians(coord_sys['rotation']['y_deg']),
    np.radians(coord_sys['rotation']['z_deg'])
])

# Scale
scale = np.array([
    coord_sys['scale_factor']['x'],
    coord_sys['scale_factor']['y'],
    coord_sys['scale_factor']['z']
])

# Apply transformation to points
# (Use TransformationManager for complex transformations)
```

### 2. Merge with ISPM Data

ISPM monitoring data coordinates are typically already in build platform space. Use the coordinate system information to verify alignment and merge with other data:

```python
# Load ISPM data with coordinate system
ispm_docs = mongo_client.find('ispm_monitoring_data', {'model_id': 'part_001'})

# Get coordinate system from first document (all should have same coordinate system)
if ispm_docs:
    ispm_coord_sys = ispm_docs[0].get('coordinate_system', {})
    
    # Check if ISPM is in build platform coordinates
    if ispm_coord_sys.get('type') == 'ispm_sensor':
        # ISPM coordinates are typically in build platform space
        # Can be directly compared with STL/hatching coordinates
        
        for ispm_doc in ispm_docs:
            ispm_coords = np.array(ispm_doc['spatial_coordinates'])  # [x, y, z]
            ispm_temp = ispm_doc['melt_pool_temperature']
            
            # Find nearest hatching point
            for layer_doc in hatching_docs:
                for hatch in layer_doc['hatches']:
                    points = np.array(hatch['points'])
                    distances = np.linalg.norm(points - ispm_coords, axis=1)
                    nearest_idx = np.argmin(distances)
                    if distances[nearest_idx] < 1.0:  # Within 1mm
                        nearest_point = points[nearest_idx]
                        print(f"ISPM measurement at {ispm_coords} near hatching point {nearest_point}")
                        print(f"  Temperature: {ispm_temp}°C")
                        print(f"  Distance: {distances[nearest_idx]:.3f}mm")
    
    # If coordinate systems differ, apply transformation
    # (Use TransformationManager for complex transformations)
```

### 3. Merge with CT Scan Data

CT scan data is in a voxel grid. Use the coordinate system information to map CT voxels to build platform coordinates:

```python
# Load CT scan data with coordinate system
ct_doc = mongo_client.find_one('ct_scan_data', {'model_id': 'part_001'})
ct_coord_sys = ct_doc['coordinate_system']
voxel_grid = ct_doc['voxel_grid']

# Get CT scan coordinate system parameters
ct_origin = np.array([
    ct_coord_sys['origin']['x'],
    ct_coord_sys['origin']['y'],
    ct_coord_sys['origin']['z']
])
ct_spacing = np.array([
    ct_coord_sys['voxel_spacing']['x'],
    ct_coord_sys['voxel_spacing']['y'],
    ct_coord_sys['voxel_spacing']['z']
])
ct_dimensions = np.array([
    ct_coord_sys['grid_dimensions']['nx'],
    ct_coord_sys['grid_dimensions']['ny'],
    ct_coord_sys['grid_dimensions']['nz']
])

# Convert CT voxel indices to physical coordinates (build platform space)
def voxel_to_physical(voxel_idx):
    """Convert voxel index (i, j, k) to physical coordinates (x, y, z) in mm."""
    i, j, k = voxel_idx
    x = ct_origin[0] + i * ct_spacing[0]
    y = ct_origin[1] + j * ct_spacing[1]
    z = ct_origin[2] + k * ct_spacing[2]
    return np.array([x, y, z])

# Example: Get physical coordinates of a defect
defect_locations = ct_doc['defect_locations']  # List of [i, j, k] voxel indices
for defect_voxel in defect_locations:
    physical_coords = voxel_to_physical(defect_voxel)
    # Now physical_coords is in build platform coordinates
    # Can be directly compared with hatching path coordinates

# Map CT scan to STL/hatching coordinates
# If CT scan and STL are in the same coordinate system (build_platform),
# coordinates can be directly compared. Otherwise, apply transformation.
```

## Deriving Information from Bounding Box

As you mentioned, we can derive coordinate system information from the bounding box:

1. **Part Center**: `(bbox_min + bbox_max) / 2`
2. **Part Dimensions**: `bbox_max - bbox_min`
3. **Part Volume**: `np.prod(bbox_max - bbox_min)`
4. **Coordinate Ranges**: 
   - X: `[bbox_min[0], bbox_max[0]]`
   - Y: `[bbox_min[1], bbox_max[1]]`
   - Z: `[bbox_min[2], bbox_max[2]]`

## Best Practices

1. **Always check coordinate system type** before merging data
2. **Store coordinate system info** with all spatial data (STL, hatching, ISPM, CT)
3. **Use TransformationManager** from the framework for complex transformations
4. **Document coordinate system** in data warehouse schemas
5. **Validate transformations** by checking that transformed points fall within expected bounds

## Example: Complete Coordinate System Usage

```python
# 1. Load STL model with coordinate system
stl_doc = mongo_client.find_one('stl_models', {'model_id': 'part_001'})
stl_coord_sys = stl_doc['metadata']['coordinate_system']

# 2. Load hatching layers (same coordinate system as STL)
hatching_docs = mongo_client.find('hatching_layers', {'model_id': 'part_001'})

# 3. Extract laser path points (already in build platform coordinates)
for layer_doc in hatching_docs:
    for hatch in layer_doc['hatches']:
        points = np.array(hatch['points'])  # Already in build platform coords
        # Use these points directly for merging with ISPM/CT data

# 4. Load CT scan data with coordinate system
ct_doc = mongo_client.find_one('ct_scan_data', {'model_id': 'part_001'})
ct_coord_sys = ct_doc['coordinate_system']

# 5. Convert CT voxel indices to physical coordinates
def ct_voxel_to_physical(voxel_idx):
    """Convert CT voxel index to physical coordinates."""
    i, j, k = voxel_idx
    origin = np.array([
        ct_coord_sys['origin']['x'],
        ct_coord_sys['origin']['y'],
        ct_coord_sys['origin']['z']
    ])
    spacing = np.array([
        ct_coord_sys['voxel_spacing']['x'],
        ct_coord_sys['voxel_spacing']['y'],
        ct_coord_sys['voxel_spacing']['z']
    ])
    return origin + np.array([i, j, k]) * spacing

# 6. Get defect locations in physical coordinates
defect_locations = ct_doc['defect_locations']  # List of [i, j, k] voxel indices
for defect_voxel in defect_locations:
    defect_physical = ct_voxel_to_physical(defect_voxel)
    # defect_physical is now in build platform coordinates
    # Can be directly compared with hatching path coordinates
    
    # Find nearest hatching point
    for layer_doc in hatching_docs:
        for hatch in layer_doc['hatches']:
            points = np.array(hatch['points'])
            distances = np.linalg.norm(points - defect_physical, axis=1)
            nearest_idx = np.argmin(distances)
            if distances[nearest_idx] < 0.5:  # Within 0.5mm
                print(f"Defect near hatching point: {points[nearest_idx]}")

# 7. Load ISPM data with coordinate system
ispm_docs = mongo_client.find('ispm_monitoring_data', {'model_id': 'part_001'})
if ispm_docs:
    ispm_coord_sys = ispm_docs[0].get('coordinate_system', {})
    
    # ISPM coordinates are typically already in build platform space
    for ispm_doc in ispm_docs:
        ispm_coords = np.array(ispm_doc['spatial_coordinates'])  # [x, y, z]
        ispm_temp = ispm_doc['melt_pool_temperature']
        
        # Find nearest hatching point
        for layer_doc in hatching_docs:
            for hatch in layer_doc['hatches']:
                points = np.array(hatch['points'])
                distances = np.linalg.norm(points - ispm_coords, axis=1)
                nearest_idx = np.argmin(distances)
                if distances[nearest_idx] < 1.0:  # Within 1mm
                    print(f"ISPM temp {ispm_temp}°C at {ispm_coords} near hatch point {points[nearest_idx]}")
        
        # Find nearest CT scan defect
        for defect_voxel in defect_locations:
            defect_physical = ct_voxel_to_physical(defect_voxel)
            distance = np.linalg.norm(ispm_coords - defect_physical)
            if distance < 0.5:  # Within 0.5mm
                print(f"ISPM measurement near CT defect at {defect_physical}")
    
    # Now all data sources are merged in build platform coordinates
    # ...
```

## Notes

### STL/Hatching Coordinate Systems
- The coordinate system information is extracted from the `pyslm.Part` object **after** `dropToPlatform()` is called
- This ensures coordinates are in the **build platform space** (machine coordinates)
- All hatching path coordinates (`points` in `hatches`) are already in this coordinate system
- Type: `build_platform`

### ISPM Coordinate Systems
- ISPM coordinate system is typically aligned with build platform (rotation = 0,0,0)
- Measurement coordinates (x, y, z) are in build platform space
- Sensors are positioned at fixed locations on the machine
- Type: `ispm_sensor`
- **Important**: ISPM coordinates are typically in the same build platform space, so they can be directly compared with STL/hatching/CT scan coordinates if all have the same origin/rotation

### CT Scan Coordinate Systems
- CT scan coordinate system is defined by the voxel grid origin, spacing, and dimensions
- Typically aligned with build platform (rotation = 0,0,0)
- Voxel indices (i, j, k) can be converted to physical coordinates using: `physical = origin + [i, j, k] * spacing`
- Type: `ct_scan`
- **Important**: CT scan coordinates are in the same build platform space, so they can be directly compared with STL/hatching/ISPM coordinates if all have the same origin/rotation

### Merging Data Sources
- When merging with other data sources, check the `coordinate_system.type` field
- If types match and origins/rotations are the same, coordinates can be directly compared
- If different, apply transformations using the `TransformationManager` from the framework
- Always validate transformations by checking that transformed points fall within expected bounds

