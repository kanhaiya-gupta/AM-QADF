# Voxelization Module - Testing Guide

## Test Files

Tests mirror `src/am_qadf/voxelization/` module names:

- `test_uniform_resolution.py` - Uniform resolution VoxelGrid (add_point, finalize, get_voxel, get_statistics)
- `test_adaptive_resolution.py` - Adaptive resolution grids
- `test_multi_resolution.py` - Multi-resolution hierarchies
- `test_geometry_voxelizer.py` - STL/hatching voxelization, get_stl_bounding_box, export_to_paraview

Coordinate system tests live under `tests/unit/python/coordinate_systems/` (see `src/am_qadf/coordinate_systems/`).

## Key Tests

### Uniform Resolution (VoxelGrid)
- Voxel indexing (world → voxel conversion)
- Grid creation and manipulation
- Signal storage and retrieval
- Grid finalization

### Resolution Adaptation
- Resolution adaptation logic
- Memory efficiency
- Performance with adaptive grids

### Multi-Resolution
- Hierarchy construction
- Level-of-detail selection
- Memory management

## Coverage Target

**90%+**

## Example Tests

```python
def test_world_to_voxel_conversion():
    """Test world coordinate to voxel index conversion."""
    
def test_coordinate_transformation_accuracy():
    """Test coordinate transformation accuracy."""
    
def test_adaptive_resolution_logic():
    """Test adaptive resolution adaptation logic."""
```

## Running Voxelization Module Tests

```bash
# Run all voxelization tests
pytest tests/unit/python/voxelization/ -m unit

# Run specific test files
pytest tests/unit/python/voxelization/test_uniform_resolution.py
pytest tests/unit/python/voxelization/test_adaptive_resolution.py
pytest tests/unit/python/voxelization/test_geometry_voxelizer.py

# Run with coverage
pytest tests/unit/python/voxelization/ --cov=am_qadf.voxelization --cov-report=term-missing

# Run property-based tests for voxel grids
pytest tests/property_based/test_voxel_grid_properties.py -m property_based
```

## Related

- [Property-Based Tests](../04-test-categories/property-based-tests.md) - Voxel grid properties

---

**Parent**: [Module Testing Guides](README.md)

