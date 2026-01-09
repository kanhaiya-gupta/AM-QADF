# Voxelization Module - Testing Guide

## Test Files

- `test_voxel_grid.py` - Core voxel grid operations
- `test_coordinate_systems.py` - Coordinate transformations
- `test_adaptive_resolution.py` - Adaptive resolution grids
- `test_multi_resolution.py` - Multi-resolution hierarchies
- `test_transformer.py` - Coordinate transformer

## Key Tests

### Voxel Grid Operations
- Voxel indexing (world → voxel conversion)
- Grid creation and manipulation
- Signal storage and retrieval
- Grid finalization

### Coordinate System Transformations
- All coordinate system combinations
- Transformation accuracy
- Inverse transformations
- Transformation chaining

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
pytest tests/unit/voxelization/ -m unit

# Run specific test files
pytest tests/unit/voxelization/test_voxel_grid.py
pytest tests/unit/voxelization/test_coordinate_systems.py
pytest tests/unit/voxelization/test_adaptive_resolution.py

# Run with coverage
pytest tests/unit/voxelization/ --cov=am_qadf.voxelization --cov-report=term-missing

# Run property-based tests for voxel grids
pytest tests/property_based/test_voxel_grid_properties.py -m property_based
```

## Related

- [Property-Based Tests](../04-test-categories/property-based-tests.md) - Voxel grid properties

---

**Parent**: [Module Testing Guides](README.md)

