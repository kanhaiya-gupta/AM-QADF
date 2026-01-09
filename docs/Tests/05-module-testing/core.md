# Core Module - Testing Guide

## Test Files

- `test_entities.py` - VoxelData, Signal entities
- `test_value_objects.py` - VoxelCoordinates, QualityMetric
- `test_exceptions.py` - Exception hierarchy

## Key Tests

### Entity Tests
- Entity immutability
- Entity validation
- Entity serialization/deserialization

### Value Object Tests
- Value object validation
- Value object immutability
- Value object equality

### Exception Tests
- Exception message clarity
- Exception hierarchy
- Exception context preservation

## Coverage Target

**95%+** - Core module is fundamental

## Example Tests

```python
def test_voxel_data_immutability():
    """Test that VoxelData is immutable after creation."""
    
def test_value_object_validation():
    """Test value object validation rules."""
    
def test_exception_hierarchy():
    """Test exception inheritance hierarchy."""
```

## Running Core Module Tests

```bash
# Run all core module tests
pytest tests/unit/core/ -m unit

# Run specific test file
pytest tests/unit/core/test_entities.py
pytest tests/unit/core/test_value_objects.py
pytest tests/unit/core/test_exceptions.py

# Run with verbose output
pytest tests/unit/core/ -v

# Run with coverage
pytest tests/unit/core/ --cov=am_qadf.core --cov-report=term-missing
```

---

**Parent**: [Module Testing Guides](README.md)

