# Voxel Domain Module - Testing Guide

## Test Files

- `test_voxel_domain_client.py` - Main orchestrator
- `test_voxel_storage.py` - MongoDB/GridFS storage

## Key Tests

### Voxel Domain Client
- End-to-end workflow
- Multi-source signal mapping
- Coordinate system handling
- Error handling

### Storage
- Storage/retrieval correctness
- GridFS operations
- Metadata management
- Performance with large grids

## Coverage Target

**85%+**

## Example Tests

```python
def test_voxel_domain_end_to_end():
    """Test complete voxel domain workflow."""
    
def test_storage_retrieval():
    """Test voxel grid storage and retrieval."""
    
def test_large_grid_performance():
    """Test performance with large grids."""
```

## Running Voxel Domain Module Tests

```bash
# Run all voxel domain tests
pytest tests/unit/voxel_domain/ -m unit

# Run specific test files
pytest tests/unit/voxel_domain/test_voxel_domain_client.py
pytest tests/unit/voxel_domain/test_voxel_storage.py

# Run with coverage
pytest tests/unit/voxel_domain/ --cov=am_qadf.voxel_domain --cov-report=term-missing

# Run integration tests
pytest tests/integration/test_voxel_domain_workflow.py -m integration

# Run E2E tests
pytest tests/e2e/test_complete_pipeline.py -m e2e
```

## Related

- [E2E Tests](../04-test-categories/e2e-tests.md) - Complete pipeline tests

---

**Parent**: [Module Testing Guides](README.md)

