# Unit Tests

## Purpose

Test individual functions/classes in isolation.

## Characteristics

- **Speed**: < 1 second per test
- **Scope**: Single module/class
- **Dependencies**: Mocked external dependencies
- **Count**: 141+ test files

## Key Areas

- Core domain entities and value objects
- Interpolation methods (mathematical correctness)
- Coordinate transformations
- Fusion algorithms
- Quality metrics calculations
- Statistical functions

## Example Structure

```python
@pytest.mark.unit
def test_nearest_neighbor_interpolation():
    """Test nearest neighbor interpolation."""
    # Arrange
    points = np.array([[0, 0, 0], [1, 1, 1]])
    values = np.array([1.0, 2.0])
    voxel_grid = create_test_voxel_grid()
    
    # Act
    result = NearestNeighborInterpolation().interpolate(
        points, values, voxel_grid
    )
    
    # Assert
    assert result.shape == voxel_grid.shape
    assert np.allclose(result[0, 0, 0], 1.0)
```

## Test Organization

Unit tests are organized by module:
- `tests/unit/core/` - Core module tests
- `tests/unit/query/` - Query module tests
- `tests/unit/voxelization/` - Voxelization tests
- etc.

## Best Practices

1. **Isolation**: Each test should be independent
2. **Mocking**: Use mocks for external dependencies
3. **Fast**: Tests should run quickly (< 1s)
4. **Clear**: Use descriptive test names
5. **AAA Pattern**: Arrange-Act-Assert

## Running Unit Tests

```bash
# Run all unit tests
pytest tests/unit/ -m unit

# Run specific module
pytest tests/unit/voxelization/

# Run with coverage
pytest tests/unit/ --cov=am_qadf.voxelization
```

## Related

- [Module Testing Guides](../05-module-testing/) - Module-specific unit test guides
- [Best Practices](../11-best-practices.md) - Testing best practices

---

**Parent**: [Test Categories](README.md)

