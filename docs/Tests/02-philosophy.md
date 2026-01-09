# Testing Philosophy

## Test-Driven Development (TDD)

### Principles
- Write tests before implementation for critical paths
- Use tests to drive design decisions
- Refactor with confidence using test safety net

### Benefits
- Better design (forces thinking about interfaces)
- Early bug detection
- Living documentation
- Confidence in refactoring

## Behavior-Driven Development (BDD)

### Principles
- Use descriptive test names that explain behavior
- Structure tests to read like specifications
- Document expected behavior through tests

### Example
```python
def test_voxel_grid_should_preserve_signal_values_when_finalized():
    """Test that signal values are preserved correctly after finalization."""
    # Arrange
    grid = VoxelGrid(...)
    grid.add_point(0, 0, 0, signals={'power': 200.0})
    
    # Act
    grid.finalize()
    
    # Assert
    assert grid.get_signal('power', 0, 0, 0) == 200.0
```

## Property-Based Testing

### Principles
- Use Hypothesis for generating test cases
- Test mathematical properties (commutativity, associativity)
- Validate invariants across input ranges

### Example Properties
- **Invertibility**: Transform then inverse = identity
- **Commutativity**: Order doesn't matter
- **Associativity**: Chaining transformations
- **Idempotency**: Same operation twice = same result
- **Boundedness**: Output within input range

### Example
```python
from hypothesis import given, strategies as st

@given(points=st.lists(st.tuples(st.floats(), st.floats(), st.floats())))
def test_coordinate_transformation_invertible(points):
    """Test that coordinate transformations are invertible."""
    transformer = CoordinateSystemTransformer()
    points_array = np.array(points)
    
    transformed = transformer.transform(points_array, 'build', 'machine')
    back_transformed = transformer.transform(transformed, 'machine', 'build')
    
    assert np.allclose(points_array, back_transformed, atol=1e-6)
```

## Test Pyramid

```
        /\
       /E2E\          Few, slow, comprehensive
      /------\
     /Integration\    Moderate number, moderate speed
    /------------\
   /   Unit Tests \   Many, fast, isolated
  /----------------\
```

### Distribution
- **Unit Tests**: 80-90% of tests
- **Integration Tests**: 10-15% of tests
- **E2E Tests**: 5-10% of tests

## Testing Principles Summary

1. **Fast**: Tests should run quickly
2. **Independent**: Tests should not depend on each other
3. **Repeatable**: Tests should produce consistent results
4. **Self-validating**: Tests should clearly pass or fail
5. **Timely**: Tests should be written close to implementation

---

**Related**: [Overview](01-overview.md) | [Test Categories](04-test-categories/)

