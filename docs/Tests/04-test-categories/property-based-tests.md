# Property-Based Tests

## Purpose

Test mathematical properties and invariants.

## Characteristics

- **Speed**: Variable (generates many test cases)
- **Scope**: Mathematical operations
- **Dependencies**: Hypothesis library
- **Count**: 4 test files

## Test Files

- `test_voxel_grid_properties.py` - Voxel grid mathematical properties
- `test_interpolation_properties.py` - Interpolation properties
- `test_fusion_properties.py` - Fusion properties
- `test_coordinate_transformations.py` - Transformation properties

## Key Areas

- Coordinate transformation properties (invertibility, associativity)
- Interpolation properties (boundary conditions, monotonicity)
- Fusion properties (commutativity, idempotency)
- Statistical properties

## Properties Tested

### Invertibility
Transform then inverse = identity

### Commutativity
Order doesn't matter (for applicable operations)

### Associativity
Chaining transformations

### Idempotency
Same operation twice = same result

### Boundedness
Output within input range

### Linearity
Scaling input scales output

## Example Structure

```python
from hypothesis import given, strategies as st

@pytest.mark.property_based
@given(
    points=st.lists(
        st.tuples(st.floats(), st.floats(), st.floats()),
        min_size=3, max_size=100
    )
)
def test_coordinate_transformation_invertible(points):
    """Test that coordinate transformations are invertible."""
    # Arrange
    transformer = CoordinateSystemTransformer()
    points_array = np.array(points)
    
    # Act
    transformed = transformer.transform(points_array, 'build', 'machine')
    back_transformed = transformer.transform(transformed, 'machine', 'build')
    
    # Assert
    assert np.allclose(points_array, back_transformed, atol=1e-6)
```

## Running Property-Based Tests

```bash
# Run all property-based tests
pytest tests/property_based/ -m property_based

# Run with verbose output
pytest tests/property_based/ -v -m property_based
```

## Hypothesis Configuration

Property-based tests use Hypothesis with:
- Multiple examples per test (20-50)
- Edge case handling with `assume()`
- Numerical tolerance checks with `np.allclose()`

## Related

- [Testing Philosophy](../02-philosophy.md) - Property-based testing philosophy
- [Best Practices](../11-best-practices.md) - Property-based testing best practices

---

**Parent**: [Test Categories](README.md)

