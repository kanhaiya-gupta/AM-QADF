# AM-QADF Test Suite

Comprehensive test suite for the AM-QADF framework.

## Structure

```
tests/
├── unit/              # Unit tests (fast, isolated)
├── integration/       # Integration tests
├── performance/       # Performance/benchmark tests
├── fixtures/          # Test data and fixtures
├── property_based/    # Property-based tests (Hypothesis)
├── e2e/               # End-to-end tests
└── utils/             # Test utilities and helpers
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run specific test categories
```bash
# Unit tests only
pytest tests/unit -m unit

# Integration tests
pytest tests/integration -m integration

# Performance tests
pytest tests/performance -m performance

# End-to-end tests
pytest tests/e2e -m e2e
```

### Run with coverage
```bash
pytest --cov=src/am_qadf --cov-report=html
```

### Run specific test file
```bash
pytest tests/unit/core/test_entities.py
```

### Run specific test function
```bash
pytest tests/unit/core/test_entities.py::test_voxel_data_creation
```

## Test Markers

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.e2e` - End-to-end tests
- `@pytest.mark.performance` - Performance tests
- `@pytest.mark.slow` - Slow-running tests
- `@pytest.mark.requires_mongodb` - Tests requiring MongoDB
- `@pytest.mark.requires_spark` - Tests requiring Spark

## Coverage Requirements

- **Overall**: 80% minimum, 85% preferred
- **Critical paths** (signal_mapping): 95%
- **Core modules**: 90-95%

## Fixtures

Common fixtures are available in `conftest.py`:
- `sample_points_3d` - Sample 3D points
- `sample_voxel_grid` - Test voxel grid
- `mock_mongodb_client` - Mock MongoDB client
- `mock_query_client` - Mock query client
- And more...

See `conftest.py` for full list.

## Writing Tests

### Example Unit Test

```python
import pytest
import numpy as np
from am_qadf.voxelization import VoxelGrid

@pytest.mark.unit
def test_voxel_grid_creation(sample_voxel_grid):
    """Test voxel grid creation."""
    assert sample_voxel_grid is not None
    assert sample_voxel_grid.dimensions == (10, 10, 10)
    assert sample_voxel_grid.resolution == 1.0
```

### Example Integration Test

```python
@pytest.mark.integration
def test_signal_mapping_pipeline(mock_query_client, sample_voxel_grid):
    """Test complete signal mapping pipeline."""
    # Test implementation
    pass
```

## Performance Tests

Performance tests use `pytest-benchmark`:

```python
@pytest.mark.performance
def test_signal_mapping_performance(benchmark, large_point_cloud):
    """Benchmark signal mapping performance."""
    result = benchmark(map_signals_to_voxels, large_point_cloud)
    assert result is not None
```

## Property-Based Tests

Property-based tests use Hypothesis:

```python
from hypothesis import given, strategies as st

@pytest.mark.property_based
@given(points=st.lists(st.tuples(st.floats(), st.floats(), st.floats())))
def test_coordinate_transformation_invertible(points):
    """Test coordinate transformation is invertible."""
    # Test implementation
    pass
```

## CI/CD

Tests run automatically on:
- Push to main branch
- Pull requests
- Scheduled nightly runs

See `.github/workflows/tests.yml` for CI configuration.

## Contributing

When adding new tests:
1. Follow the existing structure
2. Use appropriate markers
3. Add fixtures to `conftest.py` if reusable
4. Update this README if adding new test categories



