# Test Data Management

## Test Data Sources

- **Synthetic Data**: Generated programmatically
- **Fixture Data**: Pre-computed test cases
- **Realistic Data**: Anonymized real-world samples

## Test Data Organization

```
tests/fixtures/
├── voxel_data/          # Voxel grid fixtures
│   ├── small_voxel_grid.pkl
│   ├── medium_voxel_grid.pkl
│   └── large_voxel_grid.pkl
├── point_clouds/       # Point cloud data
│   ├── hatching_paths.json
│   ├── laser_points.json
│   └── ct_points.json
├── signals/            # Signal arrays
│   └── sample_signals.npz
└── mocks/              # Mock objects
    ├── mock_mongodb.py
    └── mock_query_clients.py
```

## Test Data Generation

- Use factories for generating test data
- Parameterized tests for different data sizes
- Property-based generation for edge cases

## Generating Fixtures

```bash
# Generate all fixtures
python scripts/generate_test_fixtures.py
```

## Using Fixtures

### Pytest Fixtures
```python
def test_something(small_voxel_grid):
    """Test using small voxel grid fixture."""
    assert small_voxel_grid is not None
```

### Direct Loading
```python
from tests.fixtures.voxel_data import load_small_voxel_grid

grid = load_small_voxel_grid()
```

## Related

- [Fixtures README](../../tests/fixtures/README.md) - Detailed fixture documentation
- [Infrastructure](06-infrastructure.md) - Fixture infrastructure

---

**Parent**: [Test Documentation](README.md)

