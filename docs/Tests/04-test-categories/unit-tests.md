# Unit Tests

## Purpose

Test individual functions and classes in isolation. Unit tests exist in both **Python** and **C++**.

## Layout

- **Python**: `tests/unit/python/` — one subdirectory per module (e.g. `voxelization/`, `signal_mapping/`, `query/`).
- **C++**: `tests/unit/cpp/` — one subdirectory per area (e.g. `voxelization/`, `signal_mapping/`, `synchronization/`, `correction/`, `processing/`, `query/`, `io/`, `fusion/`).

## Characteristics

- **Speed**: Typically &lt; 1 second per test
- **Scope**: Single module/class
- **Dependencies**: Mocked or in-memory; no live DB/network for Python; C++ may use test fixtures only

## Key Areas

- Core domain entities and value objects
- Interpolation methods (correctness)
- Coordinate and grid transformations
- Fusion and signal processing
- Quality metrics and utilities

## Running Unit Tests

### Python

```bash
# All unit tests
pytest tests/unit/ -m unit -v

# Specific module
pytest tests/unit/python/voxelization/ -v
pytest tests/unit/python/signal_mapping/ -v

# With coverage
pytest tests/unit/ -m unit --cov=src/am_qadf --cov-report=term-missing
```

### C++

C++ unit tests are built with the project when `BUILD_TESTS=ON`. From the build directory:

```bash
# After: cmake -DBUILD_TESTS=ON .. && cmake --build .
ctest --test-dir build --output-on-failure

# Run only unit-style tests (exact labels depend on CMake setup)
ctest --test-dir build -R "unit|voxelization|signal_mapping" --output-on-failure
```

See [15. Build Tests](../15-build-tests.md) for configuring and building C++ tests.

## Example (Python)

```python
@pytest.mark.unit
def test_nearest_neighbor_interpolation():
    """Test nearest neighbor interpolation."""
    points = np.array([[0, 0, 0], [1, 1, 1]])
    values = np.array([1.0, 2.0])
    voxel_grid = create_test_voxel_grid()
    result = NearestNeighborInterpolation().interpolate(points, values, voxel_grid)
    assert result.shape == voxel_grid.shape
    assert np.allclose(result[0, 0, 0], 1.0)
```

## Best Practices

1. **Isolation**: Each test independent; use mocks for I/O and external services.
2. **Fast**: Keep unit tests under ~1s each.
3. **Clear**: Descriptive names and AAA (Arrange–Act–Assert) where applicable.
4. **Deterministic**: No flakiness; same result every run.

## Related

- [Module Testing Guides](../05-module-testing/) - Per-module unit test notes
- [Build Tests](../15-build-tests.md) - C++ build and ctest
- [Running Tests](../14-running-tests.md) - All run commands

---

**Parent**: [Test Categories](README.md)
