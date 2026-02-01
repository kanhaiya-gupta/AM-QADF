# Signal Mapping Module - Testing Guide

## ⭐ CRITICAL MODULE

The signal mapping module is a **core** part of the AM-QADF framework and is covered by both Python and C++ tests.

## Test Structure

**Python** (`tests/unit/python/signal_mapping/`, `tests/integration/bridge/test_signal_mapping_bridge.py`):
- Unit tests for interpolation methods (nearest neighbor, linear, IDW, KDE, RBF) and execution
- Bridge tests for the native (C++) signal mapping API

**C++** (`tests/unit/cpp/signal_mapping/`):
- `test_nearest_neighbor.cpp`, `test_linear_interpolation.cpp`, `test_idw_interpolation.cpp`
- `test_kde_interpolation.cpp`, `test_rbf_interpolation.cpp`, `test_interpolation_base.cpp`

## Key Tests

### Mathematical Correctness
- Interpolation accuracy
- Boundary condition handling
- Edge cases (empty data, single point)
- Numerical stability (NaN/Inf handling)

### Performance
- Vectorization effectiveness
- Memory efficiency
- C++ benchmarks: `tests/performance/cpp/benchmark_signal_mapping.cpp`

### Critical Test Cases (Python)

```python
def test_interpolation_accuracy():
    """Verify interpolation methods produce accurate results."""

def test_interpolation_with_empty_data():
    """Handle empty point clouds gracefully."""

def test_interpolation_boundary_conditions():
    """Test points outside voxel grid."""

def test_sequential_execution_correctness():
    """Verify sequential execution produces expected results."""
```

## Coverage Target

**95%+** for Python signal mapping code where applicable.

## Running Signal Mapping Module Tests

### Python

```bash
# All signal mapping unit tests
pytest tests/unit/python/signal_mapping/ -m unit -v

# Bridge tests (requires built native module)
pytest tests/integration/bridge/test_signal_mapping_bridge.py -m integration -v
```

### C++

From the build directory:
```bash
ctest --test-dir build -R signal_mapping --output-on-failure
```

## Related

- [Build Tests](../15-build-tests.md) - Building C++ tests
- [Running Tests](../14-running-tests.md) - Command reference
- [Integration Bridge](../04-test-categories/integration-tests.md) - Bridge tests

---

**Parent**: [Module Testing](README.md)
