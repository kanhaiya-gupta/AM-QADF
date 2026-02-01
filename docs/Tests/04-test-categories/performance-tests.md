# Performance Tests

## Purpose

Validate performance claims and detect regressions. Performance testing is implemented in **Python** (pytest-benchmark, regression suites) and **C++** (Google Benchmark, run via ctest).

## Layout

- **Python**: `tests/performance/python/`
  - `benchmarks/` — benchmark_*.py (signal mapping, voxel fusion, interpolation methods)
  - `regression/` — test_performance_regression.py, test_memory_regression.py, baseline JSON files
  - Other performance-related tests (e.g. monitoring, streaming)
- **C++**: `tests/performance/cpp/`
  - benchmark_*.cpp (fusion, query, signal_mapping, synchronization, voxelization)
  - Built when `ENABLE_BENCHMARKS=ON` (default if Google Benchmark is found or fetched)

## Characteristics

- **Speed**: Variable; benchmarks can be slow
- **Scope**: Performance-critical code paths
- **Dependencies**: Realistic or large-enough test data; C++ needs a full build

## Key Benchmarks

- Signal mapping: large point sets → voxel grid
- Voxel fusion: multiple signals, large grids
- Interpolation methods: speed and accuracy comparison
- C++: fusion, query, signal mapping, synchronization, voxelization

## Running Performance Tests

### Python

```bash
# Regression tests
pytest tests/performance/python/regression/ -m performance -v

# Benchmarks (pytest-benchmark)
pytest tests/performance/python/benchmarks/ -m benchmark --benchmark-only

# All performance-related Python tests
pytest tests/performance/python/ -m performance -v
```

### C++

From the build directory (build with `ENABLE_BENCHMARKS=ON`):

```bash
# Run benchmark tests
ctest --test-dir build -L benchmark --output-on-failure

# Or run benchmark executables directly from build/tests/performance/cpp/
```

See [15. Build Tests](../15-build-tests.md) for enabling and building C++ benchmarks.

## Baseline Management (Python)

Baselines are stored in:
- `tests/performance/python/regression/performance_baselines.json`
- `tests/performance/python/regression/memory_baselines.json`

## Example (Python)

```python
@pytest.mark.benchmark
@pytest.mark.performance
def test_signal_mapping_performance(benchmark):
    """Benchmark signal mapping."""
    large_dataset = create_large_test_dataset()
    result = benchmark(map_signals_to_voxels, large_dataset, method='nearest_neighbor')
    assert result is not None
```

## Related

- [Performance Testing Strategy](../09-performance.md) - Overall strategy
- [Build Tests](../15-build-tests.md) - C++ benchmarks build
- [Running Tests](../14-running-tests.md) - Command reference

---

**Parent**: [Test Categories](README.md)
