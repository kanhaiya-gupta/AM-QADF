# Performance Testing

## Overview

Performance testing is implemented in **Python** (pytest-benchmark, regression suites) and **C++** (Google Benchmark). Python tests live under `tests/performance/python/`; C++ benchmarks under `tests/performance/cpp/`.

## Python Performance Tests

**Key benchmarks**:
- Signal mapping: large point sets → voxel grid
- Voxel fusion: multiple signals, large grids
- Interpolation methods: speed and accuracy comparison
- Regression: execution time and memory vs baselines

**Locations**:
- `tests/performance/python/benchmarks/` - benchmark_*.py
- `tests/performance/python/regression/` - test_*_regression.py, baseline JSON files

## C++ Performance Tests

**Key benchmarks** (Google Benchmark, run via ctest or executables):
- benchmark_fusion, benchmark_query, benchmark_signal_mapping
- benchmark_synchronization, benchmark_voxelization

**Location**: `tests/performance/cpp/`. Built when `ENABLE_BENCHMARKS=ON` (default if Google Benchmark is found or fetched).

## Performance Regression (Python)

- Track execution time over time
- Alert on performance degradation (e.g. vs baselines)
- Compare against baseline benchmarks

Baselines:
- `tests/performance/python/regression/performance_baselines.json`
- `tests/performance/python/regression/memory_baselines.json`

## Memory Profiling (Python)

- Memory usage for large datasets
- Memory leak detection
- Peak memory tracking

## Running Performance Tests

### Python

```bash
# Regression tests
pytest tests/performance/python/regression/ -m performance -v

# Benchmarks
pytest tests/performance/python/benchmarks/ -m benchmark --benchmark-only

# All performance Python tests
pytest tests/performance/python/ -m performance -v
```

### C++

From the build directory:
```bash
ctest --test-dir build -L benchmark --output-on-failure
```

See [15. Build Tests](15-build-tests.md) for building C++ benchmarks.

## Related

- [Performance Test Category](04-test-categories/performance-tests.md) - Category details
- [Build Tests](15-build-tests.md) - C++ build
- [Success Metrics](12-success-metrics.md) - Performance metrics

---

**Parent**: [Test Documentation](README.md)
