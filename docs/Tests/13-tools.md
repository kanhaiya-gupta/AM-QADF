# Testing Tools and Libraries

## Python Testing (pytest)

### Required Tools
- **pytest** - Test framework
- **pytest-cov** - Coverage reporting
- **pytest-benchmark** - Performance benchmarking
- **pytest-mock** - Mocking utilities
- **hypothesis** - Property-based testing
- **numpy.testing** - NumPy-specific assertions
- **pytest-xdist** - Parallel test execution

### Optional Tools
- **pytest-timeout** - Test timeout management
- **pytest-asyncio** - Async test support
- **pytest-html** - HTML test reports
- **memory_profiler** - Memory profiling
- **line_profiler** - Line-by-line profiling

### Installation

```bash
# Core testing tools
pip install pytest pytest-cov pytest-benchmark pytest-mock

# Property-based testing
pip install hypothesis

# Parallel execution
pip install pytest-xdist

# Optional tools
pip install pytest-timeout pytest-html memory_profiler
```

## C++ Testing (CMake + CTest)

- **CMake** (3.15+) - Configure and build C++ tests
- **Catch2** - C++ unit/integration test framework (included in project)
- **Google Benchmark** - C++ performance benchmarks (optional; fetched if not found when `ENABLE_BENCHMARKS=ON`)
- **CTest** - Run C++ tests (part of CMake)

C++ tests are built with the project when `BUILD_TESTS=ON`. See [15. Build Tests](15-build-tests.md) for setup.

## Usage Examples

### Pytest (Python)
```bash
pytest tests/ -v
pytest tests/unit/ -m unit
pytest tests/unit/python/voxelization/ -v
pytest tests/integration/ -m integration
pytest tests/ -k "test_voxel"
```

### Coverage (Python)
```bash
pytest tests/ --cov=src/am_qadf --cov-report=html
```

### Benchmarking (Python)
```bash
pytest tests/performance/python/benchmarks/ --benchmark-only
pytest tests/performance/python/ -m performance -v
```

### Parallel Execution (Python)
```bash
pytest tests/ -n auto
pytest tests/ -n 4
```

### C++ (CTest)
```bash
# From project root after building
ctest --test-dir build --output-on-failure

# Only benchmarks
ctest --test-dir build -L benchmark --output-on-failure

# Filter by name
ctest --test-dir build -R voxelization --output-on-failure
```

## Related

- [Infrastructure](06-infrastructure.md) - Pytest and C++ configuration
- [Build Tests](15-build-tests.md) - C++ build and ctest
- [Running Tests](14-running-tests.md) - Full command reference
- [Best Practices](11-best-practices.md) - Tool usage best practices

---

**Parent**: [Test Documentation](README.md)
