# Test Infrastructure

## Python: Pytest Configuration

**`pytest.ini`** (or equivalent in `pyproject.toml`):
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --strict-markers
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    performance: Performance/benchmark tests
    slow: Slow-running tests
    requires_mongodb: Tests requiring MongoDB
    property_based: Property-based tests
    bridge: Python–C++ bridge tests
    regression: Performance regression tests
```

Coverage and other addopts may be set in CI or locally (e.g. `--cov=src/am_qadf`, `--cov-report=html`).

## Shared Fixtures (`conftest.py`)

Common fixtures available to Python tests:

- `sample_points` - Sample 3D points
- `sample_voxel_grid` - Test voxel grid
- `mock_mongodb_client` - Mock MongoDB client
- `mock_query_client` - Mock query client

## Test Utilities

**`tests/utils/`**:
- `test_helpers.py` - Common test utilities
- `assertions.py` - Custom assertion functions

## Fixtures

See [Test Data Management](10-test-data.md) for fixture documentation.

## Mock Objects

**`tests/fixtures/mocks/`**:
- `mock_mongodb.py` - MongoDB mock client
- `mock_query_clients.py` - Query client mocks

## C++ Test Infrastructure

C++ tests are built with **CMake** when `BUILD_TESTS=ON` (default). They use **Catch2** for unit/integration and **Google Benchmark** for performance (when `ENABLE_BENCHMARKS=ON`).

- **Unit/integration**: Executables under `tests/unit/cpp/` and `tests/integration/cpp/`; run via `ctest --test-dir build --output-on-failure`.
- **Benchmarks**: Executables under `tests/performance/cpp/`; run via `ctest --test-dir build -L benchmark` or by running the benchmark binaries directly.

See [15. Build Tests](15-build-tests.md) for configuring and building C++ tests.

## Related

- [Test Data Management](10-test-data.md) - Fixtures and test data
- [Build Tests](15-build-tests.md) - C++ build and ctest
- [Best Practices](11-best-practices.md) - Infrastructure best practices

---

**Parent**: [Test Documentation](README.md)
