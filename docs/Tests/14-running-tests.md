# Running Tests - Quick Reference Guide

Tests are split into **Python** (pytest) and **C++** (CMake/ctest). Layout:

- **Python**: `tests/unit/python/`, `tests/integration/python/`, `tests/integration/bridge/` (NumPy–OpenVDB bridge), `tests/performance/python/`, `tests/e2e/`, `tests/property_based/`, `tests/utils/`, `tests/fixtures/`
- **C++**: `tests/unit/cpp/`, `tests/integration/cpp/`, `tests/performance/cpp/`

## Quick Commands

### Run All Python Tests
```bash
# Run entire Python test suite (unit, integration, bridge, e2e, property_based, utils, fixtures)
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run with coverage (Python only)
pytest tests/ --cov=am_qadf --cov-report=html
```

### Run C++ Tests
C++ tests are **built with the full project** when `BUILD_TESTS=ON` (default). From project root:

```bash
# Configure and build (tests included when BUILD_TESTS=ON)
cmake -B build -DBUILD_TESTS=ON
cmake --build build

# Run all C++ tests (unit + integration; includes C++ benchmarks when ENABLE_BENCHMARKS=ON, default)
ctest --test-dir build --output-on-failure

# Run only C++ unit tests (if ctest labels are set)
ctest --test-dir build -R "unit" --output-on-failure

# Run only C++ performance benchmarks (built by default when Google Benchmark is found)
ctest --test-dir build -L benchmark --output-on-failure
```

**C++ performance benchmarks** (`tests/performance/cpp/`): Built by default when **`ENABLE_BENCHMARKS=ON`** (default). If Google Benchmark is not installed, CMake fetches it from GitHub (FetchContent) at configure time. To use a system/conda install instead, install the library (e.g. build from [google/benchmark](https://github.com/google/benchmark)) and ensure `find_package(benchmark)` can find it; or disable benchmarks with `-DENABLE_BENCHMARKS=OFF`. Then `ctest --test-dir build --output-on-failure` runs unit, integration, and benchmark tests; use `ctest --test-dir build -L benchmark` to run only the C++ benchmarks. You can also run benchmark executables by hand (e.g. `./build/tests/performance/cpp/bench_voxelization`) for detailed output. See [tests/performance/cpp/README.md](../../tests/performance/cpp/README.md).

To build **without** tests: `cmake -B build -DBUILD_TESTS=OFF`. To build **only** a test target: `cmake --build build --target test_voxelization_cpp` (or another test executable name). See [Build Tests](15-build-tests.md) for full options.

**When C++ test shows "Subprocess killed" with no details:** run the test executable directly from the build directory so you see which test was running and any crash output:

```bash
cd build
# Executable path (from ctest -V output): tests/unit/cpp/<module>/test_<module>_cpp
./tests/unit/cpp/voxelization/test_voxelization_cpp --reporter console 2>&1 | tee voxelization.log
```

The last lines of `voxelization.log` (or the terminal) show the test that was running when the process died. To isolate which test causes "Killed", run tests by section (name filter). Run each line in turn; when one gets Killed, the failing test is in that section:

```bash
./tests/unit/cpp/voxelization/test_voxelization_cpp "OpenVDB: initialize" --reporter console
./tests/unit/cpp/voxelization/test_voxelization_cpp "UniformVoxelGrid:*" --reporter console
./tests/unit/cpp/voxelization/test_voxelization_cpp "MultiResolutionVoxelGrid:*" --reporter console
./tests/unit/cpp/voxelization/test_voxelization_cpp "AdaptiveResolutionVoxelGrid:*" --reporter console
./tests/unit/cpp/voxelization/test_voxelization_cpp "UnifiedGridFactory:*" --reporter console
./tests/unit/cpp/voxelization/test_voxelization_cpp "HatchingVoxelizer:*" --reporter console
./tests/unit/cpp/voxelization/test_voxelization_cpp "STLVoxelizer:*" --reporter console
./tests/unit/cpp/voxelization/test_voxelization_cpp "VoxelGridFactory:*" --reporter console
```

Alternatively, run CTest with verbose output: `ctest --test-dir build -R test_voxelization_cpp -V --output-on-failure`. To list all test names: `./tests/unit/cpp/voxelization/test_voxelization_cpp --list-tests`.

**MongoDB (C++ io tests):** Tests that use MongoDB (e.g. `test_io_cpp` / `test_mongodb_writer`) read the URI from the environment when set. Use the project **development.env** so authenticated MongoDB is used:

```bash
# From project root (WSL/Linux)
set -a && source development.env && set +a
ctest --test-dir build --output-on-failure
```

The test uses `TEST_MONGODB_URL` or `MONGODB_URL` from the environment (see `development.env`: `TEST_MONGODB_URL=mongodb://admin:password@localhost:27017/am_qadf_test`). If unset, it falls back to `mongodb://localhost:27017` and skips with a clear message when auth is required.

**Recommended for C++ development:** use system compiler and system libs to avoid ABI issues (e.g. `mkdir -p build && cd build`, then `CC=gcc CXX=g++ cmake -DUSE_SYSTEM_LIBS=ON ..`, `cmake --build . --config Release -j $(nproc)`). See [Build Tests](15-build-tests.md#recommended-c-development-system-compiler--system-libs) for full steps.

### Run by Category

```bash
# Python unit tests (fast; under tests/unit/python/)
pytest tests/unit/ -m unit
# or explicitly:
pytest tests/unit/python/ -m unit

# Python integration tests (under tests/integration/python/ and root workflow tests)
pytest tests/integration/ -m integration

# Python–C++ bridge tests (under tests/integration/bridge/)
pytest tests/integration/bridge/ -m integration

# Python performance tests and benchmarks
pytest tests/performance/ -m performance

# Property-based tests
pytest tests/property_based/ -m property_based

# E2E tests (slow)
pytest tests/e2e/ -m e2e
```

### Run Python Tests by Module

```bash
# Core module
pytest tests/unit/python/core/

# Query module
pytest tests/unit/python/query/

# Voxelization module
pytest tests/unit/python/voxelization/

# Signal mapping module (critical)
pytest tests/unit/python/signal_mapping/

# Synchronization module
pytest tests/unit/python/synchronization/

# Correction module
pytest tests/unit/python/correction/

# Processing module
pytest tests/unit/python/processing/

# Fusion module
pytest tests/unit/python/fusion/

# Quality module
pytest tests/unit/python/quality/

# Analytics module
pytest tests/unit/python/analytics/

# Anomaly detection module
pytest tests/unit/python/anomaly_detection/

# Visualization module
pytest tests/unit/python/visualization/

# Voxel domain module
pytest tests/unit/python/voxel_domain/

# Bridge (am_qadf_native NumPy–OpenVDB: numpy_to_openvdb, openvdb_to_numpy).
pytest tests/unit/python/bridge/
```

## Advanced Usage

### Run Specific Test File
```bash
pytest tests/unit/python/voxelization/test_uniform_resolution.py

# NumPy–OpenVDB bridge unit tests
pytest tests/unit/python/bridge/test_numpy_openvdb_bridge.py
```

### Run Specific Test Function
```bash
pytest tests/unit/python/voxelization/test_uniform_resolution.py::test_voxel_grid_creation
```

### Run Tests Matching Pattern
```bash
# Run all tests with "interpolation" in name
pytest tests/ -k "interpolation"

# Run all tests with "fusion" in name
pytest tests/ -k "fusion"
```

### Run with Markers
```bash
# Run only slow tests
pytest tests/ -m slow

# Run unit tests excluding slow ones
pytest tests/unit/python/ -m "unit and not slow"

# Run E2E and integration tests
pytest tests/ -m "e2e or integration"

# Run Python–C++ bridge tests only
pytest tests/integration/bridge/ -m integration
```

### Parallel Execution
```bash
# Auto-detect CPU count
pytest tests/ -n auto

# Use specific number of workers
pytest tests/ -n 4
```

### Coverage Reports
```bash
# Terminal report
pytest tests/ --cov=am_qadf --cov-report=term-missing

# HTML report
pytest tests/ --cov=am_qadf --cov-report=html
# Open htmlcov/index.html in browser

# XML report (for CI/CD)
pytest tests/ --cov=am_qadf --cov-report=xml
```

### Performance tests (Python)
All Python performance tests live under `tests/performance/python/`:

| Path | Contents |
|------|----------|
| `tests/performance/python/benchmarks/` | pytest-benchmark suites (interpolation, signal mapping, voxel fusion) |
| `tests/performance/python/regression/` | Performance and memory regression tests vs baselines |
| `tests/performance/python/signal_mapping/` | Signal-mapping and RBF performance tests |
| `tests/performance/python/test_*.py` | Monitoring and streaming performance tests |

**Benchmarks** (require **pytest-benchmark**; install with `pip install -e ".[dev]"` or `pip install pytest-benchmark`):

```bash
# Run all performance tests (by marker)
pytest tests/performance/ -m performance

# Run only benchmark suites (timed, no assertions)
pytest tests/performance/python/benchmarks/ --benchmark-only

# Run with benchmark report / autosave
pytest tests/performance/python/benchmarks/ --benchmark-only --benchmark-autosave

# Run performance regression tests (vs baselines)
pytest tests/performance/python/regression/

# Run signal-mapping performance tests
pytest tests/performance/python/signal_mapping/

# Run monitoring / streaming performance tests
pytest tests/performance/python/test_monitoring_performance.py tests/performance/python/test_streaming_performance.py
```

### Verbose Output
```bash
# Verbose output
pytest tests/ -v

# Very verbose (show print statements)
pytest tests/ -vv -s

# Show local variables on failure
pytest tests/ -vv -l
```

### Stop on First Failure
```bash
pytest tests/ -x
```

### Run Last Failed Tests
```bash
# Run only tests that failed last time
pytest tests/ --lf

# Run failed tests first, then rest
pytest tests/ --ff
```

## Common Test Scenarios

### Development Workflow
```bash
# Run tests for module you're working on (Python)
pytest tests/unit/python/signal_mapping/ -v

# Run with coverage to see what's missing
pytest tests/unit/python/signal_mapping/ --cov=am_qadf.signal_mapping --cov-report=term-missing

# Run specific test while developing
pytest tests/unit/python/signal_mapping/methods/test_nearest_neighbor.py -k "interpolation" -vv
```

### Pre-Commit Checks
```bash
# Run fast tests only
pytest tests/unit/ -m "unit and not slow"

# Check coverage threshold
pytest tests/unit/ --cov=am_qadf --cov-fail-under=80
```

### CI/CD Pipeline

GitHub Actions (`.github/workflows/ci.yml`, `pr.yml`, `nightly.yml`, `release.yml`) run Python tests (pytest) and a dedicated **C++ job** that builds with CMake (conda-forge deps: OpenVDB, ITK, libmongocxx, pybind11) and runs `ctest`.

```bash
# Full Python test suite with coverage
pytest tests/ --cov=am_qadf --cov-report=xml --cov-report=html

# Python unit tests
pytest tests/unit/python/ -m unit --cov=am_qadf

# Python integration and bridge tests
pytest tests/integration/ -m integration

# C++ tests (after cmake --build build)
ctest --test-dir build --output-on-failure

# Python performance benchmarks (nightly)
pytest tests/performance/python/benchmarks/ --benchmark-only
```

### Debugging Failed Tests
```bash
# Run with debug output
pytest tests/ -vv -s --tb=short

# Run with pdb debugger on failure
pytest tests/ --pdb

# Run with pdb on all failures
pytest tests/ --pdb --maxfail=1
```

## Test Execution Times

| Category | Typical Time | Command |
|----------|--------------|---------|
| Python unit tests | < 1 minute | `pytest tests/unit/python/ -m unit` |
| Python integration tests | 1-5 minutes | `pytest tests/integration/ -m integration` |
| Python–C++ bridge tests | 1-3 minutes | `pytest tests/integration/bridge/ -m integration` |
| C++ unit/integration tests | 1-5 minutes | `ctest --test-dir build --output-on-failure` |
| Python performance benchmarks | 5-15 minutes | `pytest tests/performance/python/benchmarks/ --benchmark-only` |
| Property-based tests | 2-10 minutes | `pytest tests/property_based/ -m property_based` |
| E2E tests | 5-30 minutes | `pytest tests/e2e/ -m e2e` |
| **Full Python suite** | **10-60 minutes** | `pytest tests/` |

## Related

- [Build Tests](15-build-tests.md) - Building Python and C++ test infrastructure
- [Test Structure](03-test-structure.md) - Directory layout (Python vs C++)
- [Test Categories](04-test-categories/) - Category-specific running instructions
- [Module Testing Guides](05-module-testing/) - Module-specific running instructions
- [Tools](13-tools.md) - Testing tools and installation

---

**Parent**: [Test Documentation](README.md)

