# Test Structure

## Directory Organization

Tests are split into **Python** (pytest) and **C++** (CMake/ctest). Python tests live under `unit/python/`, `integration/python/`, `integration/bridge/`, and `performance/python/`; C++ tests under `unit/cpp/`, `integration/cpp/`, and `performance/cpp/`.

```
tests/
├── __init__.py
├── conftest.py                    # Shared pytest fixtures
│
├── unit/                          # Unit tests
│   ├── python/                    # Python unit tests (pytest)
│   │   ├── analytics/
│   │   ├── correction/
│   │   ├── fusion/
│   │   ├── processing/
│   │   ├── query/
│   │   ├── signal_mapping/
│   │   ├── synchronization/
│   │   ├── voxel_domain/
│   │   ├── voxelization/
│   │   └── ... (other modules)
│   └── cpp/                       # C++ unit tests (CMake/ctest)
│       ├── correction/
│       ├── fusion/
│       ├── io/
│       ├── processing/
│       ├── query/
│       ├── signal_mapping/
│       ├── synchronization/
│       └── voxelization/
│
├── integration/                   # Integration tests
│   ├── python/                   # Python integration (pytest)
│   │   ├── analytics/, deployment/, monitoring/, spc/, streaming/, validation/
│   │   ├── test_analytics_workflow.py
│   │   ├── test_end_to_end_workflow.py
│   │   ├── test_fusion_workflow.py
│   │   ├── test_signal_mapping_pipeline.py
│   │   ├── test_voxel_domain_workflow.py
│   │   └── ...
│   ├── bridge/                   # Python–C++ bridge tests (pytest)
│   │   ├── test_correction_bridge.py
│   │   ├── test_processing_bridge.py
│   │   ├── test_python_cpp_bridge.py
│   │   ├── test_voxelization_bridge.py
│   │   └── ...
│   └── cpp/                      # C++ integration (CMake/ctest)
│       ├── test_correction_pipeline.cpp
│       ├── test_fusion_pipeline.cpp
│       ├── test_signal_mapping_pipeline.cpp
│       └── ...
│
├── performance/                   # Performance tests
│   ├── python/                   # Python benchmarks (pytest)
│   │   ├── benchmarks/           # benchmark_*.py
│   │   └── regression/           # test_*_regression.py, baselines
│   └── cpp/                      # C++ benchmarks (Google Benchmark, ctest)
│       ├── benchmark_fusion.cpp
│       ├── benchmark_signal_mapping.cpp
│       ├── benchmark_voxelization.cpp
│       └── ...
│
├── fixtures/                      # Test data and mocks
│   ├── mocks/
│   ├── point_clouds/, signals/, voxel_data/
│   ├── openvdb/, spc/, streaming/, validation/
│   └── ...
│
├── property_based/                # Property-based tests (pytest)
├── e2e/                           # End-to-end tests (pytest)
└── utils/                         # Test utilities (pytest)
```

## Naming Conventions

### Python
- **Files**: `test_<module_or_feature>.py`
- **Functions**: `test_<functionality>_<condition>`
- **Classes**: `Test<ClassName>`

### C++
- **Files**: `test_<module>.cpp` or `benchmark_<area>.cpp`
- **Test cases**: Catch2 `TEST_CASE` / `SECTION` names

## Running by Layer

| Layer        | Python command                    | C++ (after build)           |
|-------------|------------------------------------|-----------------------------|
| Unit        | `pytest tests/unit/python/ -m unit` | `ctest --test-dir build -R unit` (or run executables) |
| Integration | `pytest tests/integration/python/ tests/integration/bridge/ -m integration` | `ctest --test-dir build -R integration` |
| Performance | `pytest tests/performance/python/ -m performance` | `ctest --test-dir build -L benchmark` |

See [14-running-tests.md](14-running-tests.md) and [15-build-tests.md](15-build-tests.md) for full commands.

## Related Documentation

- [Module Testing Guides](05-module-testing/) - Per-module structure
- [Test Categories](04-test-categories/) - Category details
- [Infrastructure](06-infrastructure.md) - Fixtures and configuration
- [Build Tests](15-build-tests.md) - C++ build and ctest

---

**Related**: [Overview](01-overview.md) | [Test Categories](04-test-categories/)
