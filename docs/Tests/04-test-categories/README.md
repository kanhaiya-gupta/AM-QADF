# Test Categories

This directory documents each test category in the AM-QADF framework. Tests are implemented in **Python** (pytest) and **C++** (CMake/ctest), with **bridge** tests for the Python–C++ interface.

## Categories

1. **[Unit Tests](unit-tests.md)** - Python (`tests/unit/python/`) and C++ (`tests/unit/cpp/`) unit tests
2. **[Integration Tests](integration-tests.md)** - Python (`integration/python/`), bridge (`integration/bridge/`), and C++ (`integration/cpp/`)
3. **[Performance Tests](performance-tests.md)** - Python benchmarks/regression and C++ benchmarks
4. **[Property-Based Tests](property-based-tests.md)** - Hypothesis-based (Python)
5. **[E2E Tests](e2e-tests.md)** - End-to-end workflows (Python)

## Quick Reference

| Category       | Python location              | C++ location           | Run (Python)              | Run (C++)           |
|----------------|-----------------------------|------------------------|----------------------------|---------------------|
| Unit           | `tests/unit/python/`        | `tests/unit/cpp/`      | `pytest tests/unit/ -m unit` | `ctest --test-dir build` |
| Integration    | `integration/python/`, `integration/bridge/` | `integration/cpp/` | `pytest tests/integration/ -m integration` | same ctest |
| Performance    | `performance/python/`       | `performance/cpp/`     | `pytest tests/performance/ -m performance` | `ctest -L benchmark` |
| Property-Based | `tests/property_based/`     | —                      | `pytest tests/property_based/ -m property_based` | — |
| E2E            | `tests/e2e/`                | —                      | `pytest tests/e2e/ -m e2e` | — |

## Navigation

- **[Unit Tests](unit-tests.md)** - Fast, isolated Python and C++ tests
- **[Integration Tests](integration-tests.md)** - Workflows and Python–C++ bridge
- **[Performance Tests](performance-tests.md)** - Benchmarks and regression
- **[Property-Based Tests](property-based-tests.md)** - Hypothesis-based testing
- **[E2E Tests](e2e-tests.md)** - Full pipeline tests

---

**Parent**: [Test Documentation](../README.md)
