# Testing Overview

## Goals

The AM-QADF testing framework aims to ensure:

- **Reliability**: Framework correctness and stability
- **Maintainability**: Confident refactoring
- **Documentation**: Tests as executable documentation
- **Performance**: Validation of performance-critical code (Python and C++)
- **Compatibility**: Cross-platform and version compatibility

## Testing Principles

1. **Test Pyramid**: Many unit tests, fewer integration tests, minimal E2E tests
2. **AAA Pattern**: Arrange-Act-Assert for clarity
3. **FIRST Principles**: Fast, Independent, Repeatable, Self-validating, Timely
4. **Isolation**: Tests should not depend on external services (use mocks/fixtures)
5. **Deterministic**: Tests must produce consistent results
6. **Fast Feedback**: Unit tests run in seconds

## Python vs C++ Tests

| Aspect       | Python                         | C++                          |
|-------------|---------------------------------|------------------------------|
| **Framework** | pytest                        | CMake + CTest (+ Catch2)     |
| **Unit**      | `tests/unit/python/`          | `tests/unit/cpp/`            |
| **Integration** | `tests/integration/python/`, `tests/integration/bridge/` | `tests/integration/cpp/` |
| **Performance** | `tests/performance/python/`  | `tests/performance/cpp/`     |
| **Run**       | `pytest tests/`               | `ctest --test-dir build`     |
| **Build**     | None (pip install)            | CMake with `BUILD_TESTS=ON`  |

**Bridge tests** (`tests/integration/bridge/`) run in Python and exercise the native module (Python–C++ interface).

## Test Categories

| Category           | Purpose                    | Speed   | Where                    |
|--------------------|----------------------------|---------|---------------------------|
| **Unit (Python)**  | Individual functions/classes | < 1s  | `tests/unit/python/`      |
| **Unit (C++)**     | C++ components in isolation | < 1s  | `tests/unit/cpp/`        |
| **Integration (Python)** | Module interactions   | 1–10s | `tests/integration/python/` |
| **Integration (C++)**    | C++ pipelines          | 1–10s | `tests/integration/cpp/`  |
| **Bridge**         | Python–C++ interface        | 1–5s  | `tests/integration/bridge/` |
| **Performance (Python)** | Benchmarks, regression | Variable | `tests/performance/python/` |
| **Performance (C++)**    | C++ benchmarks (Google Benchmark) | Variable | `tests/performance/cpp/` |
| **Property-Based** | Mathematical properties     | Variable | `tests/property_based/`   |
| **E2E**            | Complete workflows          | 10–60s | `tests/e2e/`              |

## Next Steps

- Read [Test Structure](03-test-structure.md) for the full directory layout
- Review [Test Categories](04-test-categories/) for each category
- Use [Running Tests](14-running-tests.md) for pytest and ctest commands
- Use [Build Tests](15-build-tests.md) for building C++ tests

---

**Related**: [Test Structure](03-test-structure.md) | [Test Categories](04-test-categories/)
