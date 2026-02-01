# AM-QADF Testing Documentation

**Version**: 1.0  
**Last Updated**: 2025  
**Status**: ✅ Complete

## Overview

This directory contains testing documentation for the AM-QADF framework. Tests are split into **Python** (pytest) and **C++** (CMake/ctest), including **bridge** tests that exercise the Python–C++ interface.

## Documentation Structure

```
docs/Tests/
├── README.md                    # This file - navigation guide
├── 00-INDEX.md                  # Documentation index
├── 01-overview.md               # Testing goals and principles
├── 02-philosophy.md             # TDD, BDD, Property-Based Testing
├── 03-test-structure.md         # Directory structure (Python + C++)
├── 04-test-categories/          # Test category documentation
│   ├── unit-tests.md
│   ├── integration-tests.md
│   ├── performance-tests.md
│   ├── property-based-tests.md
│   └── e2e-tests.md
├── 05-module-testing/           # Module-specific test guides
├── 06-infrastructure.md         # Pytest config, fixtures, C++ build
├── 07-coverage.md               # Coverage (Python)
├── 08-cicd.md                   # CI/CD integration
├── 09-performance.md            # Performance testing strategy
├── 10-test-data.md              # Test data and fixtures
├── 11-best-practices.md         # Testing best practices
├── 12-success-metrics.md        # Success metrics and KPIs
├── 13-tools.md                  # Testing tools (pytest, CMake, ctest)
├── 14-running-tests.md         # ⚡ Quick reference - running tests
└── 15-build-tests.md            # Building Python and C++ tests
```

## Quick Navigation

### Getting Started
- **[Overview](01-overview.md)** - Goals, principles, Python vs C++
- **[Test Structure](03-test-structure.md)** - Directory layout (unit/python, unit/cpp, integration, etc.)
- **[Running Tests](14-running-tests.md)** - ⚡ Quick reference for pytest and ctest
- **[Build Tests](15-build-tests.md)** - Building C++ tests and benchmarks

### Test Categories
- **[Unit Tests](04-test-categories/unit-tests.md)** - Python and C++ unit tests
- **[Integration Tests](04-test-categories/integration-tests.md)** - Python, C++, and bridge
- **[Performance Tests](04-test-categories/performance-tests.md)** - Python benchmarks and C++ benchmarks
- **[Property-Based](04-test-categories/property-based-tests.md)** | **[E2E](04-test-categories/e2e-tests.md)**

### For Developers
- **[Module Testing Guides](05-module-testing/)** - Module-specific instructions
- **[Best Practices](11-best-practices.md)** - Conventions and guidelines
- **[Infrastructure](06-infrastructure.md)** - Fixtures, pytest config, C++ test build

### For DevOps
- **[CI/CD](08-cicd.md)** - GitHub Actions (Python + C++ jobs)
- **[Performance](09-performance.md)** - Benchmarks and regression
- **[Success Metrics](12-success-metrics.md)** - KPIs

### Reference
- **[Coverage](07-coverage.md)** - Python coverage targets
- **[Test Data](10-test-data.md)** - Fixtures and test data
- **[Tools](13-tools.md)** - pytest, CMake, ctest

## Test Layout Summary

| Layer        | Python                          | C++                    |
|-------------|----------------------------------|------------------------|
| **Unit**    | `tests/unit/python/`            | `tests/unit/cpp/`      |
| **Integration** | `tests/integration/python/`, `tests/integration/bridge/` | `tests/integration/cpp/` |
| **Performance** | `tests/performance/python/`  | `tests/performance/cpp/` |
| **E2E / Other** | `tests/e2e/`, `tests/property_based/`, `tests/utils/` | — |

- **Python**: `pytest tests/` (see [14-running-tests.md](14-running-tests.md)).
- **C++**: Build with CMake (`BUILD_TESTS=ON`), then `ctest --test-dir build --output-on-failure` (see [15-build-tests.md](15-build-tests.md)).

---

**Last Updated**: 2025
