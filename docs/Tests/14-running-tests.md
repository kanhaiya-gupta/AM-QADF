# Running Tests - Quick Reference Guide

## Quick Commands

### Run All Tests
```bash
# Run entire test suite
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=am_qadf --cov-report=html
```

### Run by Category

```bash
# Unit tests (fast)
pytest tests/unit/ -m unit

# Integration tests
pytest tests/integration/ -m integration

# Performance tests
pytest tests/performance/ -m performance

# Property-based tests
pytest tests/property_based/ -m property_based

# E2E tests (slow)
pytest tests/e2e/ -m e2e
```

### Run by Module

```bash
# Core module
pytest tests/unit/core/

# Query module
pytest tests/unit/query/

# Voxelization module
pytest tests/unit/voxelization/

# Signal mapping module (critical)
pytest tests/unit/signal_mapping/

# Synchronization module
pytest tests/unit/synchronization/

# Correction module
pytest tests/unit/correction/

# Processing module
pytest tests/unit/processing/

# Fusion module
pytest tests/unit/fusion/

# Quality module
pytest tests/unit/quality/

# Analytics module
pytest tests/unit/analytics/

# Anomaly detection module
pytest tests/unit/anomaly_detection/

# Visualization module
pytest tests/unit/visualization/

# Voxel domain module
pytest tests/unit/voxel_domain/
```

## Advanced Usage

### Run Specific Test File
```bash
pytest tests/unit/voxelization/test_voxel_grid.py
```

### Run Specific Test Function
```bash
pytest tests/unit/voxelization/test_voxel_grid.py::test_voxel_grid_creation
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
pytest tests/unit/ -m "unit and not slow"

# Run E2E and integration tests
pytest tests/ -m "e2e or integration"
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

### Performance Benchmarks
```bash
# Run benchmarks only
pytest tests/performance/benchmarks/ --benchmark-only

# Run with benchmark comparison
pytest tests/performance/benchmarks/ --benchmark-compare

# Run regression tests
pytest tests/performance/regression/ -m regression
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
# Run tests for module you're working on
pytest tests/unit/signal_mapping/ -v

# Run with coverage to see what's missing
pytest tests/unit/signal_mapping/ --cov=am_qadf.signal_mapping --cov-report=term-missing

# Run specific test while developing
pytest tests/unit/signal_mapping/methods/test_nearest_neighbor.py::test_interpolation_accuracy -vv
```

### Pre-Commit Checks
```bash
# Run fast tests only
pytest tests/unit/ -m "unit and not slow"

# Check coverage threshold
pytest tests/unit/ --cov=am_qadf --cov-fail-under=80
```

### CI/CD Pipeline
```bash
# Full test suite with coverage
pytest tests/ --cov=am_qadf --cov-report=xml --cov-report=html

# Unit tests
pytest tests/unit/ -m unit --cov=am_qadf

# Integration tests
pytest tests/integration/ -m integration

# Performance benchmarks (nightly)
pytest tests/performance/benchmarks/ --benchmark-only
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
| Unit Tests | < 1 minute | `pytest tests/unit/ -m unit` |
| Integration Tests | 1-5 minutes | `pytest tests/integration/ -m integration` |
| Performance Benchmarks | 5-15 minutes | `pytest tests/performance/benchmarks/ --benchmark-only` |
| Property-Based Tests | 2-10 minutes | `pytest tests/property_based/ -m property_based` |
| E2E Tests | 5-30 minutes | `pytest tests/e2e/ -m e2e` |
| **Full Suite** | **10-60 minutes** | `pytest tests/` |

## Related

- [Test Categories](04-test-categories/) - Category-specific running instructions
- [Module Testing Guides](05-module-testing/) - Module-specific running instructions
- [Tools](13-tools.md) - Testing tools and installation

---

**Parent**: [Test Documentation](README.md)

