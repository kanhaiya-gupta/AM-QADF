# Testing Tools and Libraries

## Required Tools

- **pytest**: Test framework
- **pytest-cov**: Coverage reporting
- **pytest-benchmark**: Performance benchmarking
- **pytest-mock**: Mocking utilities
- **hypothesis**: Property-based testing
- **numpy.testing**: NumPy-specific assertions
- **pytest-xdist**: Parallel test execution

## Optional Tools

- **pytest-timeout**: Test timeout management
- **pytest-asyncio**: Async test support
- **pytest-html**: HTML test reports
- **memory_profiler**: Memory profiling
- **line_profiler**: Line-by-line profiling

## Installation

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

## Usage Examples

### Pytest
```bash
pytest tests/ -v
pytest tests/unit/ -m unit
pytest tests/ -k "test_voxel"
```

### Coverage
```bash
pytest tests/ --cov=am_qadf --cov-report=html
```

### Benchmarking
```bash
pytest tests/performance/benchmarks/ --benchmark-only
```

### Parallel Execution
```bash
pytest tests/ -n auto  # Auto-detect CPU count
pytest tests/ -n 4     # Use 4 workers
```

## Related

- [Infrastructure](06-infrastructure.md) - Tool configuration
- [Best Practices](11-best-practices.md) - Tool usage best practices

---

**Parent**: [Test Documentation](README.md)

