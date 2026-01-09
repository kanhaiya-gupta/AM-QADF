# Performance Tests

## Purpose

Validate performance claims and detect regressions.

## Characteristics

- **Speed**: Variable (benchmarks can be slow)
- **Scope**: Performance-critical operations
- **Dependencies**: Realistic test data
- **Count**: 6 test files (4 benchmarks + 2 regression)

## Test Structure

### Benchmarks (`tests/performance/benchmarks/`)

- `benchmark_signal_mapping.py` - Signal mapping performance
- `benchmark_voxel_fusion.py` - Voxel fusion performance
- `benchmark_interpolation_methods.py` - Interpolation method comparison
- `benchmark_parallel_execution.py` - Parallel execution speedup

### Regression Tests (`tests/performance/regression/`)

- `test_performance_regression.py` - Performance degradation detection
- `test_memory_regression.py` - Memory leak detection

## Key Benchmarks

- Signal mapping: 1M points → voxel grid
- Voxel fusion: 10 signals, 1M voxels
- Interpolation methods: Compare speed/accuracy
- Parallel execution: Speedup vs sequential
- Spark execution: Scalability

## Example Structure

```python
@pytest.mark.benchmark
@pytest.mark.performance
def test_signal_mapping_performance(benchmark):
    """Benchmark signal mapping performance."""
    # Arrange
    large_dataset = create_large_test_dataset()
    
    # Act
    result = benchmark(
        map_signals_to_voxels,
        large_dataset,
        method='nearest_neighbor'
    )
    
    # Assert
    assert result is not None
```

## Performance Regression

### Detection
- Track execution time over time
- Alert on performance degradation (>10%)
- Compare against baseline benchmarks

### Memory Profiling
- Memory usage for large datasets
- Memory leak detection
- Peak memory tracking

## Running Performance Tests

```bash
# Run benchmarks
pytest tests/performance/benchmarks/ --benchmark-only

# Run regression tests
pytest tests/performance/regression/ -m regression

# Run all performance tests
pytest tests/performance/ -m performance
```

## Baseline Management

Performance baselines are stored in:
- `tests/performance/regression/performance_baselines.json`
- `tests/performance/regression/memory_baselines.json`

## Related

- [Performance Testing Strategy](../09-performance.md) - Detailed performance testing guide
- [Best Practices](../11-best-practices.md) - Performance testing best practices

---

**Parent**: [Test Categories](README.md)

