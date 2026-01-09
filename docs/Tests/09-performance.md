# Performance Testing

## Benchmark Tests

**Key Benchmarks**:
- Signal mapping: 1M points → voxel grid
- Voxel fusion: 10 signals, 1M voxels
- Interpolation methods: Compare speed/accuracy
- Parallel execution: Speedup vs sequential
- Spark execution: Scalability

## Performance Regression Tests

- Track execution time over time
- Alert on performance degradation (>10%)
- Compare against baseline benchmarks

## Memory Profiling

- Memory usage for large datasets
- Memory leak detection
- Peak memory tracking

## Baseline Management

Performance baselines are stored in:
- `tests/performance/regression/performance_baselines.json`
- `tests/performance/regression/memory_baselines.json`

## Running Performance Tests

```bash
# Run benchmarks
pytest tests/performance/benchmarks/ --benchmark-only

# Run regression tests
pytest tests/performance/regression/ -m regression

# Run all performance tests
pytest tests/performance/ -m performance
```

## Related

- [Performance Test Category](04-test-categories/performance-tests.md) - Detailed performance test documentation
- [Success Metrics](12-success-metrics.md) - Performance metrics

---

**Parent**: [Test Documentation](README.md)

