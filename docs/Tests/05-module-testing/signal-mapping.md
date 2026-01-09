# Signal Mapping Module - Testing Guide

## ⭐ CRITICAL MODULE

The signal mapping module is the **core** of the AM-QADF framework and requires the highest test coverage (95%+).

## Test Structure

```
tests/unit/signal_mapping/
├── methods/
│   ├── test_nearest_neighbor.py
│   ├── test_linear.py
│   ├── test_idw.py
│   └── test_kde.py
├── execution/
│   ├── test_sequential.py
│   ├── test_parallel.py
│   └── test_spark.py
└── utils/
    ├── test_coordinate_utils.py
    ├── test_performance.py
    └── test_spark_utils.py
```

## Key Tests

### Mathematical Correctness
- Interpolation accuracy
- Boundary condition handling
- Edge cases (empty data, single point)
- Numerical stability (NaN/Inf handling)

### Performance
- Vectorization effectiveness
- Parallel execution correctness
- Spark distributed execution
- Memory efficiency

### Critical Test Cases

```python
def test_interpolation_accuracy():
    """Verify interpolation methods produce accurate results."""
    
def test_interpolation_with_empty_data():
    """Handle empty point clouds gracefully."""
    
def test_interpolation_boundary_conditions():
    """Test points outside voxel grid."""
    
def test_parallel_execution_correctness():
    """Verify parallel execution produces same results as sequential."""
    
def test_spark_execution_scalability():
    """Test Spark execution with large datasets."""
```

## Coverage Target

**95%+** - This is the most critical module.

## Running Signal Mapping Module Tests

```bash
# Run all signal mapping tests
pytest tests/unit/signal_mapping/ -m unit

# Run by submodule
pytest tests/unit/signal_mapping/methods/        # Interpolation methods
pytest tests/unit/signal_mapping/execution/      # Execution strategies
pytest tests/unit/signal_mapping/utils/          # Utilities

# Run specific interpolation method tests
pytest tests/unit/signal_mapping/methods/test_nearest_neighbor.py
pytest tests/unit/signal_mapping/methods/test_linear.py
pytest tests/unit/signal_mapping/methods/test_idw.py

# Run execution tests
pytest tests/unit/signal_mapping/execution/test_sequential.py
pytest tests/unit/signal_mapping/execution/test_parallel.py
pytest tests/unit/signal_mapping/execution/test_spark.py

# Run with coverage (critical module)
pytest tests/unit/signal_mapping/ --cov=am_qadf.signal_mapping --cov-report=html

# Run property-based tests
pytest tests/property_based/test_interpolation_properties.py -m property_based

# Run performance benchmarks
pytest tests/performance/benchmarks/benchmark_interpolation_methods.py --benchmark-only
```

## Related

- [Property-Based Tests](../04-test-categories/property-based-tests.md) - Mathematical properties
- [Performance Tests](../04-test-categories/performance-tests.md) - Performance benchmarks

---

**Parent**: [Module Testing Guides](README.md)

