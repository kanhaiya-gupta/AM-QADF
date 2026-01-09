# Fusion Module - Testing Guide

## Test Files

- `test_voxel_fusion.py` - Voxel-level fusion
- `test_fusion_quality.py` - Quality metrics
- `test_fusion_methods.py` - Fusion strategies

## Key Tests

### Fusion Strategies
- Weighted average fusion
- Median fusion
- Quality-based fusion
- Max/Min fusion

### Quality Metrics
- Quality metric calculations
- Quality-based weighting
- Quality score validation

### Performance
- Vectorized operations
- Large dataset handling
- Memory efficiency

### Edge Cases
- Missing signals handling
- NaN handling
- Empty data handling

## Coverage Target

**90%+**

## Example Tests

```python
def test_weighted_average_fusion():
    """Test weighted average fusion strategy."""
    
def test_fusion_quality_metrics():
    """Test fusion quality metric calculations."""
    
def test_fusion_with_missing_signals():
    """Test fusion with missing signals."""
```

## Running Fusion Module Tests

```bash
# Run all fusion tests
pytest tests/unit/fusion/ -m unit

# Run specific test files
pytest tests/unit/fusion/test_voxel_fusion.py
pytest tests/unit/fusion/test_fusion_quality.py
pytest tests/unit/fusion/test_fusion_methods.py

# Run with coverage
pytest tests/unit/fusion/ --cov=am_qadf.fusion --cov-report=term-missing

# Run property-based tests
pytest tests/property_based/test_fusion_properties.py -m property_based

# Run performance benchmarks
pytest tests/performance/benchmarks/benchmark_voxel_fusion.py --benchmark-only
```

## Related

- [Property-Based Tests](../04-test-categories/property-based-tests.md) - Fusion properties

---

**Parent**: [Module Testing Guides](README.md)

