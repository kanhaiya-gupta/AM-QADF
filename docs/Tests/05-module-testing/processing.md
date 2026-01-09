# Processing Module - Testing Guide

## Test Files

- `test_noise_reduction.py` - Noise filtering
- `test_signal_generation.py` - Derived signal generation

## Key Tests

### Noise Reduction
- Noise reduction effectiveness
- Filter parameter validation
- Signal preservation

### Signal Generation
- Signal generation accuracy
- Derived signal calculations
- Signal validation

## Coverage Target

**85%+**

## Example Tests

```python
def test_noise_reduction_effectiveness():
    """Test noise reduction effectiveness."""
    
def test_signal_generation_accuracy():
    """Test derived signal generation accuracy."""
```

## Running Processing Module Tests

```bash
# Run all processing tests
pytest tests/unit/processing/ -m unit

# Run specific test files
pytest tests/unit/processing/test_noise_reduction.py
pytest tests/unit/processing/test_signal_generation.py

# Run with coverage
pytest tests/unit/processing/ --cov=am_qadf.processing --cov-report=term-missing
```

---

**Parent**: [Module Testing Guides](README.md)

