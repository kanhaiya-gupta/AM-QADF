# Quality Module - Testing Guide

## Test Files

- `test_completeness.py` - Completeness metrics
- `test_signal_quality.py` - Signal quality (SNR, etc.)
- `test_alignment_accuracy.py` - Alignment validation
- `test_data_quality.py` - Overall quality metrics

## Key Tests

### Completeness
- Completeness detection
- Coverage calculation
- Gap identification

### Signal Quality
- SNR calculation
- Signal-to-noise ratio
- Quality score calculation

### Alignment Accuracy
- Alignment error calculation
- Coordinate system alignment
- Alignment validation

### Data Quality
- Overall quality metrics
- Quality score ranges (0-1)
- Quality aggregation

## Coverage Target

**85%+**

## Example Tests

```python
def test_completeness_detection():
    """Test completeness detection."""
    
def test_signal_quality_snr():
    """Test signal quality SNR calculation."""
    
def test_alignment_accuracy():
    """Test alignment accuracy calculation."""
```

## Running Quality Module Tests

```bash
# Run all quality tests
pytest tests/unit/quality/ -m unit

# Run specific test files
pytest tests/unit/quality/test_completeness.py
pytest tests/unit/quality/test_signal_quality.py
pytest tests/unit/quality/test_alignment_accuracy.py
pytest tests/unit/quality/test_data_quality.py

# Run with coverage
pytest tests/unit/quality/ --cov=am_qadf.quality --cov-report=term-missing
```

---

**Parent**: [Module Testing Guides](README.md)

