# Correction Module - Testing Guide

## Test Files

- `test_geometric_distortion.py` - Distortion models
- `test_calibration.py` - Calibration procedures
- `test_validation.py` - Validation metrics

## Key Tests

### Geometric Distortion
- Distortion model accuracy
- Distortion correction
- Model parameter validation

### Calibration
- Calibration convergence
- Calibration accuracy
- Parameter estimation

### Validation
- Validation metric correctness
- Validation threshold checking
- Error reporting

## Coverage Target

**80%+**

## Example Tests

```python
def test_distortion_model_accuracy():
    """Test distortion model accuracy."""
    
def test_calibration_convergence():
    """Test calibration convergence."""
    
def test_validation_metrics():
    """Test validation metric calculations."""
```

## Running Correction Module Tests

```bash
# Run all correction tests
pytest tests/unit/correction/ -m unit

# Run specific test files
pytest tests/unit/correction/test_geometric_distortion.py
pytest tests/unit/correction/test_calibration.py
pytest tests/unit/correction/test_validation.py

# Run with coverage
pytest tests/unit/correction/ --cov=am_qadf.correction --cov-report=term-missing
```

---

**Parent**: [Module Testing Guides](README.md)

