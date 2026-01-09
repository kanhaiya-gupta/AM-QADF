# Visualization Module - Testing Guide

## Test Files

- `test_voxel_renderer.py` - 3D rendering
- `test_multi_resolution_viewer.py` - Multi-resolution viewing
- `test_multi_resolution_widgets.py` - Multi-resolution widgets
- `test_adaptive_resolution_widgets.py` - Adaptive resolution widgets
- `test_notebook_widgets.py` - Jupyter notebook widgets

## Key Tests

### Rendering
- Rendering correctness (visual regression tests)
- 3D visualization accuracy
- Color mapping

### Widgets
- Widget functionality
- Interactive features
- Widget state management

### Performance
- Frame rate
- Memory usage
- Large dataset handling

## Coverage Target

**70%+** - Visualization is harder to test automatically

## Example Tests

```python
def test_voxel_rendering_correctness():
    """Test voxel rendering correctness."""
    
def test_widget_functionality():
    """Test widget interactive features."""
    
def test_rendering_performance():
    """Test rendering performance."""
```

## Running Visualization Module Tests

```bash
# Run all visualization tests
pytest tests/unit/visualization/ -m unit

# Run specific test files
pytest tests/unit/visualization/test_voxel_renderer.py
pytest tests/unit/visualization/test_multi_resolution_viewer.py
pytest tests/unit/visualization/test_notebook_widgets.py

# Run with coverage
pytest tests/unit/visualization/ --cov=am_qadf.visualization --cov-report=term-missing

# Note: Some visualization tests may require manual verification
```

## Note

Visualization tests may require manual verification for visual correctness.

---

**Parent**: [Module Testing Guides](README.md)

