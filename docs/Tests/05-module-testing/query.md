# Query Module - Testing Guide

## Test Files

- `test_base_query_client.py` - Abstract base class
- `test_hatching_client.py` - Hatching path queries
- `test_laser_parameter_client.py` - Laser parameter queries
- `test_ct_scan_client.py` - CT scan queries
- `test_ispm_thermal_client.py`, `test_ispm_optical_client.py`, `test_ispm_acoustic_client.py`, `test_ispm_plume_client.py`, `test_ispm_strain_client.py` - ISPM client queries
- `test_unified_query_client.py` - Unified interface
- `test_stl_model_client.py` - STL model queries
- `test_thermal_client.py` - Thermal queries
- `test_build_metadata_client.py` - Build metadata queries
- `test_query_utils.py` - Query utilities

## Key Tests

### Query Construction
- Query building
- Parameter validation
- Query optimization

### Spatial/Temporal Filtering
- Bounding box queries
- Time range queries
- Combined filters

### Error Handling
- Missing data handling
- Invalid query handling
- Connection errors

### Result Formatting
- Result structure
- Data type conversion
- Metadata preservation

### Mock MongoDB Interactions
- Collection queries
- Aggregation pipelines
- GridFS operations

## Coverage Target

**85%+**

## Example Tests

```python
def test_hatching_client_query_construction():
    """Test hatching query construction."""
    
def test_spatial_filtering():
    """Test spatial bounding box filtering."""
    
def test_error_handling_missing_data():
    """Test handling of missing data."""
```

## Running Query Module Tests

```bash
# Run all query module tests
pytest tests/unit/query/ -m unit

# Run specific client tests
pytest tests/unit/query/test_hatching_client.py
pytest tests/unit/query/test_laser_parameter_client.py
pytest tests/unit/query/test_unified_query_client.py

# Run with coverage
pytest tests/unit/query/ --cov=am_qadf.query --cov-report=term-missing

# Run specific test
pytest tests/unit/query/test_hatching_client.py::test_hatching_client_query_construction
```

---

**Parent**: [Module Testing Guides](README.md)

