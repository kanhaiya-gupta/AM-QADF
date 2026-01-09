# Test Infrastructure

## Pytest Configuration

**`pytest.ini`**:
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --strict-markers
    --cov=src/am_qadf
    --cov-report=term-missing
    --cov-report=html
    --cov-report=xml
    --cov-fail-under=80
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    performance: Performance/benchmark tests
    slow: Slow-running tests
    requires_mongodb: Tests requiring MongoDB
    requires_spark: Tests requiring Spark
    property_based: Property-based tests
```

## Shared Fixtures (`conftest.py`)

Common fixtures available to all tests:

- `sample_points` - Sample 3D points
- `sample_voxel_grid` - Test voxel grid
- `mock_mongodb_client` - Mock MongoDB client
- `mock_spark_session` - Mock Spark session
- `mock_query_client` - Mock query client

## Test Utilities

**`tests/utils/`**:
- `test_helpers.py` - Common test utilities
- `assertions.py` - Custom assertion functions

## Fixtures

See [Test Data Management](10-test-data.md) for fixture documentation.

## Mock Objects

**`tests/fixtures/mocks/`**:
- `mock_mongodb.py` - MongoDB mock client
- `mock_spark.py` - Spark session mocks
- `mock_query_clients.py` - Query client mocks

## Related

- [Test Data Management](10-test-data.md) - Fixtures and test data
- [Best Practices](11-best-practices.md) - Infrastructure best practices

---

**Parent**: [Test Documentation](README.md)

