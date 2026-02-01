# Integration Tests

## Purpose

Test interactions between modules and, where applicable, the Python–C++ boundary. Integration tests are split into **Python**, **bridge**, and **C++**.

## Layout

- **Python**: `tests/integration/python/` — workflow and cross-module tests (e.g. analytics, fusion, signal mapping pipeline, voxel domain).
- **Bridge**: `tests/integration/bridge/` — Python tests that call into the native (C++) module (correction, processing, query, signal mapping, synchronization, voxelization, etc.).
- **C++**: `tests/integration/cpp/` — C++-only pipelines (e.g. correction, fusion, signal mapping, synchronization).

## Characteristics

- **Speed**: Roughly 1–10 seconds per test
- **Scope**: Multiple components working together
- **Dependencies**: In-memory or mocked services where possible; bridge tests require the built native module

## Key Areas

- Signal mapping pipeline (query → transform → voxelize → interpolate)
- Voxel domain workflow (multi-source data, fusion, storage)
- Analytics and quality assessment workflows
- Python–C++ bridge (native module API, OpenVDB, calibration, etc.)
- C++ pipelines (correction, fusion, processing, synchronization)

## Running Integration Tests

### Python and bridge

```bash
# All integration tests (Python + bridge)
pytest tests/integration/ -m integration -v

# Python-only workflows
pytest tests/integration/python/ -m integration -v

# Bridge tests only (requires built am_qadf_native)
pytest tests/integration/bridge/ -m integration -v
```

### C++

From the build directory (after building with `BUILD_TESTS=ON`):

```bash
ctest --test-dir build --output-on-failure
# Or filter: ctest --test-dir build -R integration --output-on-failure
```

See [15. Build Tests](../15-build-tests.md) for building the C++ tests.

## Example (Python integration)

```python
@pytest.mark.integration
def test_signal_mapping_pipeline():
    """Test complete signal mapping pipeline."""
    query_client = MockQueryClient()
    voxel_grid = create_test_voxel_grid()
    result = VoxelDomainClient(query_client).map_signals_to_voxels(
        model_id="test_model", signals=["laser_power", "temperature"]
    )
    assert result.has_signal("laser_power")
    assert result.has_signal("temperature")
```

## Best Practices

1. **Realistic data**: Use fixtures that resemble real usage.
2. **Isolation**: Prefer mocks for MongoDB and other external services.
3. **Clear workflows**: One test per workflow or scenario where possible.
4. **Error paths**: Include tests for failure and edge cases.

## Related

- [E2E Tests](e2e-tests.md) - Full pipeline tests
- [Module Testing Guides](../05-module-testing/) - Per-module notes
- [Build Tests](../15-build-tests.md) - C++ build

---

**Parent**: [Test Categories](README.md)
