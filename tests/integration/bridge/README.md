# Python–C++ Bridge Integration Tests

Python tests that exercise the `am_qadf_native` (pybind11) module from Python. They verify that the C++ API is correctly exposed and callable from Python. Aligned with [Test_New_Plans.md](../../../docs/Tests/Test_New_Plans.md) (tests/integration/bridge/).

## Requirements

- `am_qadf_native` must be built (pybind11 extension) and importable from Python (e.g. `PYTHONPATH` includes the build directory or the package is installed).

## Running

From the repo root:

```bash
# With am_qadf_native on PYTHONPATH (e.g. after building in build/):
export PYTHONPATH=build/src/am_qadf_native:$PYTHONPATH  # adjust path as needed
pytest tests/integration/bridge/ -v -m "integration and bridge"
```

If `am_qadf_native` is not importable, all bridge tests are skipped via the `native_module` fixture.

## Test files

| File | Coverage |
|------|----------|
| `test_python_cpp_bridge.py` | Module import, submodule presence (io, signal_mapping, fusion, etc.) |
| `test_voxelization_bridge.py` | UniformVoxelGrid, VoxelGridFactory, add_point, get_value, get_statistics |
| `test_signal_mapping_bridge.py` | NearestNeighborMapper, LinearMapper, IDWMapper, numpy_to_points, Point |
| `test_fusion_bridge.py` | GridFusion.fuse, fuse_weighted |
| `test_synchronization_bridge.py` | GridSynchronizer.synchronize_spatial, SpatialAlignment |
| `test_query_bridge.py` | QueryResult, num_points, empty, MongoDBQueryClient |
| `test_correction_bridge.py` | SignalNoiseReduction (reduce_noise, apply_gaussian_filter, remove_outliers), Calibration, Validation |
| `test_processing_bridge.py` | SignalProcessing (normalize, moving_average, derivative, integral), SignalGeneration.generate_random |
| `test_io_bridge.py` | VDBWriter, OpenVDBReader (write/read round-trip), ParaViewExporter |

## Fixtures

- `conftest.py` defines a session-scoped `native_module` fixture that imports `am_qadf_native` and skips the test session if the import fails.
