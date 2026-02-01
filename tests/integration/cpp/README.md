# C++ Integration Tests

C++ integration tests for `am_qadf_native` (Catch2). They chain multiple modules: e.g. voxelization → io, signal_mapping → io, fusion → io. Aligned with [Test_New_Plans.md](../../../docs/Tests/Test_New_Plans.md) (tests/integration/cpp/).

## Building

Same as [unit C++ tests](../../unit/cpp/README.md): the main project builds `am_qadf_native` as a Python module. To build and run these integration tests you need a static library `am_qadf_native_static` and to include this directory from the root, e.g.:

```cmake
if(BUILD_TESTS)
  add_library(am_qadf_native_static STATIC ...)  # same sources as am_qadf_native
  add_subdirectory(tests/unit/cpp)
  add_subdirectory(tests/integration/cpp)
endif()
```

Then:

```bash
mkdir build && cd build
cmake -DBUILD_TESTS=ON ..
cmake --build .
ctest -R test_integration_cpp
```

## Test files

| File | Pipeline |
|------|----------|
| `test_openvdb_integration.cpp` | Create grid → VDBWriter → OpenVDBReader → verify |
| `test_grid_operations.cpp` | UniformVoxelGrid → add points → VDBWriter → OpenVDBReader → verify |
| `test_signal_mapping_pipeline.cpp` | Grid + NearestNeighborMapper.map → (optional) write/read |
| `test_fusion_pipeline.cpp` | Two grids → GridFusion.fuse → (optional) write/read |
| `test_synchronization_pipeline.cpp` | Source + reference grids → GridSynchronizer.synchronizeSpatial |
| `test_correction_pipeline.cpp` | QueryResult → SignalNoiseReduction (gaussian, outlier removal) |
| `test_processing_pipeline.cpp` | SignalProcessing (normalize, movingAverage, derivative/integral) + SignalGeneration |

All tests use temporary files where needed and clean up in the same process.
