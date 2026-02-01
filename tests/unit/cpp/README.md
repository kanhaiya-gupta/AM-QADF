# C++ Unit Tests

C++ unit tests for `am_qadf_native` (Catch2). Aligned with [Test_New_Plans.md](../../../docs/Tests/Test_New_Plans.md).

## Building

The main project builds `am_qadf_native` as a **Python module** (pybind11), not a static library. To build and run these C++ tests you have two options:

### Option 1: Add a static library in the root CMakeLists.txt

In the root `CMakeLists.txt`, add a static library from the same sources as the Python module, then include this directory:

```cmake
if(BUILD_TESTS)
  add_library(am_qadf_native_static STATIC
    src/am_qadf_native/src/voxelization/openvdb_grid.cpp
    src/am_qadf_native/src/voxelization/numpy_converter.cpp
    src/am_qadf_native/src/voxelization/hatching_voxelizer.cpp
    src/am_qadf_native/src/voxelization/stl_voxelizer.cpp
    src/am_qadf_native/src/voxelization/unified_grid_factory.cpp
    # ... other native sources and io (vdb_writer, etc.) as needed
  )
  target_include_directories(am_qadf_native_static PUBLIC src/am_qadf_native/include)
  # Same TBB, OpenVDB, etc. as am_qadf_native
  add_subdirectory(tests/unit/cpp)
  add_subdirectory(tests/integration/cpp)
endif()
```

Then configure and build:

```bash
mkdir build && cd build
cmake -DBUILD_TESTS=ON ..
cmake --build .
ctest -R test_voxelization_cpp
```

### Option 2: Build from this directory (standalone)

From the repo root, with OpenVDB, TBB, and Catch2 available:

```bash
cd tests/unit/cpp/voxelization
mkdir build && cd build
cmake ../  # or point to repo root and use a config that provides am_qadf_native_static
cmake --build .
./test_voxelization_cpp
```

If `am_qadf_native_static` is not defined, the voxelization `CMakeLists.txt` still creates the test executable but linking will fail until the static lib is added in the root.

## Structure

- `voxelization/` – tests for `openvdb_grid`, `numpy_converter`, `unified_grid_factory`, `hatching_voxelizer`, `stl_voxelizer`, `VoxelGridFactory`
- `signal_mapping/` – tests for `interpolation_base`, `nearest_neighbor`, `linear_interpolation`, `idw_interpolation`, `kde_interpolation`, `rbf_interpolation`
- `fusion/` – tests for `grid_fusion`, `fusion_strategies`, `fusion_quality` (placeholder)
- `synchronization/` – tests for `grid_spatial_alignment`, `grid_temporal_alignment`, `grid_synchronizer`, `point_bounds`, `point_transform` (point_* when EIGEN_AVAILABLE)
- `query/` – tests for `query_result`, `point_converter`, `ct_image_reader`, `mongodb_query_client`, `ct_scan_query`, `laser_monitoring_query` (MongoDB tests skip when server unavailable)
- `correction/` – tests for `signal_noise_reduction`, `spatial_noise_filtering`, `geometric_correction`, `calibration`, `validation`
- `processing/` – tests for `signal_processing`, `signal_generation`
- `io/` – tests for `openvdb_reader`, `vdb_writer`, `paraview_exporter`, `mongodb_writer` (optional; skips when MongoDB unavailable)

## Test files (voxelization)

| File | Source under test |
|------|-------------------|
| `test_openvdb_grid.cpp` | `openvdb_grid.cpp` (UniformVoxelGrid, MultiResolution, Adaptive) |
| `test_numpy_converter.cpp` | `numpy_converter.cpp` (requires pybind11) |
| `test_unified_grid_factory.cpp` | `unified_grid_factory.cpp` (requires EIGEN_AVAILABLE) |
| `test_hatching_voxelizer.cpp` | `hatching_voxelizer.cpp` |
| `test_stl_voxelizer.cpp` | `stl_voxelizer.cpp` |
| `test_grid_factory.cpp` | `openvdb_grid.hpp` (VoxelGridFactory) |

## Test files (signal_mapping)

| File | Source under test |
|------|-------------------|
| `test_interpolation_base.cpp` | `interpolation_base.cpp` (Point, numpy_to_points) |
| `test_nearest_neighbor.cpp` | `nearest_neighbor.cpp` |
| `test_linear_interpolation.cpp` | `linear_interpolation.cpp` |
| `test_idw_interpolation.cpp` | `idw_interpolation.cpp` |
| `test_kde_interpolation.cpp` | `kde_interpolation.cpp` |
| `test_rbf_interpolation.cpp` | `rbf_interpolation.cpp` (EIGEN_AVAILABLE for full solve) |

## Test files (fusion)

| File | Source under test |
|------|-------------------|
| `test_grid_fusion.cpp` | `grid_fusion.cpp` (fuse, fuseWeighted) |
| `test_fusion_strategies.cpp` | `fusion_strategies.cpp` (WeightedAverage, Max, Min, Median) |
| `test_fusion_quality.cpp` | Placeholder (no C++ implementation yet) |

## Test files (synchronization)

The synchronization module has **11 headers** and **10 .cpp** sources. Coverage:

| Test file | Source / header | Status |
|-----------|-----------------|--------|
| `test_grid_spatial_alignment.cpp` | `grid_spatial_alignment.cpp` | ✅ Full |
| `test_grid_temporal_alignment.cpp` | `grid_temporal_alignment.cpp` | ✅ Full |
| `test_grid_synchronizer.cpp` | `grid_synchronizer.cpp` | ✅ Full |
| `test_point_bounds.cpp` | `point_bounds.cpp` | ✅ Full (EIGEN_AVAILABLE) |
| `test_point_transform.cpp` | `point_transform.cpp` | ✅ Full (EIGEN_AVAILABLE) |
| `test_point_coordinate_transform.cpp` | `point_coordinate_transform.cpp` | ✅ Added |
| `test_point_temporal_alignment.cpp` | `point_temporal_alignment.cpp` | ✅ Added |
| `test_point_transformation_estimate.cpp` | `point_transformation_estimate.cpp` | ✅ Added |
| `test_point_transformation_sampling.cpp` | `point_transformation_sampling.cpp` | ✅ Added |
| `test_point_transformation_validate.cpp` | `point_transformation_validate.cpp` | ✅ Added |
| `test_hatching_sampling.cpp` | `hatching_sampling.hpp` (no .cpp; API only) | Placeholder |

All point_* and hatching tests are built only when **EIGEN_AVAILABLE**.

## Test files (query)

| File | Source under test |
|------|-------------------|
| `test_query_result.cpp` | `query_result.hpp` (struct: empty, num_points, num_vectors, format, has_contours, has_multiple_signals) |
| `test_point_converter.cpp` | `point_converter.cpp` (pointsToEigenMatrix, eigenMatrixToPoints, queryResultToEigenMatrix; EIGEN_AVAILABLE) |
| `test_ct_image_reader.cpp` | `ct_image_reader.cpp` (readRawBinary, readDICOMSeries, readNIfTI, readTIFFStack) |
| `test_mongodb_query_client.cpp` | `mongodb_query_client.cpp` (skips when MongoDB unavailable) |
| `test_ct_scan_query.cpp` | `ct_scan_query.cpp` (skips when MongoDB unavailable) |
| `test_laser_monitoring_query.cpp` | `laser_monitoring_query.cpp` (skips when MongoDB unavailable) |

## Test files (correction)

| File | Source under test |
|------|-------------------|
| `test_signal_noise_reduction.cpp` | `signal_noise_reduction.cpp` (reduceNoise, applyGaussianFilter, applySavitzkyGolay, removeOutliers) |
| `test_spatial_noise_filtering.cpp` | `spatial_noise_filtering.cpp` (apply, applyMedianFilter, applyBilateralFilter, applyGaussianFilter) |
| `test_geometric_correction.cpp` | `geometric_correction.cpp` (correctDistortions, correctLensDistortion, correctSensorMisalignment) |
| `test_calibration.cpp` | `calibration.cpp` (loadFromFile, saveToFile, computeCalibration, validateCalibration) |
| `test_validation.cpp` | `validation.cpp` (validateGrid, validateSignalData, validateCoordinates, checkConsistency) |

## Test files (processing)

| File | Source under test |
|------|-------------------|
| `test_signal_processing.cpp` | `signal_processing.cpp` (normalize, movingAverage, derivative, integral, fft, ifft, frequencyFilter) |
| `test_signal_generation.cpp` | `signal_generation.cpp` (generateSynthetic, generateGaussian, generateSineWave, generateRandom, generateFromExpression) |

## Test files (io)

| File | Source under test |
|------|-------------------|
| `test_openvdb_reader.cpp` | `openvdb_reader.cpp` (loadGridByName, loadAllGrids; pybind11 methods not tested) |
| `test_vdb_writer.cpp` | `vdb_writer.cpp` (write, writeMultiple, writeMultipleWithNames, writeCompressed, append) |
| `test_paraview_exporter.cpp` | `paraview_exporter.cpp` (exportToParaView, exportMultipleToParaView, exportWithMetadata) |
| `test_mongodb_writer.cpp` | `mongodb_writer.cpp` (optional; EIGEN_AVAILABLE; skips when MongoDB unavailable) |
