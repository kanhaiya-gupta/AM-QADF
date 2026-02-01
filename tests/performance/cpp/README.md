# C++ Performance Benchmarks

C++ performance benchmarks for `am_qadf_native` using [Google Benchmark](https://github.com/google/benchmark). Aligned with [Test_New_Plans.md](../../../docs/Tests/Test_New_Plans.md) (tests/performance/cpp/).

## Requirements

- **Google Benchmark**: If not installed, the root CMakeLists.txt will fetch it from GitHub (FetchContent) when `ENABLE_BENCHMARKS=ON`. Otherwise use `find_package(benchmark)` (build from [google/benchmark](https://github.com/google/benchmark)—there is no conda package named `google-benchmark`).
- Static library `am_qadf_native_static` (same as C++ unit/integration tests).
- OpenVDB, TBB, etc. as for the native library.

## Building

Benchmarks are enabled by default (`ENABLE_BENCHMARKS=ON`). Configure and build:

```bash
mkdir build && cd build
cmake -DBUILD_TESTS=ON ..
cmake --build .
```

If Google Benchmark is not on the system, CMake will download and build it at configure time.

## Executables

| Executable           | Source                     | Benchmarks |
|---------------------|----------------------------|------------|
| `bench_voxelization` | `benchmark_voxelization.cpp` | UniformVoxelGrid create, add_point_at_voxel (100/1k/10k), add_point world, get_value |
| `bench_signal_mapping` | `benchmark_signal_mapping.cpp` | NearestNeighbor/Linear/IDW map with 100/500/1k/5k points |
| `bench_fusion`      | `benchmark_fusion.cpp`     | GridFusion.fuse (2 grids, 4 grids; side 8/16/32) |
| `bench_synchronization` | `benchmark_synchronization.cpp` | GridSynchronizer.synchronize_spatial (side 8/16/32, 2/4/8 sources) |
| `bench_query`       | `benchmark_query.cpp`      | QueryResult build (100/1k/10k/100k points), num_points, empty check |

## Running

```bash
./bench_voxelization
./bench_signal_mapping --benchmark_filter="NearestNeighbor"
./bench_fusion --benchmark_format=console
```

Use `--benchmark_filter=REGEX` to run a subset of benchmarks. Results are in nanoseconds per iteration by default; some benchmarks use `kMicrosecond` or `kMillisecond`.
