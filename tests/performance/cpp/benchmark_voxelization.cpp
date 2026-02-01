/**
 * C++ performance benchmarks: voxelization (UniformVoxelGrid, add_point).
 * Aligned with docs/Tests/Test_New_Plans.md (tests/performance/cpp/).
 */

#include <benchmark/benchmark.h>
#include "am_qadf_native/voxelization/openvdb_grid.hpp"
#include <openvdb/openvdb.h>
#include <random>

using namespace am_qadf_native::voxelization;

namespace {
void initOpenVDB() {
    openvdb::initialize();
}
} // namespace

static void BM_UniformVoxelGrid_Create(benchmark::State& state) {
    initOpenVDB();
    for (auto _ : state) {
        UniformVoxelGrid grid(1.0f, 0.0f, 0.0f, 0.0f);
        grid.setSignalName("signal");
        benchmark::DoNotOptimize(grid.getGrid());
    }
}
BENCHMARK(BM_UniformVoxelGrid_Create);

static void BM_UniformVoxelGrid_AddPointAtVoxel(benchmark::State& state) {
    initOpenVDB();
    const int n = static_cast<int>(state.range(0));
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 127);
    for (auto _ : state) {
        state.PauseTiming();
        UniformVoxelGrid grid(1.0f, 0.0f, 0.0f, 0.0f);
        state.ResumeTiming();
        for (int i = 0; i < n; ++i) {
            int x = dist(rng), y = dist(rng), z = dist(rng);
            grid.addPointAtVoxel(x, y, z, static_cast<float>(i));
        }
        benchmark::DoNotOptimize(grid.getGrid());
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
}
BENCHMARK(BM_UniformVoxelGrid_AddPointAtVoxel)->Arg(100)->Arg(1000)->Arg(10000)->Unit(benchmark::kMicrosecond);

static void BM_UniformVoxelGrid_AddPointWorld(benchmark::State& state) {
    initOpenVDB();
    const int n = static_cast<int>(state.range(0));
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 128.0f);
    for (auto _ : state) {
        state.PauseTiming();
        UniformVoxelGrid grid(1.0f, 0.0f, 0.0f, 0.0f);
        state.ResumeTiming();
        for (int i = 0; i < n; ++i) {
            float x = dist(rng), y = dist(rng), z = dist(rng);
            grid.addPoint(x, y, z, static_cast<float>(i));
        }
        benchmark::DoNotOptimize(grid.getGrid());
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
}
BENCHMARK(BM_UniformVoxelGrid_AddPointWorld)->Arg(100)->Arg(1000)->Arg(10000)->Unit(benchmark::kMicrosecond);

static void BM_UniformVoxelGrid_GetValue(benchmark::State& state) {
    initOpenVDB();
    UniformVoxelGrid grid(1.0f, 0.0f, 0.0f, 0.0f);
    for (int i = 0; i < 64; ++i)
        grid.addPointAtVoxel(i % 16, (i / 16) % 4, i / 64, static_cast<float>(i));
    float sum = 0.0f;
    for (auto _ : state) {
        for (int i = 0; i < 64; ++i)
            sum += grid.getValue(i % 16, (i / 16) % 4, i / 64);
    }
    benchmark::DoNotOptimize(sum);
    state.SetItemsProcessed(state.iterations() * 64);
}
BENCHMARK(BM_UniformVoxelGrid_GetValue);

BENCHMARK_MAIN();
