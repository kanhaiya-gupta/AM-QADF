/**
 * C++ performance benchmarks: synchronization (GridSynchronizer.synchronize_spatial).
 * Aligned with docs/Tests/Test_New_Plans.md (tests/performance/cpp/).
 */

#include <benchmark/benchmark.h>
#include "am_qadf_native/voxelization/openvdb_grid.hpp"
#include "am_qadf_native/synchronization/grid_synchronizer.hpp"
#include <openvdb/openvdb.h>
#include <vector>

using namespace am_qadf_native::voxelization;
using namespace am_qadf_native::synchronization;

namespace {
void initOpenVDB() {
    openvdb::initialize();
}

void fillGrid(UniformVoxelGrid& grid, int side) {
    for (int i = 0; i < side; ++i)
        for (int j = 0; j < side; ++j)
            for (int k = 0; k < side; ++k)
                grid.addPointAtVoxel(i, j, k, static_cast<float>(i + j + k));
}
} // namespace

static void BM_GridSynchronizer_SynchronizeSpatial(benchmark::State& state) {
    initOpenVDB();
    const int side = static_cast<int>(state.range(0));
    UniformVoxelGrid ref(1.0f, 0.0f, 0.0f, 0.0f);
    fillGrid(ref, side);
    UniformVoxelGrid src(1.0f, 0.0f, 0.0f, 0.0f);
    fillGrid(src, side);
    std::vector<openvdb::FloatGrid::Ptr> sources = { src.getGrid() };
    openvdb::FloatGrid::Ptr reference = ref.getGrid();
    GridSynchronizer sync;
    for (auto _ : state) {
        std::vector<openvdb::FloatGrid::Ptr> out = sync.synchronizeSpatial(sources, reference, "trilinear");
        benchmark::DoNotOptimize(out);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_GridSynchronizer_SynchronizeSpatial)->Arg(8)->Arg(16)->Arg(32)->Unit(benchmark::kMillisecond);

static void BM_GridSynchronizer_SynchronizeSpatialMultiple(benchmark::State& state) {
    initOpenVDB();
    const int numSources = static_cast<int>(state.range(0));
    const int side = 16;
    UniformVoxelGrid ref(1.0f, 0.0f, 0.0f, 0.0f);
    fillGrid(ref, side);
    std::vector<openvdb::FloatGrid::Ptr> sources;
    for (int i = 0; i < numSources; ++i) {
        UniformVoxelGrid src(1.0f, 0.0f, 0.0f, 0.0f);
        fillGrid(src, side);
        sources.push_back(src.getGrid());
    }
    GridSynchronizer sync;
    for (auto _ : state) {
        std::vector<openvdb::FloatGrid::Ptr> out = sync.synchronizeSpatial(sources, ref.getGrid(), "trilinear");
        benchmark::DoNotOptimize(out);
    }
    state.SetItemsProcessed(state.iterations() * numSources);
}
BENCHMARK(BM_GridSynchronizer_SynchronizeSpatialMultiple)->Arg(2)->Arg(4)->Arg(8)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
