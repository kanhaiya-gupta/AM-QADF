/**
 * C++ performance benchmarks: fusion (GridFusion.fuse).
 * Aligned with docs/Tests/Test_New_Plans.md (tests/performance/cpp/).
 */

#include <benchmark/benchmark.h>
#include "am_qadf_native/voxelization/openvdb_grid.hpp"
#include "am_qadf_native/fusion/grid_fusion.hpp"
#include <openvdb/openvdb.h>
#include <vector>

using namespace am_qadf_native::voxelization;
using namespace am_qadf_native::fusion;

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

static void BM_GridFusion_FuseTwoGrids(benchmark::State& state) {
    initOpenVDB();
    const int side = static_cast<int>(state.range(0));
    UniformVoxelGrid g1(1.0f, 0.0f, 0.0f, 0.0f);
    UniformVoxelGrid g2(1.0f, 0.0f, 0.0f, 0.0f);
    fillGrid(g1, side);
    fillGrid(g2, side);
    std::vector<openvdb::FloatGrid::Ptr> grids = { g1.getGrid(), g2.getGrid() };
    GridFusion fusion;
    for (auto _ : state) {
        openvdb::FloatGrid::Ptr fused = fusion.fuse(grids, "weighted_average");
        benchmark::DoNotOptimize(fused);
    }
    state.SetItemsProcessed(state.iterations() * 2);
}
BENCHMARK(BM_GridFusion_FuseTwoGrids)->Arg(8)->Arg(16)->Arg(32)->Unit(benchmark::kMillisecond);

static void BM_GridFusion_FuseFourGrids(benchmark::State& state) {
    initOpenVDB();
    const int side = 16;
    std::vector<openvdb::FloatGrid::Ptr> grids;
    for (int t = 0; t < 4; ++t) {
        UniformVoxelGrid grid(1.0f, 0.0f, 0.0f, 0.0f);
        fillGrid(grid, side);
        grids.push_back(grid.getGrid());
    }
    GridFusion fusion;
    for (auto _ : state) {
        openvdb::FloatGrid::Ptr fused = fusion.fuse(grids, "weighted_average");
        benchmark::DoNotOptimize(fused);
    }
    state.SetItemsProcessed(state.iterations() * 4);
}
BENCHMARK(BM_GridFusion_FuseFourGrids)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
