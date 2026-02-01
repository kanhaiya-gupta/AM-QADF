/**
 * C++ performance benchmarks: signal mapping (NearestNeighbor, Linear, IDW).
 * Aligned with docs/Tests/Test_New_Plans.md (tests/performance/cpp/).
 */

#include <benchmark/benchmark.h>
#include "am_qadf_native/voxelization/openvdb_grid.hpp"
#include "am_qadf_native/signal_mapping/nearest_neighbor.hpp"
#include "am_qadf_native/signal_mapping/linear_interpolation.hpp"
#include "am_qadf_native/signal_mapping/idw_interpolation.hpp"
#include <openvdb/openvdb.h>
#include <vector>
#include <random>

using namespace am_qadf_native::voxelization;
using namespace am_qadf_native::signal_mapping;

namespace {
void initOpenVDB() {
    openvdb::initialize();
}

std::vector<Point> makePoints(int n, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(1.0f, 64.0f);
    std::vector<Point> pts;
    pts.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i)
        pts.emplace_back(dist(rng), dist(rng), dist(rng));
    return pts;
}

std::vector<float> makeValues(int n) {
    std::vector<float> v;
    v.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i)
        v.push_back(static_cast<float>(i));
    return v;
}
} // namespace

static void BM_NearestNeighbor_Map(benchmark::State& state) {
    initOpenVDB();
    const int n = static_cast<int>(state.range(0));
    std::mt19937 rng(42);
    std::vector<Point> points = makePoints(n, rng);
    std::vector<float> values = makeValues(n);
    UniformVoxelGrid grid(1.0f, 0.0f, 0.0f, 0.0f);
    openvdb::FloatGrid::Ptr g = grid.getGrid();
    NearestNeighborMapper mapper;
    for (auto _ : state) {
        mapper.map(g, points, values);
        benchmark::DoNotOptimize(g);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
}
BENCHMARK(BM_NearestNeighbor_Map)->Arg(100)->Arg(1000)->Arg(5000)->Unit(benchmark::kMicrosecond);

static void BM_Linear_Map(benchmark::State& state) {
    initOpenVDB();
    const int n = static_cast<int>(state.range(0));
    std::mt19937 rng(42);
    std::vector<Point> points = makePoints(n, rng);
    std::vector<float> values = makeValues(n);
    UniformVoxelGrid grid(1.0f, 0.0f, 0.0f, 0.0f);
    openvdb::FloatGrid::Ptr g = grid.getGrid();
    LinearMapper mapper;
    for (auto _ : state) {
        mapper.map(g, points, values);
        benchmark::DoNotOptimize(g);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
}
BENCHMARK(BM_Linear_Map)->Arg(100)->Arg(1000)->Arg(5000)->Unit(benchmark::kMicrosecond);

static void BM_IDW_Map(benchmark::State& state) {
    initOpenVDB();
    const int n = static_cast<int>(state.range(0));
    std::mt19937 rng(42);
    std::vector<Point> points = makePoints(n, rng);
    std::vector<float> values = makeValues(n);
    UniformVoxelGrid grid(1.0f, 0.0f, 0.0f, 0.0f);
    openvdb::FloatGrid::Ptr g = grid.getGrid();
    IDWMapper mapper(2.0f, 10);
    for (auto _ : state) {
        mapper.map(g, points, values);
        benchmark::DoNotOptimize(g);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
}
BENCHMARK(BM_IDW_Map)->Arg(100)->Arg(500)->Arg(1000)->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
