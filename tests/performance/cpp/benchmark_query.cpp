/**
 * C++ performance benchmarks: query (QueryResult build, point/value fill).
 * Aligned with docs/Tests/Test_New_Plans.md (tests/performance/cpp/).
 * Does not require MongoDB; benchmarks in-memory QueryResult construction.
 */

#include <benchmark/benchmark.h>
#include "am_qadf_native/query/query_result.hpp"
#include <vector>
#include <array>
#include <random>

using namespace am_qadf_native::query;

namespace {

QueryResult makeQueryResult(int n) {
    QueryResult r;
    r.points.reserve(static_cast<size_t>(n));
    r.values.reserve(static_cast<size_t>(n));
    r.timestamps.reserve(static_cast<size_t>(n));
    r.layers.reserve(static_cast<size_t>(n));
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> pos(0.0f, 100.0f);
    for (int i = 0; i < n; ++i) {
        r.points.push_back({pos(rng), pos(rng), pos(rng)});
        r.values.push_back(static_cast<float>(i));
        r.timestamps.push_back(static_cast<float>(i) * 0.01f);
        r.layers.push_back(i % 10);
    }
    return r;
}

} // namespace

static void BM_QueryResult_Build(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    for (auto _ : state) {
        QueryResult r = makeQueryResult(n);
        benchmark::DoNotOptimize(r.points.size());
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
}
BENCHMARK(BM_QueryResult_Build)->Arg(100)->Arg(1000)->Arg(10000)->Arg(100000)->Unit(benchmark::kMicrosecond);

static void BM_QueryResult_NumPoints(benchmark::State& state) {
    const int n = 10000;
    QueryResult r = makeQueryResult(n);
    int64_t total = 0;
    for (auto _ : state) {
        total += r.num_points();
    }
    benchmark::DoNotOptimize(total);
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_QueryResult_NumPoints);

static void BM_QueryResult_EmptyCheck(benchmark::State& state) {
    QueryResult r = makeQueryResult(1000);
    bool e = false;
    for (auto _ : state) {
        e = e ^ r.empty();
    }
    benchmark::DoNotOptimize(e);
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_QueryResult_EmptyCheck);

BENCHMARK_MAIN();
