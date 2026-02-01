/**
 * C++ integration tests: correction pipeline.
 * Chains: build QueryResult → SignalNoiseReduction.reduceNoise / applyGaussianFilter → verify output.
 *
 * Aligned with docs/Tests/Test_New_Plans.md (tests/integration/cpp/).
 */

#include <catch2/catch_test_macros.hpp>
#include "am_qadf_native/query/query_result.hpp"
#include "am_qadf_native/correction/signal_noise_reduction.hpp"
#include <vector>
#include <array>

using namespace am_qadf_native::query;
using namespace am_qadf_native::correction;

namespace {

QueryResult makeSimpleResult() {
    QueryResult r;
    r.points = { {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {2.0f, 0.0f, 0.0f} };
    r.values = { 10.0f, 12.0f, 11.0f };
    r.timestamps = { 0.0f, 1.0f, 2.0f };
    r.layers = { 0, 0, 0 };
    return r;
}

} // namespace

TEST_CASE("Correction pipeline: reduceNoise gaussian", "[integration][correction]") {
    QueryResult raw = makeSimpleResult();
    SignalNoiseReduction reducer;
    QueryResult out = reducer.reduceNoise(raw, "gaussian", 1.0f);
    REQUIRE(out.points.size() == raw.points.size());
    REQUIRE(out.values.size() == raw.values.size());
}

TEST_CASE("Correction pipeline: applyGaussianFilter", "[integration][correction]") {
    QueryResult raw = makeSimpleResult();
    SignalNoiseReduction reducer;
    QueryResult out = reducer.applyGaussianFilter(raw, 0.5f);
    REQUIRE(out.points.size() == raw.points.size());
    REQUIRE(out.values.size() == raw.values.size());
}

TEST_CASE("Correction pipeline: removeOutliers", "[integration][correction]") {
    QueryResult raw = makeSimpleResult();
    raw.values[1] = 1000.0f;  // outlier
    SignalNoiseReduction reducer;
    QueryResult out = reducer.removeOutliers(raw, 3.0f);
    REQUIRE(out.points.size() == raw.points.size());
    REQUIRE(out.values.size() == raw.values.size());
}
