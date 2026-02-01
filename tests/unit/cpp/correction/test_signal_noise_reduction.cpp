/**
 * C++ unit tests for SignalNoiseReduction (signal_noise_reduction.cpp).
 * Tests reduceNoise, applyGaussianFilter, applySavitzkyGolay, removeOutliers.
 *
 * Aligned with docs/Tests/Test_New_Plans.md (tests/unit/cpp/correction/).
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "am_qadf_native/correction/signal_noise_reduction.hpp"
#include "am_qadf_native/query/query_result.hpp"
#include <vector>
#include <array>
#include <string>

using namespace am_qadf_native::correction;

namespace {

QueryResult makeQueryResult(const std::vector<float>& values) {
    QueryResult r;
    r.values = values;
    r.points.reserve(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        r.points.push_back({ static_cast<float>(i), 0.0f, 0.0f });
    }
    return r;
}

} // namespace

TEST_CASE("SignalNoiseReduction: reduceNoise unknown method returns original", "[correction][signal_noise_reduction]") {
    SignalNoiseReduction reducer;
    QueryResult raw = makeQueryResult({ 1.0f, 2.0f, 3.0f });
    QueryResult out = reducer.reduceNoise(raw, "unknown", 1.0f);
    REQUIRE(out.values.size() == raw.values.size());
    REQUIRE(out.values[0] == 1.0f);
}

TEST_CASE("SignalNoiseReduction: reduceNoise gaussian delegates", "[correction][signal_noise_reduction]") {
    SignalNoiseReduction reducer;
    QueryResult raw = makeQueryResult({ 1.0f, 2.0f, 3.0f });
    QueryResult out = reducer.reduceNoise(raw, "gaussian", 1.0f);
    REQUIRE(out.values.size() == raw.values.size());
}

TEST_CASE("SignalNoiseReduction: applyGaussianFilter empty values returns copy", "[correction][signal_noise_reduction]") {
    SignalNoiseReduction reducer;
    QueryResult data = makeQueryResult({});
    QueryResult out = reducer.applyGaussianFilter(data, 1.0f);
    REQUIRE(out.values.empty());
}

TEST_CASE("SignalNoiseReduction: applyGaussianFilter sigma <= 0 returns copy", "[correction][signal_noise_reduction]") {
    SignalNoiseReduction reducer;
    QueryResult data = makeQueryResult({ 1.0f, 2.0f, 3.0f });
    QueryResult out = reducer.applyGaussianFilter(data, 0.0f);
    REQUIRE(out.values.size() == data.values.size());
    REQUIRE(out.values[0] == 1.0f);
}

TEST_CASE("SignalNoiseReduction: applyGaussianFilter smooths values", "[correction][signal_noise_reduction]") {
    SignalNoiseReduction reducer;
    QueryResult data = makeQueryResult({ 1.0f, 2.0f, 3.0f, 2.0f, 1.0f });
    QueryResult out = reducer.applyGaussianFilter(data, 1.0f);
    REQUIRE(out.values.size() == data.values.size());
    REQUIRE(out.values[2] >= 1.0f - 1e-5f);
    REQUIRE(out.values[2] <= 3.0f + 1e-5f);
}

TEST_CASE("SignalNoiseReduction: applySavitzkyGolay empty returns copy", "[correction][signal_noise_reduction]") {
    SignalNoiseReduction reducer;
    QueryResult data = makeQueryResult({});
    QueryResult out = reducer.applySavitzkyGolay(data, 5, 3);
    REQUIRE(out.values.empty());
}

TEST_CASE("SignalNoiseReduction: applySavitzkyGolay even window returns copy", "[correction][signal_noise_reduction]") {
    SignalNoiseReduction reducer;
    QueryResult data = makeQueryResult({ 1.0f, 2.0f, 3.0f });
    QueryResult out = reducer.applySavitzkyGolay(data, 4, 3);
    REQUIRE(out.values.size() == data.values.size());
    REQUIRE(out.values[0] == 1.0f);
}

TEST_CASE("SignalNoiseReduction: applySavitzkyGolay valid window", "[correction][signal_noise_reduction]") {
    SignalNoiseReduction reducer;
    QueryResult data = makeQueryResult({ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f });
    QueryResult out = reducer.applySavitzkyGolay(data, 5, 3);
    REQUIRE(out.values.size() == data.values.size());
}

TEST_CASE("SignalNoiseReduction: removeOutliers", "[correction][signal_noise_reduction]") {
    SignalNoiseReduction reducer;
    QueryResult data = makeQueryResult({ 1.0f, 2.0f, 3.0f, 2.0f, 1.0f });
    QueryResult out = reducer.removeOutliers(data, 3.0f);
    REQUIRE(out.values.size() <= data.values.size());
    REQUIRE(!out.values.empty());
}

TEST_CASE("SignalNoiseReduction: reduceNoise outlier_removal delegates", "[correction][signal_noise_reduction]") {
    SignalNoiseReduction reducer;
    QueryResult raw = makeQueryResult({ 1.0f, 2.0f, 3.0f });
    QueryResult out = reducer.reduceNoise(raw, "outlier_removal", 1.0f);
    REQUIRE(out.values.size() <= raw.values.size());
}
