/**
 * C++ integration tests: processing pipeline.
 * Chains: raw signal vector → SignalProcessing (normalize, movingAverage, derivative) → verify.
 *
 * Aligned with docs/Tests/Test_New_Plans.md (tests/integration/cpp/).
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "am_qadf_native/processing/signal_processing.hpp"
#include "am_qadf_native/processing/signal_generation.hpp"
#include <vector>
#include <cmath>

using namespace am_qadf_native::processing;

namespace {

std::vector<float> makeRamp(int n) {
    std::vector<float> v;
    v.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i)
        v.push_back(static_cast<float>(i));
    return v;
}

} // namespace

TEST_CASE("Processing pipeline: normalize then verify range", "[integration][processing]") {
    std::vector<float> raw = { 0.0f, 50.0f, 100.0f };
    SignalProcessing proc;
    // Map data [0, 100] to output range [0, 1]
    std::vector<float> out = proc.normalize(raw, 0.0f, 1.0f);
    REQUIRE(out.size() == raw.size());
    REQUIRE(out[0] == Catch::Approx(0.0f));
    REQUIRE(out[1] == Catch::Approx(0.5f));
    REQUIRE(out[2] == Catch::Approx(1.0f));
}

TEST_CASE("Processing pipeline: movingAverage then verify length", "[integration][processing]") {
    std::vector<float> raw = makeRamp(10);
    SignalProcessing proc;
    std::vector<float> out = proc.movingAverage(raw, 3);
    REQUIRE(out.size() == raw.size());
}

TEST_CASE("Processing pipeline: derivative then integral round-trip", "[integration][processing]") {
    std::vector<float> raw = makeRamp(5);
    SignalProcessing proc;
    std::vector<float> d = proc.derivative(raw);
    REQUIRE(d.size() == raw.size());
    std::vector<float> back = proc.integral(d);
    REQUIRE(back.size() == d.size());
}

TEST_CASE("Processing pipeline: signal generation then process", "[integration][processing]") {
    SignalGeneration gen;
    std::vector<float> signal = gen.generateRandom(100, 0.0f, 1.0f, 42u);
    REQUIRE(signal.size() == 100u);
    SignalProcessing proc;
    std::vector<float> normalized = proc.normalize(signal, 0.0f, 1.0f);
    REQUIRE(normalized.size() == signal.size());
}
