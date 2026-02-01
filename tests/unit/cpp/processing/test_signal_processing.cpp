/**
 * C++ unit tests for SignalProcessing (signal_processing.cpp).
 * Tests normalize, movingAverage, derivative, integral, fft, ifft, frequencyFilter.
 *
 * Aligned with docs/Tests/Test_New_Plans.md (tests/unit/cpp/processing/).
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "am_qadf_native/processing/signal_processing.hpp"
#include <vector>
#include <complex>
#include <string>
#include <cmath>

using namespace am_qadf_native::processing;

namespace { const float tol = 1e-5f; }

TEST_CASE("SignalProcessing: normalize empty returns copy", "[processing][signal_processing]") {
    SignalProcessing proc;
    std::vector<float> values;
    std::vector<float> out = proc.normalize(values, 0.0f, 1.0f);
    REQUIRE(out.empty());
}

TEST_CASE("SignalProcessing: normalize constant range returns copy", "[processing][signal_processing]") {
    SignalProcessing proc;
    std::vector<float> values = { 5.0f, 5.0f, 5.0f };
    std::vector<float> out = proc.normalize(values, 0.0f, 1.0f);
    REQUIRE(out.size() == values.size());
    REQUIRE(out[0] == 5.0f);
}

TEST_CASE("SignalProcessing: normalize maps to range", "[processing][signal_processing]") {
    SignalProcessing proc;
    std::vector<float> values = { 0.0f, 1.0f, 2.0f };
    std::vector<float> out = proc.normalize(values, 0.0f, 1.0f);
    REQUIRE(out.size() == 3);
    REQUIRE_THAT(out[0], Catch::Matchers::WithinAbs(0.0f, tol));
    REQUIRE_THAT(out[1], Catch::Matchers::WithinAbs(0.5f, tol));
    REQUIRE_THAT(out[2], Catch::Matchers::WithinAbs(1.0f, tol));
}

TEST_CASE("SignalProcessing: movingAverage empty returns copy", "[processing][signal_processing]") {
    SignalProcessing proc;
    std::vector<float> values;
    std::vector<float> out = proc.movingAverage(values, 5);
    REQUIRE(out.empty());
}

TEST_CASE("SignalProcessing: movingAverage window_size <= 0 returns copy", "[processing][signal_processing]") {
    SignalProcessing proc;
    std::vector<float> values = { 1.0f, 2.0f, 3.0f };
    std::vector<float> out = proc.movingAverage(values, 0);
    REQUIRE(out.size() == values.size());
    REQUIRE(out[0] == 1.0f);
}

TEST_CASE("SignalProcessing: movingAverage smooths", "[processing][signal_processing]") {
    SignalProcessing proc;
    std::vector<float> values = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
    std::vector<float> out = proc.movingAverage(values, 3);
    REQUIRE(out.size() == values.size());
    REQUIRE(out[2] >= 1.0f - tol);
    REQUIRE(out[2] <= 5.0f + tol);
}

TEST_CASE("SignalProcessing: derivative size < 2 returns zeros", "[processing][signal_processing]") {
    SignalProcessing proc;
    std::vector<float> values = { 1.0f };
    std::vector<float> out = proc.derivative(values);
    REQUIRE(out.size() == 1);
    REQUIRE(out[0] == 0.0f);
}

TEST_CASE("SignalProcessing: derivative", "[processing][signal_processing]") {
    SignalProcessing proc;
    std::vector<float> values = { 1.0f, 2.0f, 4.0f };
    std::vector<float> out = proc.derivative(values);
    REQUIRE(out.size() == 3);
    REQUIRE(out[0] == 0.0f);
    REQUIRE_THAT(out[1], Catch::Matchers::WithinAbs(1.0f, tol));
    REQUIRE_THAT(out[2], Catch::Matchers::WithinAbs(2.0f, tol));
}

TEST_CASE("SignalProcessing: integral empty returns copy", "[processing][signal_processing]") {
    SignalProcessing proc;
    std::vector<float> values;
    std::vector<float> out = proc.integral(values);
    REQUIRE(out.empty());
}

TEST_CASE("SignalProcessing: integral cumulative sum", "[processing][signal_processing]") {
    SignalProcessing proc;
    std::vector<float> values = { 1.0f, 1.0f, 1.0f };
    std::vector<float> out = proc.integral(values);
    REQUIRE(out.size() == 3);
    REQUIRE_THAT(out[0], Catch::Matchers::WithinAbs(1.0f, tol));
    REQUIRE_THAT(out[1], Catch::Matchers::WithinAbs(2.0f, tol));
    REQUIRE_THAT(out[2], Catch::Matchers::WithinAbs(3.0f, tol));
}

TEST_CASE("SignalProcessing: fft empty returns empty", "[processing][signal_processing]") {
    SignalProcessing proc;
    std::vector<float> values;
    std::vector<std::complex<float>> out = proc.fft(values);
    REQUIRE(out.empty());
}

TEST_CASE("SignalProcessing: fft returns spectrum of same size", "[processing][signal_processing]") {
    SignalProcessing proc;
    std::vector<float> values = { 1.0f, 0.0f, 1.0f, 0.0f };
    std::vector<std::complex<float>> out = proc.fft(values);
    REQUIRE(out.size() == values.size());
}

TEST_CASE("SignalProcessing: ifft empty returns empty", "[processing][signal_processing]") {
    SignalProcessing proc;
    std::vector<std::complex<float>> spectrum;
    std::vector<float> out = proc.ifft(spectrum);
    REQUIRE(out.empty());
}

TEST_CASE("SignalProcessing: ifft returns values of same size", "[processing][signal_processing]") {
    SignalProcessing proc;
    std::vector<std::complex<float>> spectrum(4, std::complex<float>(0.0f, 0.0f));
    spectrum[0] = std::complex<float>(1.0f, 0.0f);
    std::vector<float> out = proc.ifft(spectrum);
    REQUIRE(out.size() == spectrum.size());
}

TEST_CASE("SignalProcessing: frequencyFilter empty returns copy", "[processing][signal_processing]") {
    SignalProcessing proc;
    std::vector<float> values;
    std::vector<float> out = proc.frequencyFilter(values, 0.1f, "lowpass");
    REQUIRE(out.empty());
}

TEST_CASE("SignalProcessing: frequencyFilter cutoff <= 0 returns copy", "[processing][signal_processing]") {
    SignalProcessing proc;
    std::vector<float> values = { 1.0f, 2.0f, 3.0f };
    std::vector<float> out = proc.frequencyFilter(values, 0.0f, "lowpass");
    REQUIRE(out.size() == values.size());
}

TEST_CASE("SignalProcessing: frequencyFilter lowpass returns same size", "[processing][signal_processing]") {
    SignalProcessing proc;
    std::vector<float> values = { 1.0f, 0.0f, 1.0f, 0.0f, 1.0f };
    std::vector<float> out = proc.frequencyFilter(values, 0.25f, "lowpass");
    REQUIRE(out.size() == values.size());
}

TEST_CASE("SignalProcessing: frequencyFilter highpass", "[processing][signal_processing]") {
    SignalProcessing proc;
    std::vector<float> values = { 1.0f, 2.0f, 1.0f, 2.0f };
    std::vector<float> out = proc.frequencyFilter(values, 0.1f, "highpass");
    REQUIRE(out.size() == values.size());
}

TEST_CASE("SignalProcessing: frequencyFilter bandpass", "[processing][signal_processing]") {
    SignalProcessing proc;
    std::vector<float> values = { 1.0f, 0.0f, 1.0f, 0.0f };
    std::vector<float> out = proc.frequencyFilter(values, 0.1f, "bandpass");
    REQUIRE(out.size() == values.size());
}
