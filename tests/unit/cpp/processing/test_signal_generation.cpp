/**
 * C++ unit tests for SignalGeneration (signal_generation.cpp).
 * Tests generateSynthetic, generateGaussian, generateSineWave, generateRandom,
 * generateFromExpression.
 *
 * Aligned with docs/Tests/Test_New_Plans.md (tests/unit/cpp/processing/).
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "am_qadf_native/processing/signal_generation.hpp"
#include <vector>
#include <array>
#include <string>
#include <cmath>

using namespace am_qadf_native::processing;

namespace { const float tol = 1e-5f; }

TEST_CASE("SignalGeneration: generateSynthetic unknown type returns zeros", "[processing][signal_generation]") {
    SignalGeneration gen;
    std::vector<std::array<float, 3>> points = { {0,0,0}, {1,0,0} };
    std::vector<float> out = gen.generateSynthetic(points, "unknown", 1.0f, 1.0f);
    REQUIRE(out.size() == points.size());
    REQUIRE(out[0] == 0.0f);
    REQUIRE(out[1] == 0.0f);
}

TEST_CASE("SignalGeneration: generateSynthetic gaussian", "[processing][signal_generation]") {
    SignalGeneration gen;
    std::vector<std::array<float, 3>> points = { {0,0,0}, {1,0,0} };
    std::vector<float> out = gen.generateSynthetic(points, "gaussian", 1.0f, 1.0f);
    REQUIRE(out.size() == points.size());
    REQUIRE(out[0] >= 0.0f);
    REQUIRE(out[0] <= 1.0f + tol);
}

TEST_CASE("SignalGeneration: generateSynthetic sine", "[processing][signal_generation]") {
    SignalGeneration gen;
    std::vector<std::array<float, 3>> points = { {0,0,0}, {0.25f,0,0} };
    std::vector<float> out = gen.generateSynthetic(points, "sine", 1.0f, 1.0f);
    REQUIRE(out.size() == points.size());
    REQUIRE(out[0] >= -1.0f - tol);
    REQUIRE(out[0] <= 1.0f + tol);
}

TEST_CASE("SignalGeneration: generateSynthetic random", "[processing][signal_generation]") {
    SignalGeneration gen;
    std::vector<std::array<float, 3>> points = { {0,0,0}, {1,1,1} };
    std::vector<float> out = gen.generateSynthetic(points, "random", 1.0f, 1.0f);
    REQUIRE(out.size() == points.size());
    REQUIRE(out[0] >= 0.0f - tol);
    REQUIRE(out[0] <= 1.0f + tol);
}

TEST_CASE("SignalGeneration: generateGaussian center peak", "[processing][signal_generation]") {
    SignalGeneration gen;
    std::vector<std::array<float, 3>> points = { {0,0,0}, {1,0,0} };
    std::array<float, 3> center = { 0, 0, 0 };
    std::vector<float> out = gen.generateGaussian(points, center, 1.0f, 1.0f);
    REQUIRE(out.size() == points.size());
    REQUIRE_THAT(out[0], Catch::Matchers::WithinAbs(1.0f, tol));
    REQUIRE(out[1] < out[0]);
}

TEST_CASE("SignalGeneration: generateGaussian empty points", "[processing][signal_generation]") {
    SignalGeneration gen;
    std::vector<std::array<float, 3>> points;
    std::array<float, 3> center = { 0, 0, 0 };
    std::vector<float> out = gen.generateGaussian(points, center, 1.0f, 1.0f);
    REQUIRE(out.empty());
}

TEST_CASE("SignalGeneration: generateSineWave", "[processing][signal_generation]") {
    SignalGeneration gen;
    std::vector<std::array<float, 3>> points = { {0,0,0}, {0.25f,0,0} };
    std::array<float, 3> dir = { 1, 0, 0 };
    std::vector<float> out = gen.generateSineWave(points, 1.0f, 1.0f, dir);
    REQUIRE(out.size() == points.size());
    REQUIRE_THAT(out[0], Catch::Matchers::WithinAbs(0.0f, tol));
    REQUIRE(out[1] >= -1.0f - tol);
    REQUIRE(out[1] <= 1.0f + tol);
}

TEST_CASE("SignalGeneration: generateSineWave zero direction returns zeros", "[processing][signal_generation]") {
    SignalGeneration gen;
    std::vector<std::array<float, 3>> points = { {1,0,0} };
    std::array<float, 3> dir = { 0, 0, 0 };
    std::vector<float> out = gen.generateSineWave(points, 1.0f, 1.0f, dir);
    REQUIRE(out.size() == points.size());
    REQUIRE(out[0] == 0.0f);
}

TEST_CASE("SignalGeneration: generateRandom size and range", "[processing][signal_generation]") {
    SignalGeneration gen;
    std::vector<float> out = gen.generateRandom(10, 0.0f, 1.0f, 42u);
    REQUIRE(out.size() == 10);
    for (float v : out) {
        REQUIRE(v >= 0.0f - tol);
        REQUIRE(v <= 1.0f + tol);
    }
}

TEST_CASE("SignalGeneration: generateRandom deterministic with seed", "[processing][signal_generation]") {
    SignalGeneration gen;
    std::vector<float> a = gen.generateRandom(5, 0.0f, 1.0f, 123u);
    std::vector<float> b = gen.generateRandom(5, 0.0f, 1.0f, 123u);
    REQUIRE(a.size() == b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        REQUIRE_THAT(a[i], Catch::Matchers::WithinAbs(b[i], tol));
    }
}

TEST_CASE("SignalGeneration: generateFromExpression empty expression returns zeros", "[processing][signal_generation]") {
    SignalGeneration gen;
    std::vector<std::array<float, 3>> points = { {1,0,0} };
    std::vector<float> out = gen.generateFromExpression(points, "");
    REQUIRE(out.size() == points.size());
    REQUIRE(out[0] == 0.0f);
}

TEST_CASE("SignalGeneration: generateFromExpression constant", "[processing][signal_generation]") {
    SignalGeneration gen;
    std::vector<std::array<float, 3>> points = { {1,2,3} };
    std::vector<float> out = gen.generateFromExpression(points, "1");
    REQUIRE(out.size() == points.size());
    REQUIRE_THAT(out[0], Catch::Matchers::WithinAbs(1.0f, tol));
}

TEST_CASE("SignalGeneration: generateFromExpression x+y+z", "[processing][signal_generation]") {
    SignalGeneration gen;
    std::vector<std::array<float, 3>> points = { {1.0f, 2.0f, 3.0f} };
    std::vector<float> out = gen.generateFromExpression(points, "x+y+z");
    REQUIRE(out.size() == points.size());
    REQUIRE_THAT(out[0], Catch::Matchers::WithinAbs(6.0f, tol));
}

TEST_CASE("SignalGeneration: generateFromExpression sqrt", "[processing][signal_generation]") {
    SignalGeneration gen;
    std::vector<std::array<float, 3>> points = { {4.0f, 0, 0} };
    std::vector<float> out = gen.generateFromExpression(points, "sqrt(x)");
    REQUIRE(out.size() == points.size());
    REQUIRE_THAT(out[0], Catch::Matchers::WithinAbs(2.0f, tol));
}
