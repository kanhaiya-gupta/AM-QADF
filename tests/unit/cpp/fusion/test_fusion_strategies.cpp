/**
 * C++ unit tests for FusionStrategy implementations (fusion_strategies.cpp).
 * Tests WeightedAverageStrategy, MaxStrategy, MinStrategy, MedianStrategy.
 *
 * Aligned with src/am_qadf_native/fusion/fusion_strategies.hpp
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "am_qadf_native/fusion/fusion_strategies.hpp"
#include <vector>
#include <cmath>

using namespace am_qadf_native::fusion;

// ---------------------------------------------------------------------------
// WeightedAverageStrategy
// ---------------------------------------------------------------------------

TEST_CASE("WeightedAverageStrategy: fuseValues equal weights", "[fusion][fusion_strategies]") {
    std::vector<float> weights = { 0.5f, 0.5f };
    WeightedAverageStrategy strategy(weights);
    std::vector<float> values = { 10.0f, 30.0f };
    float result = strategy.fuseValues(values);
    REQUIRE_THAT(result, Catch::Matchers::WithinAbs(20.0f, 1e-6f));
}

TEST_CASE("WeightedAverageStrategy: fuseValues unequal weights", "[fusion][fusion_strategies]") {
    std::vector<float> weights = { 0.25f, 0.75f };
    WeightedAverageStrategy strategy(weights);
    std::vector<float> values = { 10.0f, 30.0f };
    float result = strategy.fuseValues(values);
    REQUIRE_THAT(result, Catch::Matchers::WithinAbs(25.0f, 1e-6f));
}

TEST_CASE("WeightedAverageStrategy: fuseValues single value", "[fusion][fusion_strategies]") {
    std::vector<float> weights = { 1.0f };
    WeightedAverageStrategy strategy(weights);
    std::vector<float> values = { 42.0f };
    float result = strategy.fuseValues(values);
    REQUIRE_THAT(result, Catch::Matchers::WithinAbs(42.0f, 1e-6f));
}

TEST_CASE("WeightedAverageStrategy: values and weights size mismatch throws", "[fusion][fusion_strategies]") {
    std::vector<float> weights = { 0.5f, 0.5f };
    WeightedAverageStrategy strategy(weights);
    std::vector<float> values = { 1.0f };
    REQUIRE_THROWS_AS(strategy.fuseValues(values), std::invalid_argument);
}

TEST_CASE("WeightedAverageStrategy: zero sum weights returns zero", "[fusion][fusion_strategies]") {
    std::vector<float> weights = { 0.0f, 0.0f };
    WeightedAverageStrategy strategy(weights);
    std::vector<float> values = { 10.0f, 20.0f };
    float result = strategy.fuseValues(values);
    REQUIRE_THAT(result, Catch::Matchers::WithinAbs(0.0f, 1e-6f));
}

// ---------------------------------------------------------------------------
// MaxStrategy
// ---------------------------------------------------------------------------

TEST_CASE("MaxStrategy: fuseValues single", "[fusion][fusion_strategies]") {
    MaxStrategy strategy;
    std::vector<float> values = { 42.0f };
    REQUIRE_THAT(strategy.fuseValues(values), Catch::Matchers::WithinAbs(42.0f, 1e-6f));
}

TEST_CASE("MaxStrategy: fuseValues multiple", "[fusion][fusion_strategies]") {
    MaxStrategy strategy;
    std::vector<float> values = { 10.0f, 50.0f, 30.0f, 20.0f };
    REQUIRE_THAT(strategy.fuseValues(values), Catch::Matchers::WithinAbs(50.0f, 1e-6f));
}

TEST_CASE("MaxStrategy: fuseValues empty returns zero", "[fusion][fusion_strategies]") {
    MaxStrategy strategy;
    std::vector<float> values;
    REQUIRE_THAT(strategy.fuseValues(values), Catch::Matchers::WithinAbs(0.0f, 1e-6f));
}

TEST_CASE("MaxStrategy: fuseValues negative", "[fusion][fusion_strategies]") {
    MaxStrategy strategy;
    std::vector<float> values = { -10.0f, -5.0f, -20.0f };
    REQUIRE_THAT(strategy.fuseValues(values), Catch::Matchers::WithinAbs(-5.0f, 1e-6f));
}

// ---------------------------------------------------------------------------
// MinStrategy
// ---------------------------------------------------------------------------

TEST_CASE("MinStrategy: fuseValues single", "[fusion][fusion_strategies]") {
    MinStrategy strategy;
    std::vector<float> values = { 42.0f };
    REQUIRE_THAT(strategy.fuseValues(values), Catch::Matchers::WithinAbs(42.0f, 1e-6f));
}

TEST_CASE("MinStrategy: fuseValues multiple", "[fusion][fusion_strategies]") {
    MinStrategy strategy;
    std::vector<float> values = { 10.0f, 50.0f, 30.0f, 5.0f };
    REQUIRE_THAT(strategy.fuseValues(values), Catch::Matchers::WithinAbs(5.0f, 1e-6f));
}

TEST_CASE("MinStrategy: fuseValues empty returns zero", "[fusion][fusion_strategies]") {
    MinStrategy strategy;
    std::vector<float> values;
    REQUIRE_THAT(strategy.fuseValues(values), Catch::Matchers::WithinAbs(0.0f, 1e-6f));
}

TEST_CASE("MinStrategy: fuseValues negative", "[fusion][fusion_strategies]") {
    MinStrategy strategy;
    std::vector<float> values = { -10.0f, -5.0f, -20.0f };
    REQUIRE_THAT(strategy.fuseValues(values), Catch::Matchers::WithinAbs(-20.0f, 1e-6f));
}

// ---------------------------------------------------------------------------
// MedianStrategy
// ---------------------------------------------------------------------------

TEST_CASE("MedianStrategy: fuseValues single", "[fusion][fusion_strategies]") {
    MedianStrategy strategy;
    std::vector<float> values = { 42.0f };
    REQUIRE_THAT(strategy.fuseValues(values), Catch::Matchers::WithinAbs(42.0f, 1e-6f));
}

TEST_CASE("MedianStrategy: fuseValues odd count", "[fusion][fusion_strategies]") {
    MedianStrategy strategy;
    std::vector<float> values = { 10.0f, 20.0f, 30.0f };
    REQUIRE_THAT(strategy.fuseValues(values), Catch::Matchers::WithinAbs(20.0f, 1e-6f));
}

TEST_CASE("MedianStrategy: fuseValues even count", "[fusion][fusion_strategies]") {
    MedianStrategy strategy;
    std::vector<float> values = { 10.0f, 20.0f, 30.0f, 40.0f };
    REQUIRE_THAT(strategy.fuseValues(values), Catch::Matchers::WithinAbs(25.0f, 1e-6f));
}

TEST_CASE("MedianStrategy: fuseValues empty returns zero", "[fusion][fusion_strategies]") {
    MedianStrategy strategy;
    std::vector<float> values;
    REQUIRE_THAT(strategy.fuseValues(values), Catch::Matchers::WithinAbs(0.0f, 1e-6f));
}

TEST_CASE("MedianStrategy: fuseValues unsorted input", "[fusion][fusion_strategies]") {
    MedianStrategy strategy;
    std::vector<float> values = { 30.0f, 10.0f, 20.0f };
    REQUIRE_THAT(strategy.fuseValues(values), Catch::Matchers::WithinAbs(20.0f, 1e-6f));
}
