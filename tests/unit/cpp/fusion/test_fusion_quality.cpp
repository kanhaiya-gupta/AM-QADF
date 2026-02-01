/**
 * C++ unit tests for FusionQualityAssessor (fusion_quality.cpp).
 * Tests assess() with small OpenVDB grids: coverage, per-signal accuracy,
 * signal consistency, quality_score.
 *
 * Aligned with docs/Infrastructure/Fusion_Quality_CPP_Plan.md
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "am_qadf_native/fusion/fusion_quality.hpp"
#include <openvdb/openvdb.h>
#include <openvdb/Grid.h>
#include <openvdb/math/Transform.h>
#include <openvdb/math/Coord.h>
#include <map>
#include <string>
#include <cmath>

using namespace am_qadf_native::fusion;
using Catch::Matchers::WithinRel;

namespace {

void initOpenVDB() {
    openvdb::initialize();
}

FloatGridPtr makeGridWithValues(float voxel_size,
    const std::vector<std::tuple<int, int, int, float>>& coord_values) {
    auto grid = openvdb::FloatGrid::create(0.0f);
    grid->setTransform(openvdb::math::Transform::createLinearTransform(voxel_size));
    for (const auto& t : coord_values) {
        grid->tree().setValue(
            openvdb::Coord(std::get<0>(t), std::get<1>(t), std::get<2>(t)),
            std::get<3>(t)
        );
    }
    return grid;
}

} // namespace

TEST_CASE("FusionQualityAssessor: assess single fused grid no sources", "[fusion][fusion_quality]") {
    initOpenVDB();
    FusionQualityAssessor assessor;
    // Fused grid with one active voxel at (0,0,0) = 10.0
    auto fused = makeGridWithValues(1.0f, {{0, 0, 0, 10.0f}});
    std::map<std::string, FloatGridPtr> sources;
    std::map<std::string, float> weights;

    FusionQualityResult result = assessor.assess(fused, sources, weights);

    REQUIRE(result.coverage_ratio >= 0.0f);
    REQUIRE(result.coverage_ratio <= 1.0f);
    REQUIRE(result.fusion_completeness == result.coverage_ratio);
    REQUIRE(result.per_signal_accuracy.empty());
    REQUIRE(result.fusion_accuracy == 0.0f);
    REQUIRE(result.signal_consistency == 0.0f);
    REQUIRE_THAT(result.quality_score, WithinRel(0.3f * result.coverage_ratio, 0.01f));
}

TEST_CASE("FusionQualityAssessor: assess fused vs one source", "[fusion][fusion_quality]") {
    initOpenVDB();
    FusionQualityAssessor assessor;
    // Same active voxels: (0,0,0), (1,1,1) - perfect match
    auto fused = makeGridWithValues(1.0f, {{0, 0, 0, 5.0f}, {1, 1, 1, 10.0f}});
    auto src = makeGridWithValues(1.0f, {{0, 0, 0, 5.0f}, {1, 1, 1, 10.0f}});
    std::map<std::string, FloatGridPtr> sources;
    sources["s1"] = src;
    std::map<std::string, float> weights;

    FusionQualityResult result = assessor.assess(fused, sources, weights);

    REQUIRE(result.per_signal_accuracy.count("s1") == 1u);
    // Perfect match -> correlation 1.0, accuracy 1.0
    REQUIRE(result.per_signal_accuracy.at("s1") >= 0.99f);
    REQUIRE(result.fusion_accuracy >= 0.99f);
    REQUIRE(result.signal_consistency >= 0.0f);
    REQUIRE(result.quality_score >= 0.0f);
    REQUIRE(result.quality_score <= 1.0f);
}

TEST_CASE("FusionQualityAssessor: assess fused vs two sources", "[fusion][fusion_quality]") {
    initOpenVDB();
    FusionQualityAssessor assessor;
    auto fused = makeGridWithValues(1.0f, {{0, 0, 0, 2.0f}, {1, 0, 0, 4.0f}, {0, 1, 0, 6.0f}});
    auto s1 = makeGridWithValues(1.0f, {{0, 0, 0, 1.0f}, {1, 0, 0, 3.0f}, {0, 1, 0, 5.0f}});
    auto s2 = makeGridWithValues(1.0f, {{0, 0, 0, 3.0f}, {1, 0, 0, 5.0f}, {0, 1, 0, 7.0f}});
    std::map<std::string, FloatGridPtr> sources;
    sources["a"] = s1;
    sources["b"] = s2;
    std::map<std::string, float> weights;

    FusionQualityResult result = assessor.assess(fused, sources, weights);

    REQUIRE(result.per_signal_accuracy.size() == 2u);
    REQUIRE(result.per_signal_accuracy.count("a") == 1u);
    REQUIRE(result.per_signal_accuracy.count("b") == 1u);
    REQUIRE(result.fusion_accuracy >= 0.0f);
    REQUIRE(result.fusion_accuracy <= 1.0f);
    REQUIRE(result.quality_score >= 0.0f);
    REQUIRE(result.quality_score <= 1.0f);
}

TEST_CASE("FusionQualityAssessor: assess with weights", "[fusion][fusion_quality]") {
    initOpenVDB();
    FusionQualityAssessor assessor;
    auto fused = makeGridWithValues(1.0f, {{0, 0, 0, 10.0f}});
    auto src = makeGridWithValues(1.0f, {{0, 0, 0, 12.0f}});
    std::map<std::string, FloatGridPtr> sources;
    sources["s1"] = src;
    std::map<std::string, float> weights;
    weights["s1"] = 0.5f;

    FusionQualityResult result = assessor.assess(fused, sources, weights);

    REQUIRE(result.per_signal_accuracy.count("s1") == 1u);
    if (result.has_residual_summary) {
        REQUIRE(result.residual_mean >= 0.0f);
        REQUIRE(result.residual_max >= 0.0f);
    }
}
