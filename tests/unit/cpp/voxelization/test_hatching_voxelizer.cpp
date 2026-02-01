/**
 * C++ unit tests for HatchingVoxelizer (hatching_voxelizer.cpp).
 * Tests voxelizeHatchingPaths, voxelizeContourPaths, voxelizeVectors.
 *
 * Aligned with src/am_qadf_native/voxelization/hatching_voxelizer.hpp
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "am_qadf_native/voxelization/hatching_voxelizer.hpp"
#include <vector>
#include <array>
#include <string>

using namespace am_qadf_native::voxelization;

// ---------------------------------------------------------------------------
// voxelizeHatchingPaths (points)
// ---------------------------------------------------------------------------

TEST_CASE("HatchingVoxelizer: voxelizeHatchingPaths empty points returns empty or minimal grids", "[voxelization][hatching]") {
    HatchingVoxelizer voxelizer;
    std::vector<HatchingPoint> empty;
    auto result = voxelizer.voxelizeHatchingPaths(empty, 0.5f);
    REQUIRE(result.count("power") == 1);
    REQUIRE(result.count("velocity") == 1);
    REQUIRE(result.count("energy") == 1);
    REQUIRE(result["power"] != nullptr);
}

TEST_CASE("HatchingVoxelizer: voxelizeHatchingPaths single point", "[voxelization][hatching]") {
    HatchingVoxelizer voxelizer;
    std::vector<HatchingPoint> points = {
        { 1.0f, 1.0f, 1.0f, 100.0f, 500.0f, 0.2f }
    };
    std::array<float, 3> bbox_min = { 0.0f, 0.0f, 0.0f };
    std::array<float, 3> bbox_max = { 10.0f, 10.0f, 10.0f };

    auto result = voxelizer.voxelizeHatchingPaths(points, 0.2f, 0.1f, bbox_min, bbox_max);
    REQUIRE(result.count("power") == 1);
    REQUIRE(result["power"] != nullptr);
    REQUIRE(result["velocity"] != nullptr);
    REQUIRE(result["energy"] != nullptr);
}

TEST_CASE("HatchingVoxelizer: voxelizeHatchingPaths line segment", "[voxelization][hatching]") {
    HatchingVoxelizer voxelizer;
    std::vector<HatchingPoint> points = {
        { 0.0f, 0.0f, 0.0f, 100.0f, 500.0f, 0.2f },
        { 2.0f, 0.0f, 0.0f, 100.0f, 500.0f, 0.2f }
    };
    std::array<float, 3> bbox_min = { -1.0f, -1.0f, -1.0f };
    std::array<float, 3> bbox_max = { 5.0f, 5.0f, 5.0f };

    auto result = voxelizer.voxelizeHatchingPaths(points, 0.25f, 0.1f, bbox_min, bbox_max);
    REQUIRE(result["power"] != nullptr);
    REQUIRE(result["velocity"] != nullptr);
    REQUIRE(result["energy"] != nullptr);
}

TEST_CASE("HatchingVoxelizer: voxelizeContourPaths closed loop", "[voxelization][hatching]") {
    HatchingVoxelizer voxelizer;
    std::vector<HatchingPoint> points = {
        { 0.0f, 0.0f, 0.0f, 80.0f, 400.0f, 0.2f },
        { 1.0f, 0.0f, 0.0f, 80.0f, 400.0f, 0.2f },
        { 1.0f, 1.0f, 0.0f, 80.0f, 400.0f, 0.2f },
        { 0.0f, 1.0f, 0.0f, 80.0f, 400.0f, 0.2f },
        { 0.0f, 0.0f, 0.0f, 80.0f, 400.0f, 0.2f }
    };
    std::array<float, 3> bbox_min = { -0.5f, -0.5f, -0.5f };
    std::array<float, 3> bbox_max = { 2.0f, 2.0f, 0.5f };

    auto result = voxelizer.voxelizeContourPaths(points, 0.2f, 0.1f, bbox_min, bbox_max);
    REQUIRE(result["power"] != nullptr);
    REQUIRE(result["energy"] != nullptr);
}

// ---------------------------------------------------------------------------
// voxelizeVectors
// ---------------------------------------------------------------------------

TEST_CASE("HatchingVoxelizer: voxelizeVectors empty returns minimal grids", "[voxelization][hatching]") {
    HatchingVoxelizer voxelizer;
    std::vector<HatchingVector> empty;
    std::array<float, 3> bbox_min = { 0.0f, 0.0f, 0.0f };
    std::array<float, 3> bbox_max = { 10.0f, 10.0f, 10.0f };

    auto result = voxelizer.voxelizeVectors(empty, 0.5f, 0.1f, bbox_min, bbox_max);
    REQUIRE(result.count("power") == 1);
    REQUIRE(result["power"] != nullptr);
}

TEST_CASE("HatchingVoxelizer: voxelizeVectors single vector", "[voxelization][hatching]") {
    HatchingVoxelizer voxelizer;
    std::vector<HatchingVector> vectors = {
        { 0.0f, 0.0f, 0.0f, 3.0f, 0.0f, 0.0f, 100.0f, 500.0f, 0.2f }
    };
    std::array<float, 3> bbox_min = { -1.0f, -1.0f, -1.0f };
    std::array<float, 3> bbox_max = { 5.0f, 5.0f, 5.0f };

    auto result = voxelizer.voxelizeVectors(vectors, 0.25f, 0.1f, bbox_min, bbox_max);
    REQUIRE(result["power"] != nullptr);
    REQUIRE(result["velocity"] != nullptr);
    REQUIRE(result["energy"] != nullptr);
}

TEST_CASE("HatchingVoxelizer: voxelizeMultiLayerHatching multiple layers", "[voxelization][hatching]") {
    HatchingVoxelizer voxelizer;
    std::vector<std::vector<HatchingPoint>> layers = {
        { { 0.0f, 0.0f, 0.0f, 100.0f, 500.0f, 0.2f }, { 1.0f, 0.0f, 0.0f, 100.0f, 500.0f, 0.2f } },
        { { 0.0f, 0.0f, 1.0f, 90.0f, 450.0f, 0.2f }, { 1.0f, 0.0f, 1.0f, 90.0f, 450.0f, 0.2f } }
    };
    std::array<float, 3> bbox_min = { -0.5f, -0.5f, -0.5f };
    std::array<float, 3> bbox_max = { 2.0f, 2.0f, 2.0f };

    auto result = voxelizer.voxelizeMultiLayerHatching(layers, 0.2f, 0.1f, bbox_min, bbox_max);
    REQUIRE(result["power"] != nullptr);
    REQUIRE(result["energy"] != nullptr);
}
