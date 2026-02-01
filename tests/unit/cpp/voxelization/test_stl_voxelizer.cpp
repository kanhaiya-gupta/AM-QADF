/**
 * C++ unit tests for STLVoxelizer (stl_voxelizer.cpp).
 * Uses fixture tests/fixtures/stl/cube.stl (small cube). Path is set at build time
 * via AM_QADF_STL_FIXTURE_PATH so the test finds the file from any cwd (e.g. build/).
 * Voxel size 1.0 mm and half_width 1.0f keep the grid small.
 *
 * Aligned with src/am_qadf_native/voxelization/stl_voxelizer.hpp
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "am_qadf_native/voxelization/stl_voxelizer.hpp"
#include <array>
#include <fstream>
#include <string>
#include <cmath>

using namespace am_qadf_native::voxelization;

#ifdef AM_QADF_STL_FIXTURE_PATH
static const char* const CUBE_FIXTURE_PATH = AM_QADF_STL_FIXTURE_PATH;
#else
static const char* const CUBE_FIXTURE_PATH = "tests/fixtures/stl/cube.stl";
#endif

static bool fixture_exists(const char* path) {
    std::ifstream f(path);
    return f.good();
}

// ---- Missing-file (error) tests ----

TEST_CASE("STLVoxelizer: voxelizeSTL missing file throws", "[voxelization][stl]") {
    STLVoxelizer voxelizer;
    REQUIRE_THROWS_AS(voxelizer.voxelizeSTL("/nonexistent/path.stl", 0.5f), std::exception);
}

TEST_CASE("STLVoxelizer: getSTLBoundingBox missing file throws", "[voxelization][stl]") {
    STLVoxelizer voxelizer;
    std::array<float, 3> bbox_min, bbox_max;
    REQUIRE_THROWS_AS(voxelizer.getSTLBoundingBox("/nonexistent.stl", bbox_min, bbox_max), std::exception);
}

// ---- getSTLBoundingBox with cube fixture ----

TEST_CASE("STLVoxelizer: getSTLBoundingBox with cube fixture returns valid bbox", "[voxelization][stl]") {
    if (!fixture_exists(CUBE_FIXTURE_PATH)) {
        SKIP("Fixture " << CUBE_FIXTURE_PATH << " not found (run from project root or set WORKING_DIRECTORY)");
    }
    STLVoxelizer voxelizer;
    std::array<float, 3> bbox_min, bbox_max;
    voxelizer.getSTLBoundingBox(CUBE_FIXTURE_PATH, bbox_min, bbox_max);

    REQUIRE(bbox_min[0] < bbox_max[0]);
    REQUIRE(bbox_min[1] < bbox_max[1]);
    REQUIRE(bbox_min[2] < bbox_max[2]);
    for (int i = 0; i < 3; ++i) {
        REQUIRE(std::isfinite(bbox_min[i]));
        REQUIRE(std::isfinite(bbox_max[i]));
    }
}

// ---- voxelizeSTL with cube fixture (1.0 mm voxels) ----

TEST_CASE("STLVoxelizer: voxelizeSTL with cube fixture returns non-null grid", "[voxelization][stl]") {
    if (!fixture_exists(CUBE_FIXTURE_PATH)) {
        SKIP("Fixture " << CUBE_FIXTURE_PATH << " not found (run from project root or set WORKING_DIRECTORY)");
    }
    STLVoxelizer voxelizer;
    const float voxel_size = 1.0f;
    const float half_width = 1.0f;
    FloatGridPtr grid = voxelizer.voxelizeSTL(CUBE_FIXTURE_PATH, voxel_size, half_width, false);

    REQUIRE(grid);
    REQUIRE(grid->activeVoxelCount() > 0);
}

TEST_CASE("STLVoxelizer: voxelizeSTL with cube fixture unsigned distance returns non-null grid", "[voxelization][stl]") {
    if (!fixture_exists(CUBE_FIXTURE_PATH)) {
        SKIP("Fixture " << CUBE_FIXTURE_PATH << " not found (run from project root or set WORKING_DIRECTORY)");
    }
    STLVoxelizer voxelizer;
    const float voxel_size = 1.0f;
    const float half_width = 1.0f;
    FloatGridPtr grid = voxelizer.voxelizeSTL(CUBE_FIXTURE_PATH, voxel_size, half_width, true);

    REQUIRE(grid);
    REQUIRE(grid->activeVoxelCount() > 0);
}

TEST_CASE("STLVoxelizer: voxelizeSTL with cube fixture occupancy values in [0,1]", "[voxelization][stl]") {
    if (!fixture_exists(CUBE_FIXTURE_PATH)) {
        SKIP("Fixture " << CUBE_FIXTURE_PATH << " not found (run from project root or set WORKING_DIRECTORY)");
    }
    STLVoxelizer voxelizer;
    FloatGridPtr grid = voxelizer.voxelizeSTL(CUBE_FIXTURE_PATH, 1.0f, 1.0f, false);
    REQUIRE(grid);

    for (auto it = grid->cbeginValueOn(); it; ++it) {
        float v = it.getValue();
        REQUIRE(v >= 0.0f);
        REQUIRE(v <= 1.0f);
    }
}

// ---- voxelizeSTLWithSignals with cube fixture ----

TEST_CASE("STLVoxelizer: voxelizeSTLWithSignals with cube fixture empty points returns occupancy grid", "[voxelization][stl]") {
    if (!fixture_exists(CUBE_FIXTURE_PATH)) {
        SKIP("Fixture " << CUBE_FIXTURE_PATH << " not found (run from project root or set WORKING_DIRECTORY)");
    }
    STLVoxelizer voxelizer;
    std::vector<std::array<float, 3>> points;
    std::vector<float> signal_values;
    FloatGridPtr grid = voxelizer.voxelizeSTLWithSignals(
        CUBE_FIXTURE_PATH, 1.0f, points, signal_values, 1.0f);

    REQUIRE(grid);
    REQUIRE(grid->activeVoxelCount() > 0);
}

TEST_CASE("STLVoxelizer: voxelizeSTLWithSignals points and signal_values size mismatch throws", "[voxelization][stl]") {
    if (!fixture_exists(CUBE_FIXTURE_PATH)) {
        SKIP("Fixture " << CUBE_FIXTURE_PATH << " not found (run from project root or set WORKING_DIRECTORY)");
    }
    STLVoxelizer voxelizer;
    std::vector<std::array<float, 3>> points = {{1.0f, 1.0f, 1.0f}};
    std::vector<float> signal_values = {1.0f, 2.0f};  // size 2 != points.size() 1
    REQUIRE_THROWS_AS(
        voxelizer.voxelizeSTLWithSignals(CUBE_FIXTURE_PATH, 1.0f, points, signal_values, 1.0f),
        std::exception);
}
