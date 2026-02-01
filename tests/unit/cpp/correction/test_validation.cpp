/**
 * C++ unit tests for Validation (validation.cpp).
 * Tests validateGrid, validateSignalData, validateCoordinates, checkConsistency.
 *
 * Aligned with docs/Tests/Test_New_Plans.md (tests/unit/cpp/correction/).
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "am_qadf_native/correction/validation.hpp"
#include <openvdb/openvdb.h>
#include <openvdb/Grid.h>
#include <openvdb/math/Transform.h>
#include <openvdb/math/Coord.h>
#include <vector>
#include <array>
#include <cmath>

using namespace am_qadf_native::correction;

namespace {

void initOpenVDB() {
    openvdb::initialize();
}

FloatGridPtr makeGrid(float voxel_size, int ix, int iy, int iz, float value) {
    auto grid = openvdb::FloatGrid::create(0.0f);
    grid->setTransform(openvdb::math::Transform::createLinearTransform(voxel_size));
    grid->tree().setValue(openvdb::Coord(ix, iy, iz), value);
    return grid;
}

} // namespace

TEST_CASE("Validation: validateGrid returns result with metrics", "[correction][validation]") {
    initOpenVDB();
    Validation val;
    FloatGridPtr grid = makeGrid(1.0f, 0, 0, 0, 5.0f);
    ValidationResult r = val.validateGrid(grid);
    REQUIRE(r.metrics.count("min") > 0);
    REQUIRE(r.metrics.count("max") > 0);
    REQUIRE(r.metrics.count("mean") > 0);
    REQUIRE(r.metrics.count("std") > 0);
}

TEST_CASE("Validation: validateSignalData valid range", "[correction][validation]") {
    Validation val;
    std::vector<float> values = { 0.0f, 0.5f, 1.0f };
    ValidationResult r = val.validateSignalData(values, 0.0f, 1.0f);
    REQUIRE(r.is_valid);
}

TEST_CASE("Validation: validateSignalData NaN invalid", "[correction][validation]") {
    Validation val;
    std::vector<float> values = { 0.0f, std::nanf(""), 1.0f };
    ValidationResult r = val.validateSignalData(values, 0.0f, 1.0f);
    REQUIRE_FALSE(r.is_valid);
    REQUIRE(!r.errors.empty());
}

TEST_CASE("Validation: validateCoordinates in bbox", "[correction][validation]") {
    Validation val;
    std::vector<std::array<float, 3>> points = { {0.5f, 0.5f, 0.5f} };
    std::array<float, 3> bbox_min = { 0, 0, 0 };
    std::array<float, 3> bbox_max = { 1, 1, 1 };
    ValidationResult r = val.validateCoordinates(points, bbox_min, bbox_max);
    REQUIRE(r.is_valid);
}

TEST_CASE("Validation: validateCoordinates outside bbox adds warning", "[correction][validation]") {
    Validation val;
    std::vector<std::array<float, 3>> points = { {2.0f, 2.0f, 2.0f} };
    std::array<float, 3> bbox_min = { 0, 0, 0 };
    std::array<float, 3> bbox_max = { 1, 1, 1 };
    ValidationResult r = val.validateCoordinates(points, bbox_min, bbox_max);
    REQUIRE(r.is_valid);
    REQUIRE(!r.warnings.empty());
}

TEST_CASE("Validation: checkConsistency null grid returns false", "[correction][validation]") {
    initOpenVDB();
    Validation val;
    FloatGridPtr grid = makeGrid(1.0f, 0, 0, 0, 1.0f);
    REQUIRE_FALSE(val.checkConsistency(nullptr, grid, 1e-6f));
    REQUIRE_FALSE(val.checkConsistency(grid, nullptr, 1e-6f));
}

TEST_CASE("Validation: checkConsistency same grid returns true", "[correction][validation]") {
    initOpenVDB();
    Validation val;
    FloatGridPtr grid = makeGrid(1.0f, 0, 0, 0, 1.0f);
    REQUIRE(val.checkConsistency(grid, grid, 1e-6f));
}

TEST_CASE("Validation: checkConsistency same transform returns true", "[correction][validation]") {
    initOpenVDB();
    Validation val;
    FloatGridPtr g1 = makeGrid(1.0f, 0, 0, 0, 1.0f);
    FloatGridPtr g2 = makeGrid(1.0f, 1, 0, 0, 2.0f);
    REQUIRE(val.checkConsistency(g1, g2, 1e-6f));
}

TEST_CASE("Validation: checkConsistency different voxel size returns false", "[correction][validation]") {
    initOpenVDB();
    Validation val;
    FloatGridPtr g1 = makeGrid(1.0f, 0, 0, 0, 1.0f);
    FloatGridPtr g2 = makeGrid(2.0f, 0, 0, 0, 1.0f);
    REQUIRE_FALSE(val.checkConsistency(g1, g2, 1e-6f));
}
