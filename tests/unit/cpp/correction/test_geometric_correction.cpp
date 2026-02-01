/**
 * C++ unit tests for GeometricCorrection (geometric_correction.cpp).
 * Tests correctDistortions, correctLensDistortion, correctSensorMisalignment.
 *
 * Aligned with docs/Tests/Test_New_Plans.md (tests/unit/cpp/correction/).
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "am_qadf_native/correction/geometric_correction.hpp"
#include <openvdb/openvdb.h>
#include <openvdb/Grid.h>
#include <openvdb/math/Transform.h>
#include <openvdb/math/Coord.h>
#include <vector>
#include <array>

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

TEST_CASE("GeometricCorrection: correctDistortions empty map returns same grid", "[correction][geometric_correction]") {
    initOpenVDB();
    GeometricCorrection corrector;
    FloatGridPtr grid = makeGrid(1.0f, 0, 0, 0, 5.0f);
    DistortionMap map;
    map.distortion_type = "lens";
    FloatGridPtr out = corrector.correctDistortions(grid, map);
    REQUIRE(out == grid);
}

TEST_CASE("GeometricCorrection: correctDistortions size mismatch returns same grid", "[correction][geometric_correction]") {
    initOpenVDB();
    GeometricCorrection corrector;
    FloatGridPtr grid = makeGrid(1.0f, 0, 0, 0, 5.0f);
    DistortionMap map;
    map.reference_points.push_back({0.0f, 0.0f, 0.0f});
    map.distortion_vectors.push_back({0.0f, 0.0f, 0.0f});
    map.distortion_vectors.push_back({1.0f, 0.0f, 0.0f});
    FloatGridPtr out = corrector.correctDistortions(grid, map);
    REQUIRE(out == grid);
}

TEST_CASE("GeometricCorrection: correctLensDistortion fewer than 5 coefficients returns same grid", "[correction][geometric_correction]") {
    initOpenVDB();
    GeometricCorrection corrector;
    FloatGridPtr grid = makeGrid(1.0f, 0, 0, 0, 5.0f);
    std::vector<float> coeffs = { 0.0f, 0.0f, 0.0f };
    FloatGridPtr out = corrector.correctLensDistortion(grid, coeffs);
    REQUIRE(out == grid);
}

TEST_CASE("GeometricCorrection: correctLensDistortion with 5 coefficients", "[correction][geometric_correction]") {
    initOpenVDB();
    GeometricCorrection corrector;
    FloatGridPtr grid = makeGrid(1.0f, 1, 1, 1, 10.0f);
    std::vector<float> coeffs = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    FloatGridPtr out = corrector.correctLensDistortion(grid, coeffs);
    REQUIRE(out != nullptr);
}

TEST_CASE("GeometricCorrection: correctSensorMisalignment identity", "[correction][geometric_correction]") {
    initOpenVDB();
    GeometricCorrection corrector;
    FloatGridPtr grid = makeGrid(1.0f, 0, 0, 0, 7.0f);
    std::array<float, 3> trans = { 0.0f, 0.0f, 0.0f };
    std::array<float, 3> rot = { 0.0f, 0.0f, 0.0f };
    FloatGridPtr out = corrector.correctSensorMisalignment(grid, trans, rot);
    REQUIRE(out != nullptr);
}
