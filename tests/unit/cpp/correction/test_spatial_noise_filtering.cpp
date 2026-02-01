/**
 * C++ unit tests for SpatialNoiseFilter (spatial_noise_filtering.cpp).
 * Tests apply, applyMedianFilter, applyBilateralFilter, applyGaussianFilter.
 *
 * Aligned with docs/Tests/Test_New_Plans.md (tests/unit/cpp/correction/).
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "am_qadf_native/correction/spatial_noise_filtering.hpp"
#include <openvdb/openvdb.h>
#include <openvdb/Grid.h>
#include <openvdb/math/Transform.h>
#include <openvdb/math/Coord.h>

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

FloatGridPtr makeGrid3x3(float voxel_size, float center_val) {
    auto grid = openvdb::FloatGrid::create(0.0f);
    grid->setTransform(openvdb::math::Transform::createLinearTransform(voxel_size));
    for (int x = 0; x < 3; ++x)
        for (int y = 0; y < 3; ++y)
            for (int z = 0; z < 3; ++z)
                grid->tree().setValue(openvdb::Coord(x, y, z), center_val);
    return grid;
}

} // namespace

TEST_CASE("SpatialNoiseFilter: apply unknown method returns same grid", "[correction][spatial_noise_filtering]") {
    initOpenVDB();
    SpatialNoiseFilter filter;
    FloatGridPtr grid = makeGrid(1.0f, 0, 0, 0, 5.0f);
    FloatGridPtr out = filter.apply(grid, "unknown", 3, 1.0f, 0.1f);
    REQUIRE(out == grid);
}

TEST_CASE("SpatialNoiseFilter: apply median", "[correction][spatial_noise_filtering]") {
    initOpenVDB();
    SpatialNoiseFilter filter;
    FloatGridPtr grid = makeGrid3x3(1.0f, 10.0f);
    FloatGridPtr out = filter.applyMedianFilter(grid, 3);
    REQUIRE(out != nullptr);
    REQUIRE(out->tree().activeVoxelCount() == grid->tree().activeVoxelCount());
}

TEST_CASE("SpatialNoiseFilter: applyMedianFilter single voxel", "[correction][spatial_noise_filtering]") {
    initOpenVDB();
    SpatialNoiseFilter filter;
    FloatGridPtr grid = makeGrid(1.0f, 0, 0, 0, 7.0f);
    FloatGridPtr out = filter.applyMedianFilter(grid, 3);
    REQUIRE(out != nullptr);
}

TEST_CASE("SpatialNoiseFilter: applyBilateralFilter", "[correction][spatial_noise_filtering]") {
    initOpenVDB();
    SpatialNoiseFilter filter;
    FloatGridPtr grid = makeGrid3x3(1.0f, 5.0f);
    FloatGridPtr out = filter.applyBilateralFilter(grid, 1.0f, 0.1f);
    REQUIRE(out != nullptr);
}

TEST_CASE("SpatialNoiseFilter: applyGaussianFilter", "[correction][spatial_noise_filtering]") {
    initOpenVDB();
    SpatialNoiseFilter filter;
    FloatGridPtr grid = makeGrid3x3(1.0f, 5.0f);
    FloatGridPtr out = filter.applyGaussianFilter(grid, 1.0f);
    REQUIRE(out != nullptr);
}

TEST_CASE("SpatialNoiseFilter: apply gaussian method", "[correction][spatial_noise_filtering]") {
    initOpenVDB();
    SpatialNoiseFilter filter;
    FloatGridPtr grid = makeGrid(1.0f, 0, 0, 0, 3.0f);
    FloatGridPtr out = filter.apply(grid, "gaussian", 3, 1.0f, 0.1f);
    REQUIRE(out != nullptr);
}
