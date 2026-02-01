/**
 * C++ unit tests for LinearMapper (linear_interpolation.cpp).
 * Trilinear distribution to 8 neighboring voxels.
 *
 * Aligned with src/am_qadf_native/signal_mapping/linear_interpolation.hpp
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "am_qadf_native/signal_mapping/linear_interpolation.hpp"
#include "am_qadf_native/signal_mapping/interpolation_base.hpp"
#include <openvdb/openvdb.h>
#include <openvdb/Grid.h>
#include <openvdb/math/Transform.h>
#include <openvdb/math/Coord.h>
#include <vector>
#include <cmath>

using namespace am_qadf_native::signal_mapping;

namespace {

FloatGridPtr makeTestGrid(float voxel_size = 1.0f) {
    openvdb::initialize();
    auto grid = openvdb::FloatGrid::create(0.0f);
    grid->setTransform(openvdb::math::Transform::createLinearTransform(voxel_size));
    return grid;
}

} // namespace

// ---------------------------------------------------------------------------
// map
// ---------------------------------------------------------------------------

TEST_CASE("LinearMapper: map single point", "[signal_mapping][linear]") {
    FloatGridPtr grid = makeTestGrid(1.0f);
    LinearMapper mapper;
    std::vector<Point> points = { Point(0.5f, 0.5f, 0.5f) };
    std::vector<float> values = { 8.0f };

    mapper.map(grid, points, values);

    // Trilinear at (0.5, 0.5, 0.5) distributes to 8 corners; each gets 1.0 if weights equal
    float v000 = grid->tree().getValue(openvdb::Coord(0, 0, 0));
    float v111 = grid->tree().getValue(openvdb::Coord(1, 1, 1));
    REQUIRE(v000 > 0.0f);
    REQUIRE(v111 > 0.0f);
}

TEST_CASE("LinearMapper: map multiple points", "[signal_mapping][linear]") {
    FloatGridPtr grid = makeTestGrid(1.0f);
    LinearMapper mapper;
    std::vector<Point> points = {
        Point(1.0f, 1.0f, 1.0f),
        Point(2.5f, 2.5f, 2.5f)
    };
    std::vector<float> values = { 10.0f, 20.0f };

    mapper.map(grid, points, values);

    REQUIRE(grid->tree().getValue(openvdb::Coord(1, 1, 1)) > 0.0f);
    REQUIRE(grid->tree().getValue(openvdb::Coord(2, 2, 2)) > 0.0f);
}

TEST_CASE("LinearMapper: map empty points does not throw", "[signal_mapping][linear]") {
    FloatGridPtr grid = makeTestGrid();
    LinearMapper mapper;
    std::vector<Point> points;
    std::vector<float> values;
    REQUIRE_NOTHROW(mapper.map(grid, points, values));
}

TEST_CASE("LinearMapper: points and values size mismatch throws", "[signal_mapping][linear]") {
    FloatGridPtr grid = makeTestGrid();
    LinearMapper mapper;
    std::vector<Point> points = { Point(0, 0, 0) };
    std::vector<float> values = { 1.0f, 2.0f };

    REQUIRE_THROWS_AS(mapper.map(grid, points, values), std::invalid_argument);
}

TEST_CASE("LinearMapper: trilinear weights sum to value", "[signal_mapping][linear]") {
    FloatGridPtr grid = makeTestGrid(1.0f);
    LinearMapper mapper;
    std::vector<Point> points = { Point(1.5f, 1.5f, 1.5f) };
    std::vector<float> values = { 100.0f };

    mapper.map(grid, points, values);

    float sum = 0.0f;
    for (int i = 1; i <= 2; ++i)
        for (int j = 1; j <= 2; ++j)
            for (int k = 1; k <= 2; ++k)
                sum += grid->tree().getValue(openvdb::Coord(i, j, k));
    REQUIRE_THAT(sum, Catch::Matchers::WithinAbs(100.0f, 1e-3f));
}
