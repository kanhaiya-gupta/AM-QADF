/**
 * C++ unit tests for InterpolationBase and helpers (interpolation_base.cpp).
 * Tests Point, numpy_to_points, and base behavior via concrete mappers.
 *
 * Aligned with src/am_qadf_native/signal_mapping/interpolation_base.hpp
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "am_qadf_native/signal_mapping/interpolation_base.hpp"
#include "am_qadf_native/signal_mapping/nearest_neighbor.hpp"
#include <openvdb/openvdb.h>
#include <openvdb/Grid.h>
#include <openvdb/math/Transform.h>
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
// Point
// ---------------------------------------------------------------------------

TEST_CASE("Point: default constructor", "[signal_mapping][interpolation_base]") {
    Point p;
    REQUIRE(p.x == 0.0f);
    REQUIRE(p.y == 0.0f);
    REQUIRE(p.z == 0.0f);
}

TEST_CASE("Point: parameterized constructor", "[signal_mapping][interpolation_base]") {
    Point p(1.0f, 2.0f, 3.0f);
    REQUIRE(p.x == 1.0f);
    REQUIRE(p.y == 2.0f);
    REQUIRE(p.z == 3.0f);
}

// ---------------------------------------------------------------------------
// numpy_to_points
// ---------------------------------------------------------------------------

TEST_CASE("numpy_to_points: empty data", "[signal_mapping][interpolation_base]") {
    std::vector<Point> points = numpy_to_points(nullptr, 0);
    REQUIRE(points.empty());
}

TEST_CASE("numpy_to_points: single point without bbox", "[signal_mapping][interpolation_base]") {
    float data[3] = { 1.0f, 2.0f, 3.0f };
    std::vector<Point> points = numpy_to_points(data, 1);
    REQUIRE(points.size() == 1);
    REQUIRE_THAT(points[0].x, Catch::Matchers::WithinAbs(1.0f, 1e-6f));
    REQUIRE_THAT(points[0].y, Catch::Matchers::WithinAbs(2.0f, 1e-6f));
    REQUIRE_THAT(points[0].z, Catch::Matchers::WithinAbs(3.0f, 1e-6f));
}

TEST_CASE("numpy_to_points: multiple points without bbox", "[signal_mapping][interpolation_base]") {
    float data[] = {
        0.0f, 0.0f, 0.0f,
        1.0f, 1.0f, 1.0f,
        2.0f, 2.0f, 2.0f
    };
    std::vector<Point> points = numpy_to_points(data, 3);
    REQUIRE(points.size() == 3);
    REQUIRE_THAT(points[1].x, Catch::Matchers::WithinAbs(1.0f, 1e-6f));
    REQUIRE_THAT(points[2].z, Catch::Matchers::WithinAbs(2.0f, 1e-6f));
}

TEST_CASE("numpy_to_points: with bbox_min offset", "[signal_mapping][interpolation_base]") {
    float data[3] = { 5.0f, 6.0f, 7.0f };
    float bbox_min[3] = { 1.0f, 2.0f, 3.0f };
    std::vector<Point> points = numpy_to_points(data, 1, bbox_min);
    REQUIRE(points.size() == 1);
    REQUIRE_THAT(points[0].x, Catch::Matchers::WithinAbs(4.0f, 1e-6f));
    REQUIRE_THAT(points[0].y, Catch::Matchers::WithinAbs(4.0f, 1e-6f));
    REQUIRE_THAT(points[0].z, Catch::Matchers::WithinAbs(4.0f, 1e-6f));
}

// ---------------------------------------------------------------------------
// InterpolationBase (via NearestNeighborMapper) – map contract
// ---------------------------------------------------------------------------

TEST_CASE("InterpolationBase: map with empty points and values", "[signal_mapping][interpolation_base]") {
    FloatGridPtr grid = makeTestGrid();
    NearestNeighborMapper mapper;
    std::vector<Point> points;
    std::vector<float> values;
    REQUIRE_NOTHROW(mapper.map(grid, points, values));
}
