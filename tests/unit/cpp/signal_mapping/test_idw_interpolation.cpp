/**
 * C++ unit tests for IDWMapper (idw_interpolation.cpp).
 * Inverse Distance Weighting: power, k_neighbors, map.
 *
 * Aligned with src/am_qadf_native/signal_mapping/idw_interpolation.hpp
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "am_qadf_native/signal_mapping/idw_interpolation.hpp"
#include "am_qadf_native/signal_mapping/interpolation_base.hpp"
#include <openvdb/openvdb.h>
#include <openvdb/Grid.h>
#include <openvdb/math/Transform.h>
#include <vector>

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
// Constructor
// ---------------------------------------------------------------------------

TEST_CASE("IDWMapper: default constructor", "[signal_mapping][idw]") {
    IDWMapper mapper;
    std::vector<Point> points = { Point(0, 0, 0) };
    std::vector<float> values = { 1.0f };
    FloatGridPtr grid = makeTestGrid();
    REQUIRE_NOTHROW(mapper.map(grid, points, values));
}

TEST_CASE("IDWMapper: constructor with power and k_neighbors", "[signal_mapping][idw]") {
    IDWMapper mapper(3.0f, 5);
    REQUIRE_NOTHROW(mapper.map(makeTestGrid(), { Point(1, 1, 1) }, { 10.0f }));
}

TEST_CASE("IDWMapper: constructor invalid power throws", "[signal_mapping][idw]") {
    REQUIRE_THROWS_AS(IDWMapper(0.0f, 5), std::invalid_argument);
    REQUIRE_THROWS_AS(IDWMapper(-1.0f, 5), std::invalid_argument);
}

TEST_CASE("IDWMapper: constructor invalid k_neighbors throws", "[signal_mapping][idw]") {
    REQUIRE_THROWS_AS(IDWMapper(2.0f, 0), std::invalid_argument);
    REQUIRE_THROWS_AS(IDWMapper(2.0f, -1), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// setPower, setKNeighbors
// ---------------------------------------------------------------------------

TEST_CASE("IDWMapper: setPower valid", "[signal_mapping][idw]") {
    IDWMapper mapper(2.0f, 10);
    REQUIRE_NOTHROW(mapper.setPower(5.0f));
}

TEST_CASE("IDWMapper: setPower invalid throws", "[signal_mapping][idw]") {
    IDWMapper mapper(2.0f, 10);
    REQUIRE_THROWS_AS(mapper.setPower(0.0f), std::invalid_argument);
}

TEST_CASE("IDWMapper: setKNeighbors valid", "[signal_mapping][idw]") {
    IDWMapper mapper(2.0f, 10);
    REQUIRE_NOTHROW(mapper.setKNeighbors(20));
}

TEST_CASE("IDWMapper: setKNeighbors invalid throws", "[signal_mapping][idw]") {
    IDWMapper mapper(2.0f, 10);
    REQUIRE_THROWS_AS(mapper.setKNeighbors(0), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// map
// ---------------------------------------------------------------------------

TEST_CASE("IDWMapper: map single point", "[signal_mapping][idw]") {
    FloatGridPtr grid = makeTestGrid(1.0f);
    IDWMapper mapper(2.0f, 5);
    std::vector<Point> points = { Point(1.0f, 1.0f, 1.0f) };
    std::vector<float> values = { 50.0f };

    mapper.map(grid, points, values);

    float val = grid->tree().getValue(openvdb::Coord(1, 1, 1));
    REQUIRE(val > 0.0f);
    REQUIRE_THAT(val, Catch::Matchers::WithinAbs(50.0f, 1e-3f));
}

TEST_CASE("IDWMapper: map multiple points", "[signal_mapping][idw]") {
    FloatGridPtr grid = makeTestGrid(1.0f);
    IDWMapper mapper(2.0f, 10);
    std::vector<Point> points = {
        Point(0.0f, 0.0f, 0.0f),
        Point(2.0f, 2.0f, 2.0f),
        Point(4.0f, 4.0f, 4.0f)
    };
    std::vector<float> values = { 10.0f, 20.0f, 30.0f };

    mapper.map(grid, points, values);

    REQUIRE(grid->tree().getValue(openvdb::Coord(0, 0, 0)) > 0.0f);
    REQUIRE(grid->tree().getValue(openvdb::Coord(2, 2, 2)) > 0.0f);
}

TEST_CASE("IDWMapper: map empty points does not throw", "[signal_mapping][idw]") {
    FloatGridPtr grid = makeTestGrid();
    IDWMapper mapper(2.0f, 5);
    std::vector<Point> points;
    std::vector<float> values;
    REQUIRE_NOTHROW(mapper.map(grid, points, values));
}

TEST_CASE("IDWMapper: points and values size mismatch throws", "[signal_mapping][idw]") {
    FloatGridPtr grid = makeTestGrid();
    IDWMapper mapper(2.0f, 5);
    std::vector<Point> points = { Point(0, 0, 0), Point(1, 1, 1) };
    std::vector<float> values = { 1.0f };

    REQUIRE_THROWS_AS(mapper.map(grid, points, values), std::invalid_argument);
}
