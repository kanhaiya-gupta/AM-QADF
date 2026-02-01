/**
 * C++ unit tests for NearestNeighborMapper (nearest_neighbor.cpp).
 *
 * Aligned with src/am_qadf_native/signal_mapping/nearest_neighbor.hpp
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "am_qadf_native/signal_mapping/nearest_neighbor.hpp"
#include "am_qadf_native/signal_mapping/interpolation_base.hpp"
#include <openvdb/openvdb.h>
#include <openvdb/Grid.h>
#include <openvdb/math/Transform.h>
#include <openvdb/math/Coord.h>
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
// map
// ---------------------------------------------------------------------------

TEST_CASE("NearestNeighborMapper: map single point", "[signal_mapping][nearest_neighbor]") {
    FloatGridPtr grid = makeTestGrid(1.0f);
    NearestNeighborMapper mapper;
    std::vector<Point> points = { Point(0.5f, 0.5f, 0.5f) };  // cell-centered (0,0,0)
    std::vector<float> values = { 10.0f };

    mapper.map(grid, points, values);

    float val = grid->tree().getValue(openvdb::Coord(0, 0, 0));
    REQUIRE_THAT(val, Catch::Matchers::WithinAbs(10.0f, 1e-6f));
}

TEST_CASE("NearestNeighborMapper: map multiple points", "[signal_mapping][nearest_neighbor]") {
    FloatGridPtr grid = makeTestGrid(1.0f);
    NearestNeighborMapper mapper;
    std::vector<Point> points = {
        Point(1.0f, 1.0f, 1.0f),
        Point(2.0f, 2.0f, 2.0f),
        Point(3.0f, 3.0f, 3.0f)
    };
    std::vector<float> values = { 10.0f, 20.0f, 30.0f };

    mapper.map(grid, points, values);

    REQUIRE_THAT(grid->tree().getValue(openvdb::Coord(1, 1, 1)), Catch::Matchers::WithinAbs(10.0f, 1e-6f));
    REQUIRE_THAT(grid->tree().getValue(openvdb::Coord(2, 2, 2)), Catch::Matchers::WithinAbs(20.0f, 1e-6f));
    REQUIRE_THAT(grid->tree().getValue(openvdb::Coord(3, 3, 3)), Catch::Matchers::WithinAbs(30.0f, 1e-6f));
}

TEST_CASE("NearestNeighborMapper: map with voxel_size 0.5", "[signal_mapping][nearest_neighbor]") {
    FloatGridPtr grid = makeTestGrid(0.5f);
    NearestNeighborMapper mapper;
    // World (0.25, 0.25, 0.25) -> index (0.5, 0.5, 0.5) -> nearest (1, 1, 1) or (0,0,0) depending on round
    std::vector<Point> points = { Point(0.25f, 0.25f, 0.25f) };
    std::vector<float> values = { 42.0f };

    mapper.map(grid, points, values);

    float v0 = grid->tree().getValue(openvdb::Coord(0, 0, 0));
    float v1 = grid->tree().getValue(openvdb::Coord(1, 1, 1));
    REQUIRE((std::abs(v0 - 42.0f) < 1e-6f || std::abs(v1 - 42.0f) < 1e-6f));
}

TEST_CASE("NearestNeighborMapper: map empty points does not throw", "[signal_mapping][nearest_neighbor]") {
    FloatGridPtr grid = makeTestGrid();
    NearestNeighborMapper mapper;
    std::vector<Point> points;
    std::vector<float> values;
    REQUIRE_NOTHROW(mapper.map(grid, points, values));
}

TEST_CASE("NearestNeighborMapper: points and values size mismatch throws", "[signal_mapping][nearest_neighbor]") {
    FloatGridPtr grid = makeTestGrid();
    NearestNeighborMapper mapper;
    std::vector<Point> points = { Point(0, 0, 0), Point(1, 1, 1) };
    std::vector<float> values = { 1.0f };  // size 1

    REQUIRE_THROWS_AS(mapper.map(grid, points, values), std::invalid_argument);
}

TEST_CASE("NearestNeighborMapper: points and values size mismatch message", "[signal_mapping][nearest_neighbor]") {
    FloatGridPtr grid = makeTestGrid();
    NearestNeighborMapper mapper;
    std::vector<Point> points = { Point(0, 0, 0) };
    std::vector<float> values = { 1.0f, 2.0f };

    REQUIRE_THROWS_WITH(mapper.map(grid, points, values), Catch::Matchers::ContainsSubstring("size mismatch"));
}
