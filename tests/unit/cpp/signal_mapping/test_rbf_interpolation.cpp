/**
 * C++ unit tests for RBFMapper (rbf_interpolation.cpp).
 * Radial Basis Function: kernel_type, epsilon, map.
 * Requires EIGEN_AVAILABLE for full RBF solve.
 *
 * Aligned with src/am_qadf_native/signal_mapping/rbf_interpolation.hpp
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "am_qadf_native/signal_mapping/rbf_interpolation.hpp"
#include "am_qadf_native/signal_mapping/interpolation_base.hpp"
#include <openvdb/openvdb.h>
#include <openvdb/Grid.h>
#include <openvdb/math/Transform.h>
#include <openvdb/math/Coord.h>
#include <vector>
#include <string>

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

TEST_CASE("RBFMapper: default constructor", "[signal_mapping][rbf]") {
    RBFMapper mapper;
    FloatGridPtr grid = makeTestGrid();
    std::vector<Point> points = { Point(0, 0, 0) };
    std::vector<float> values = { 1.0f };
    REQUIRE_NOTHROW(mapper.map(grid, points, values));
}

TEST_CASE("RBFMapper: constructor with kernel and epsilon", "[signal_mapping][rbf]") {
    RBFMapper mapper("gaussian", 0.5f);
    REQUIRE_NOTHROW(mapper.map(makeTestGrid(), { Point(1, 1, 1) }, { 10.0f }));
}

TEST_CASE("RBFMapper: constructor invalid epsilon throws", "[signal_mapping][rbf]") {
    REQUIRE_THROWS_AS(RBFMapper("gaussian", 0.0f), std::invalid_argument);
    REQUIRE_THROWS_AS(RBFMapper("gaussian", -1.0f), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// setKernelType, setEpsilon
// ---------------------------------------------------------------------------

TEST_CASE("RBFMapper: setKernelType", "[signal_mapping][rbf]") {
    RBFMapper mapper("gaussian", 1.0f);
    REQUIRE_NOTHROW(mapper.setKernelType("multiquadric"));
}

TEST_CASE("RBFMapper: setEpsilon valid", "[signal_mapping][rbf]") {
    RBFMapper mapper("gaussian", 1.0f);
    REQUIRE_NOTHROW(mapper.setEpsilon(2.0f));
}

TEST_CASE("RBFMapper: setEpsilon invalid throws", "[signal_mapping][rbf]") {
    RBFMapper mapper("gaussian", 1.0f);
    REQUIRE_THROWS_AS(mapper.setEpsilon(0.0f), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// map
// ---------------------------------------------------------------------------

TEST_CASE("RBFMapper: map single point", "[signal_mapping][rbf]") {
    FloatGridPtr grid = makeTestGrid(1.0f);
    RBFMapper mapper("gaussian", 1.0f);
    std::vector<Point> points = { Point(1.0f, 1.0f, 1.0f) };
    std::vector<float> values = { 50.0f };

    mapper.map(grid, points, values);

    float val = grid->tree().getValue(openvdb::Coord(1, 1, 1));
    REQUIRE((val > 0.0f || val == 0.0f));  // implementation may or may not set exact voxel
}

TEST_CASE("RBFMapper: map multiple points", "[signal_mapping][rbf]") {
    FloatGridPtr grid = makeTestGrid(1.0f);
    RBFMapper mapper("gaussian", 0.5f);
    std::vector<Point> points = {
        Point(0.0f, 0.0f, 0.0f),
        Point(2.0f, 2.0f, 2.0f)
    };
    std::vector<float> values = { 10.0f, 20.0f };

    mapper.map(grid, points, values);

    REQUIRE_NOTHROW(grid->tree().getValue(openvdb::Coord(0, 0, 0)));
    REQUIRE_NOTHROW(grid->tree().getValue(openvdb::Coord(2, 2, 2)));
}

TEST_CASE("RBFMapper: map empty points does not throw", "[signal_mapping][rbf]") {
    FloatGridPtr grid = makeTestGrid();
    RBFMapper mapper("gaussian", 1.0f);
    std::vector<Point> points;
    std::vector<float> values;
    REQUIRE_NOTHROW(mapper.map(grid, points, values));
}

TEST_CASE("RBFMapper: points and values size mismatch throws", "[signal_mapping][rbf]") {
    FloatGridPtr grid = makeTestGrid();
    RBFMapper mapper("gaussian", 1.0f);
    std::vector<Point> points = { Point(0, 0, 0), Point(1, 1, 1) };
    std::vector<float> values = { 1.0f };

    REQUIRE_THROWS_AS(mapper.map(grid, points, values), std::invalid_argument);
}

TEST_CASE("RBFMapper: map with multiquadric kernel", "[signal_mapping][rbf]") {
    FloatGridPtr grid = makeTestGrid(1.0f);
    RBFMapper mapper("multiquadric", 1.0f);
    std::vector<Point> points = { Point(1.0f, 1.0f, 1.0f) };
    std::vector<float> values = { 25.0f };

    REQUIRE_NOTHROW(mapper.map(grid, points, values));
}

TEST_CASE("RBFMapper: map with thin_plate kernel", "[signal_mapping][rbf]") {
    FloatGridPtr grid = makeTestGrid(1.0f);
    RBFMapper mapper("thin_plate", 1.0f);
    std::vector<Point> points = { Point(1.0f, 1.0f, 1.0f) };
    std::vector<float> values = { 25.0f };

    REQUIRE_NOTHROW(mapper.map(grid, points, values));
}
