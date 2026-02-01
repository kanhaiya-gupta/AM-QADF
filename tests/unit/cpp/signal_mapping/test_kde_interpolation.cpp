/**
 * C++ unit tests for KDEMapper (kde_interpolation.cpp).
 * Kernel Density Estimation: bandwidth, kernel_type, map.
 *
 * Aligned with src/am_qadf_native/signal_mapping/kde_interpolation.hpp
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "am_qadf_native/signal_mapping/kde_interpolation.hpp"
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

// Pre-activate voxels so KDE has a bbox to iterate (KDE uses evalActiveVoxelBoundingBox)
FloatGridPtr makeTestGridWithActiveVoxels(float voxel_size = 1.0f) {
    openvdb::initialize();
    auto grid = openvdb::FloatGrid::create(0.0f);
    grid->setTransform(openvdb::math::Transform::createLinearTransform(voxel_size));
    grid->tree().setValue(openvdb::Coord(0, 0, 0), 0.0f);
    grid->tree().setValue(openvdb::Coord(2, 2, 2), 0.0f);
    return grid;
}

} // namespace

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

TEST_CASE("KDEMapper: default constructor", "[signal_mapping][kde]") {
    KDEMapper mapper;
    FloatGridPtr grid = makeTestGridWithActiveVoxels();
    std::vector<Point> points = { Point(1, 1, 1) };
    std::vector<float> values = { 1.0f };
    REQUIRE_NOTHROW(mapper.map(grid, points, values));
}

TEST_CASE("KDEMapper: constructor with bandwidth and kernel", "[signal_mapping][kde]") {
    KDEMapper mapper(0.5f, "gaussian");
    FloatGridPtr grid = makeTestGridWithActiveVoxels();
    REQUIRE_NOTHROW(mapper.map(grid, { Point(1, 1, 1) }, { 10.0f }));
}

// ---------------------------------------------------------------------------
// setBandwidth, setKernelType
// ---------------------------------------------------------------------------

TEST_CASE("KDEMapper: setBandwidth", "[signal_mapping][kde]") {
    KDEMapper mapper(1.0f);
    REQUIRE_NOTHROW(mapper.setBandwidth(2.0f));
}

TEST_CASE("KDEMapper: setKernelType", "[signal_mapping][kde]") {
    KDEMapper mapper(1.0f, "gaussian");
    REQUIRE_NOTHROW(mapper.setKernelType("epanechnikov"));
}

// ---------------------------------------------------------------------------
// map
// ---------------------------------------------------------------------------

TEST_CASE("KDEMapper: map with active voxels", "[signal_mapping][kde]") {
    FloatGridPtr grid = makeTestGridWithActiveVoxels(1.0f);
    KDEMapper mapper(1.0f, "gaussian");
    std::vector<Point> points = { Point(1.0f, 1.0f, 1.0f) };
    std::vector<float> values = { 100.0f };

    mapper.map(grid, points, values);

    float v = grid->tree().getValue(openvdb::Coord(1, 1, 1));
    REQUIRE(v >= 0.0f);
}

TEST_CASE("KDEMapper: map multiple points", "[signal_mapping][kde]") {
    FloatGridPtr grid = makeTestGridWithActiveVoxels(1.0f);
    KDEMapper mapper(1.5f);
    std::vector<Point> points = {
        Point(0.0f, 0.0f, 0.0f),
        Point(2.0f, 2.0f, 2.0f)
    };
    std::vector<float> values = { 10.0f, 20.0f };

    mapper.map(grid, points, values);

    REQUIRE(grid->tree().getValue(openvdb::Coord(0, 0, 0)) >= 0.0f);
    REQUIRE(grid->tree().getValue(openvdb::Coord(2, 2, 2)) >= 0.0f);
}

TEST_CASE("KDEMapper: map empty points does not throw", "[signal_mapping][kde]") {
    FloatGridPtr grid = makeTestGridWithActiveVoxels();
    KDEMapper mapper(1.0f);
    std::vector<Point> points;
    std::vector<float> values;
    REQUIRE_NOTHROW(mapper.map(grid, points, values));
}

TEST_CASE("KDEMapper: points and values size mismatch throws", "[signal_mapping][kde]") {
    FloatGridPtr grid = makeTestGridWithActiveVoxels();
    KDEMapper mapper(1.0f);
    std::vector<Point> points = { Point(0, 0, 0) };
    std::vector<float> values = { 1.0f, 2.0f };

    REQUIRE_THROWS_AS(mapper.map(grid, points, values), std::invalid_argument);
}

TEST_CASE("KDEMapper: map with epanechnikov kernel", "[signal_mapping][kde]") {
    FloatGridPtr grid = makeTestGridWithActiveVoxels(1.0f);
    KDEMapper mapper(1.0f, "epanechnikov");
    std::vector<Point> points = { Point(1.0f, 1.0f, 1.0f) };
    std::vector<float> values = { 50.0f };

    mapper.map(grid, points, values);

    float v = grid->tree().getValue(openvdb::Coord(1, 1, 1));
    REQUIRE(v >= 0.0f);
}
