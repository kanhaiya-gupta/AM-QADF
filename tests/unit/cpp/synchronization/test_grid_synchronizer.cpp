/**
 * C++ unit tests for GridSynchronizer (grid_synchronizer.cpp).
 * Tests synchronize (spatial + temporal), synchronizeSpatial, synchronizeTemporal.
 *
 * Aligned with src/am_qadf_native/synchronization/grid_synchronizer.hpp
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "am_qadf_native/synchronization/grid_synchronizer.hpp"
#include <openvdb/openvdb.h>
#include <openvdb/Grid.h>
#include <openvdb/math/Transform.h>
#include <openvdb/math/Coord.h>
#include <vector>
#include <string>

using namespace am_qadf_native::synchronization;

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

// ---------------------------------------------------------------------------
// synchronizeSpatial
// ---------------------------------------------------------------------------

TEST_CASE("GridSynchronizer: synchronizeSpatial empty list returns empty", "[synchronization][grid_synchronizer]") {
    initOpenVDB();
    GridSynchronizer sync;
    std::vector<FloatGridPtr> grids;
    auto ref = makeGrid(1.0f, 0, 0, 0, 0.0f);
    auto result = sync.synchronizeSpatial(grids, ref, "trilinear");
    REQUIRE(result.empty());
}

TEST_CASE("GridSynchronizer: synchronizeSpatial single grid", "[synchronization][grid_synchronizer]") {
    initOpenVDB();
    GridSynchronizer sync;
    auto grid = makeGrid(1.0f, 0, 0, 0, 17.0f);
    auto ref = makeGrid(1.0f, 0, 0, 0, 0.0f);
    auto result = sync.synchronizeSpatial({ grid }, ref, "trilinear");
    REQUIRE(result.size() == 1);
    REQUIRE(result[0] != nullptr);
    REQUIRE_THAT(result[0]->tree().getValue(openvdb::Coord(0, 0, 0)),
        Catch::Matchers::WithinAbs(17.0f, 1e-5f));
}

TEST_CASE("GridSynchronizer: synchronizeSpatial two grids", "[synchronization][grid_synchronizer]") {
    initOpenVDB();
    GridSynchronizer sync;
    auto g1 = makeGrid(1.0f, 0, 0, 0, 1.0f);
    auto g2 = makeGrid(1.0f, 0, 0, 0, 2.0f);
    auto ref = makeGrid(1.0f, 0, 0, 0, 0.0f);
    auto result = sync.synchronizeSpatial({ g1, g2 }, ref, "nearest");
    REQUIRE(result.size() == 2);
    REQUIRE(result[0] != nullptr);
    REQUIRE(result[1] != nullptr);
}

TEST_CASE("GridSynchronizer: synchronizeSpatial method triquadratic", "[synchronization][grid_synchronizer]") {
    initOpenVDB();
    GridSynchronizer sync;
    auto grid = makeGrid(1.0f, 0, 0, 0, 5.0f);
    auto ref = makeGrid(1.0f, 0, 0, 0, 0.0f);
    auto result = sync.synchronizeSpatial({ grid }, ref, "triquadratic");
    REQUIRE(result.size() == 1);
    REQUIRE(result[0] != nullptr);
}

// ---------------------------------------------------------------------------
// synchronizeTemporal
// ---------------------------------------------------------------------------

TEST_CASE("GridSynchronizer: synchronizeTemporal empty grids", "[synchronization][grid_synchronizer]") {
    initOpenVDB();
    GridSynchronizer sync;
    std::vector<FloatGridPtr> grids;
    std::vector<std::vector<float>> timestamps;
    std::vector<std::vector<int>> layer_indices;
    auto result = sync.synchronizeTemporal(grids, timestamps, layer_indices, 0.1f, 1);
    REQUIRE(result.empty());
}

TEST_CASE("GridSynchronizer: synchronizeTemporal one grid with timestamps", "[synchronization][grid_synchronizer]") {
    initOpenVDB();
    GridSynchronizer sync;
    auto grid = makeGrid(1.0f, 0, 0, 0, 11.0f);
    std::vector<std::vector<float>> timestamps = { { 1.0f } };
    std::vector<std::vector<int>> layer_indices = { { 0 } };
    auto result = sync.synchronizeTemporal({ grid }, timestamps, layer_indices, 0.5f, 1);
    REQUIRE(result.size() >= 1);
    REQUIRE(result[0] != nullptr);
    REQUIRE_THAT(result[0]->tree().getValue(openvdb::Coord(0, 0, 0)),
        Catch::Matchers::WithinAbs(11.0f, 1e-5f));
}

TEST_CASE("GridSynchronizer: synchronizeTemporal timestamps size can be less than grids", "[synchronization][grid_synchronizer]") {
    initOpenVDB();
    GridSynchronizer sync;
    auto g1 = makeGrid(1.0f, 0, 0, 0, 1.0f);
    auto g2 = makeGrid(1.0f, 0, 0, 0, 2.0f);
    std::vector<std::vector<float>> timestamps = { { 0.0f } };  // only first grid has timestamps
    std::vector<std::vector<int>> layer_indices;
    auto result = sync.synchronizeTemporal({ g1, g2 }, timestamps, layer_indices, 0.5f, 1);
    REQUIRE(result.size() >= 1);
}

// ---------------------------------------------------------------------------
// synchronize (combined spatial + temporal)
// ---------------------------------------------------------------------------

TEST_CASE("GridSynchronizer: synchronize one source one timestamp", "[synchronization][grid_synchronizer]") {
    initOpenVDB();
    GridSynchronizer sync;
    auto source = makeGrid(1.0f, 0, 0, 0, 9.0f);
    auto ref = makeGrid(1.0f, 0, 0, 0, 0.0f);
    std::vector<std::vector<float>> timestamps = { { 0.5f } };
    std::vector<std::vector<int>> layer_indices = { { 0 } };
    auto result = sync.synchronize(
        { source }, ref, timestamps, layer_indices, 0.5f, 1
    );
    REQUIRE(result.size() >= 1);
    REQUIRE(result[0] != nullptr);
    REQUIRE_THAT(result[0]->tree().getValue(openvdb::Coord(0, 0, 0)),
        Catch::Matchers::WithinAbs(9.0f, 1e-5f));
}

TEST_CASE("GridSynchronizer: synchronize two sources", "[synchronization][grid_synchronizer]") {
    initOpenVDB();
    GridSynchronizer sync;
    auto s1 = makeGrid(1.0f, 0, 0, 0, 10.0f);
    auto s2 = makeGrid(1.0f, 0, 0, 0, 20.0f);
    auto ref = makeGrid(1.0f, 0, 0, 0, 0.0f);
    std::vector<std::vector<float>> timestamps = { { 0.0f }, { 0.0f } };
    std::vector<std::vector<int>> layer_indices = { { 0 }, { 0 } };
    auto result = sync.synchronize(
        { s1, s2 }, ref, timestamps, layer_indices, 1.0f, 1
    );
    REQUIRE(result.size() >= 1);
    REQUIRE(result[0] != nullptr);
}
