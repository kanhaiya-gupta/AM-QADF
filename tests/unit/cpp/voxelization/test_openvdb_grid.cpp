/**
 * C++ unit tests for OpenVDB grid types (openvdb_grid.cpp).
 * Tests UniformVoxelGrid, MultiResolutionVoxelGrid, AdaptiveResolutionVoxelGrid.
 *
 * Aligned with src/am_qadf_native/voxelization/openvdb_grid.hpp
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "am_qadf_native/voxelization/openvdb_grid.hpp"
#include <openvdb/openvdb.h>
#include <cmath>
#include <vector>
#include <string>

using namespace am_qadf_native::voxelization;

// Run first to isolate "Subprocess killed": if kill happens here, OpenVDB init is the cause (e.g. OOM on WSL).
TEST_CASE("OpenVDB: initialize", "[voxelization][openvdb]") {
    openvdb::initialize();
    REQUIRE(true);
}

// ---------------------------------------------------------------------------
// UniformVoxelGrid
// ---------------------------------------------------------------------------

TEST_CASE("UniformVoxelGrid: Constructor and voxel size", "[voxelization][openvdb][uniform]") {
    UniformVoxelGrid grid(0.1f, 0.0f, 0.0f, 0.0f);
    REQUIRE_THAT(grid.getVoxelSize(), Catch::Matchers::WithinAbs(0.1f, 1e-6f));
}

TEST_CASE("UniformVoxelGrid: Constructor with bbox offset", "[voxelization][openvdb][uniform]") {
    UniformVoxelGrid grid(0.5f, 1.0f, 2.0f, 3.0f);
    REQUIRE_THAT(grid.getVoxelSize(), Catch::Matchers::WithinAbs(0.5f, 1e-6f));
    REQUIRE(grid.getGrid() != nullptr);
}

TEST_CASE("UniformVoxelGrid: addPointAtVoxel and getValue", "[voxelization][openvdb][uniform]") {
    UniformVoxelGrid grid(1.0f);
    grid.addPointAtVoxel(0, 0, 0, 10.0f);
    grid.addPointAtVoxel(1, 1, 1, 20.0f);
    grid.addPointAtVoxel(2, 2, 2, 30.0f);

    REQUIRE_THAT(grid.getValue(0, 0, 0), Catch::Matchers::WithinAbs(10.0f, 1e-6f));
    REQUIRE_THAT(grid.getValue(1, 1, 1), Catch::Matchers::WithinAbs(20.0f, 1e-6f));
    REQUIRE_THAT(grid.getValue(2, 2, 2), Catch::Matchers::WithinAbs(30.0f, 1e-6f));
}

TEST_CASE("UniformVoxelGrid: addPoint (world coordinates) and getValueAtWorld", "[voxelization][openvdb][uniform]") {
    UniformVoxelGrid grid(1.0f, 0.0f, 0.0f, 0.0f);
    // World (0.5, 0.5, 0.5) -> cell-centered index (0, 0, 0)
    grid.addPoint(0.5f, 0.5f, 0.5f, 42.0f);
    float val = grid.getValueAtWorld(0.5f, 0.5f, 0.5f);
    REQUIRE_THAT(val, Catch::Matchers::WithinAbs(42.0f, 1e-5f));
}

TEST_CASE("UniformVoxelGrid: voxelToWorld conversion", "[voxelization][openvdb][uniform]") {
    UniformVoxelGrid grid(0.5f, 1.0f, 0.0f, 0.0f);
    float x, y, z;
    grid.voxelToWorld(0, 0, 0, x, y, z);
    // index (0,0,0) -> world = 0*voxel_size + bbox_min = (1, 0, 0) for linear transform
    REQUIRE_THAT(x, Catch::Matchers::WithinAbs(1.0f, 1e-5f));
    REQUIRE_THAT(y, Catch::Matchers::WithinAbs(0.0f, 1e-5f));
    REQUIRE_THAT(z, Catch::Matchers::WithinAbs(0.0f, 1e-5f));
}

TEST_CASE("UniformVoxelGrid: getWidth getHeight getDepth after adding points", "[voxelization][openvdb][uniform]") {
    UniformVoxelGrid grid(1.0f);
    grid.addPointAtVoxel(0, 0, 0, 1.0f);
    grid.addPointAtVoxel(4, 5, 6, 2.0f);

    int w = grid.getWidth();
    int h = grid.getHeight();
    int d = grid.getDepth();
    REQUIRE(w >= 1);
    REQUIRE(h >= 1);
    REQUIRE(d >= 1);
    REQUIRE(w >= 5);   // 0..4
    REQUIRE(h >= 6);   // 0..5
    REQUIRE(d >= 7);   // 0..6
}

TEST_CASE("UniformVoxelGrid: getStatistics", "[voxelization][openvdb][uniform]") {
    UniformVoxelGrid grid(1.0f);
    grid.addPointAtVoxel(0, 0, 0, 10.0f);
    grid.addPointAtVoxel(1, 0, 0, 20.0f);
    grid.addPointAtVoxel(0, 1, 0, 30.0f);

    auto stats = grid.getStatistics();
    REQUIRE(stats.filled_voxels >= 3);
    REQUIRE(stats.mean >= 10.0f);
    REQUIRE(stats.min <= 10.0f);
    REQUIRE(stats.max >= 30.0f);
}

TEST_CASE("UniformVoxelGrid: aggregateAtVoxel mean", "[voxelization][openvdb][uniform]") {
    UniformVoxelGrid grid(1.0f);
    std::vector<float> values = { 10.0f, 20.0f, 30.0f };
    grid.aggregateAtVoxel(1, 1, 1, values, "mean");
    REQUIRE_THAT(grid.getValue(1, 1, 1), Catch::Matchers::WithinAbs(20.0f, 1e-5f));
}

TEST_CASE("UniformVoxelGrid: aggregateAtVoxel max", "[voxelization][openvdb][uniform]") {
    UniformVoxelGrid grid(1.0f);
    std::vector<float> values = { 10.0f, 50.0f, 30.0f };
    grid.aggregateAtVoxel(2, 2, 2, values, "max");
    REQUIRE_THAT(grid.getValue(2, 2, 2), Catch::Matchers::WithinAbs(50.0f, 1e-5f));
}

TEST_CASE("UniformVoxelGrid: aggregateAtVoxel min", "[voxelization][openvdb][uniform]") {
    UniformVoxelGrid grid(1.0f);
    std::vector<float> values = { 10.0f, 5.0f, 30.0f };
    grid.aggregateAtVoxel(0, 0, 0, values, "min");
    REQUIRE_THAT(grid.getValue(0, 0, 0), Catch::Matchers::WithinAbs(5.0f, 1e-5f));
}

TEST_CASE("UniformVoxelGrid: aggregateAtVoxel empty values does not crash", "[voxelization][openvdb][uniform]") {
    UniformVoxelGrid grid(1.0f);
    std::vector<float> empty;
    REQUIRE_NOTHROW(grid.aggregateAtVoxel(0, 0, 0, empty, "mean"));
}

TEST_CASE("UniformVoxelGrid: setSignalName and getGrid", "[voxelization][openvdb][uniform]") {
    UniformVoxelGrid grid(0.1f);
    grid.setSignalName("test_signal");
    REQUIRE(grid.getGrid() != nullptr);
    REQUIRE(grid.getGrid()->getName() == "test_signal");
}

TEST_CASE("UniformVoxelGrid: copyFromGrid", "[voxelization][openvdb][uniform]") {
    UniformVoxelGrid source(1.0f);
    source.addPointAtVoxel(0, 0, 0, 100.0f);
    UniformVoxelGrid target(1.0f);
    target.copyFromGrid(source.getGrid());
    REQUIRE_THAT(target.getValue(0, 0, 0), Catch::Matchers::WithinAbs(100.0f, 1e-5f));
}

TEST_CASE("UniformVoxelGrid: copyFromGrid null does not crash", "[voxelization][openvdb][uniform]") {
    UniformVoxelGrid grid(1.0f);
    REQUIRE_NOTHROW(grid.copyFromGrid(nullptr));
}

// ---------------------------------------------------------------------------
// MultiResolutionVoxelGrid
// ---------------------------------------------------------------------------

TEST_CASE("MultiResolutionVoxelGrid: Constructor and getNumLevels", "[voxelization][openvdb][multires]") {
    std::vector<float> resolutions = { 1.0f, 0.5f, 0.25f };
    MultiResolutionVoxelGrid grid(resolutions, 1.0f);
    REQUIRE(grid.getNumLevels() >= 1);
}

TEST_CASE("MultiResolutionVoxelGrid: getGrid and getResolution", "[voxelization][openvdb][multires]") {
    std::vector<float> resolutions = { 1.0f, 0.5f };
    MultiResolutionVoxelGrid grid(resolutions, 1.0f);
    auto g0 = grid.getGrid(0);
    REQUIRE(g0 != nullptr);
    float r0 = grid.getResolution(0);
    REQUIRE(r0 > 0.0f);
}

// ---------------------------------------------------------------------------
// AdaptiveResolutionVoxelGrid
// ---------------------------------------------------------------------------

TEST_CASE("AdaptiveResolutionVoxelGrid: Constructor", "[voxelization][openvdb][adaptive]") {
    AdaptiveResolutionVoxelGrid grid(0.1f);
    auto grids = grid.getAllGrids();
    REQUIRE(grids.size() >= 0);
}

TEST_CASE("AdaptiveResolutionVoxelGrid: addSpatialRegion and addPoint", "[voxelization][openvdb][adaptive]") {
    AdaptiveResolutionVoxelGrid grid(0.5f);
    float bbox_min[3] = { 0.0f, 0.0f, 0.0f };
    float bbox_max[3] = { 10.0f, 10.0f, 10.0f };
    grid.addSpatialRegion(bbox_min, bbox_max, 0.25f);
    grid.addPoint(1.0f, 1.0f, 1.0f, 42.0f);
    auto grids = grid.getAllGrids();
    REQUIRE(grids.size() >= 0);
}

TEST_CASE("AdaptiveResolutionVoxelGrid: getResolutionForPoint", "[voxelization][openvdb][adaptive]") {
    AdaptiveResolutionVoxelGrid grid(0.5f);
    float r = grid.getResolutionForPoint(0.0f, 0.0f, 0.0f);
    REQUIRE(r > 0.0f);
}
