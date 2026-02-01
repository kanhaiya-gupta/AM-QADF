/**
 * C++ unit tests for VoxelGridFactory (openvdb_grid.hpp).
 * Tests createUniform, createMultiResolution, createAdaptive.
 *
 * Aligned with src/am_qadf_native/voxelization/openvdb_grid.hpp (VoxelGridFactory)
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "am_qadf_native/voxelization/openvdb_grid.hpp"
#include <vector>
#include <string>

using namespace am_qadf_native::voxelization;

// ---------------------------------------------------------------------------
// createUniform
// ---------------------------------------------------------------------------

TEST_CASE("VoxelGridFactory: createUniform returns non-null", "[voxelization][grid_factory]") {
    auto grid = VoxelGridFactory::createUniform(0.5f, "test_signal");
    REQUIRE(grid != nullptr);
    REQUIRE_THAT(grid->getVoxelSize(), Catch::Matchers::WithinAbs(0.5f, 1e-6f));
}

TEST_CASE("VoxelGridFactory: createUniform sets signal name", "[voxelization][grid_factory]") {
    auto grid = VoxelGridFactory::createUniform(0.1f, "thermal");
    REQUIRE(grid != nullptr);
    REQUIRE(grid->getGrid() != nullptr);
    REQUIRE(grid->getGrid()->getName() == "thermal");
}

TEST_CASE("VoxelGridFactory: createUniform grid is usable", "[voxelization][grid_factory]") {
    auto grid = VoxelGridFactory::createUniform(1.0f, "power");
    REQUIRE(grid != nullptr);
    grid->addPointAtVoxel(0, 0, 0, 100.0f);
    REQUIRE_THAT(grid->getValue(0, 0, 0), Catch::Matchers::WithinAbs(100.0f, 1e-6f));
}

// ---------------------------------------------------------------------------
// createMultiResolution
// ---------------------------------------------------------------------------

TEST_CASE("VoxelGridFactory: createMultiResolution returns non-null", "[voxelization][grid_factory]") {
    std::vector<float> resolutions = { 1.0f, 0.5f, 0.25f };
    auto grid = VoxelGridFactory::createMultiResolution(resolutions);
    REQUIRE(grid != nullptr);
    REQUIRE(grid->getNumLevels() == 3);
}

TEST_CASE("VoxelGridFactory: createMultiResolution single level throws", "[voxelization][grid_factory]") {
    std::vector<float> resolutions = { 0.5f };
    REQUIRE_THROWS_AS(VoxelGridFactory::createMultiResolution(resolutions), std::invalid_argument);
}

TEST_CASE("VoxelGridFactory: createMultiResolution getGrid and getResolution", "[voxelization][grid_factory]") {
    std::vector<float> resolutions = { 1.0f, 0.5f };
    auto grid = VoxelGridFactory::createMultiResolution(resolutions);
    REQUIRE(grid != nullptr);
    auto g0 = grid->getGrid(0);
    REQUIRE(g0 != nullptr);
    float r0 = grid->getResolution(0);
    REQUIRE(r0 > 0.0f);
}

// ---------------------------------------------------------------------------
// createAdaptive
// ---------------------------------------------------------------------------

TEST_CASE("VoxelGridFactory: createAdaptive returns non-null", "[voxelization][grid_factory]") {
    auto grid = VoxelGridFactory::createAdaptive(0.2f);
    REQUIRE(grid != nullptr);
}

TEST_CASE("VoxelGridFactory: createAdaptive base resolution", "[voxelization][grid_factory]") {
    auto grid = VoxelGridFactory::createAdaptive(0.1f);
    REQUIRE(grid != nullptr);
    float r = grid->getResolutionForPoint(0.0f, 0.0f, 0.0f);
    REQUIRE(r > 0.0f);
}

TEST_CASE("VoxelGridFactory: createAdaptive addSpatialRegion and addPoint", "[voxelization][grid_factory]") {
    auto grid = VoxelGridFactory::createAdaptive(0.5f);
    REQUIRE(grid != nullptr);
    float bbox_min[3] = { 0.0f, 0.0f, 0.0f };
    float bbox_max[3] = { 10.0f, 10.0f, 10.0f };
    grid->addSpatialRegion(bbox_min, bbox_max, 0.25f);
    grid->addPoint(1.0f, 1.0f, 1.0f, 42.0f);
    auto grids = grid->getAllGrids();
    REQUIRE(!grids.empty());
}
