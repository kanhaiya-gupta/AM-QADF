/**
 * C++ unit tests for UnifiedGridFactory (unified_grid_factory.cpp).
 * Requires EIGEN_AVAILABLE (BoundingBox and CoordinateSystem use Eigen).
 *
 * Aligned with src/am_qadf_native/voxelization/unified_grid_factory.hpp
 */

#ifdef EIGEN_AVAILABLE

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "am_qadf_native/voxelization/unified_grid_factory.hpp"
#include "am_qadf_native/synchronization/point_bounds.hpp"
#include "am_qadf_native/synchronization/point_coordinate_transform.hpp"
#include <openvdb/openvdb.h>
#include <cmath>

using namespace am_qadf_native::voxelization;
using namespace am_qadf_native::synchronization;

// ---------------------------------------------------------------------------
// createGrid
// ---------------------------------------------------------------------------

TEST_CASE("UnifiedGridFactory: createGrid returns non-null", "[voxelization][unified_grid_factory]") {
    BoundingBox bounds(0.0, 0.0, 0.0, 10.0, 10.0, 10.0);
    CoordinateSystem ref_system;
    ref_system.origin = Eigen::Vector3d::Zero();
    ref_system.scale = Eigen::Vector3d::Ones(3);

    UnifiedGridFactory factory;
    FloatGridPtr grid = factory.createGrid(bounds, ref_system, 0.5f);
    REQUIRE(grid != nullptr);
}

TEST_CASE("UnifiedGridFactory: createGrid with background value", "[voxelization][unified_grid_factory]") {
    BoundingBox bounds(0.0, 0.0, 0.0, 5.0, 5.0, 5.0);
    CoordinateSystem ref_system;

    UnifiedGridFactory factory;
    FloatGridPtr grid = factory.createGrid(bounds, ref_system, 0.25f, -1.0f);
    REQUIRE(grid != nullptr);
    REQUIRE_THAT(grid->background(), Catch::Matchers::WithinAbs(-1.0f, 1e-6f));
}

TEST_CASE("UnifiedGridFactory: createGrid invalid bounds throws", "[voxelization][unified_grid_factory]") {
    BoundingBox invalid_bounds(10.0, 10.0, 10.0, 0.0, 0.0, 0.0);  // min > max
    CoordinateSystem ref_system;
    UnifiedGridFactory factory;

    REQUIRE_THROWS_AS(
        factory.createGrid(invalid_bounds, ref_system, 0.5f),
        std::invalid_argument
    );
}

TEST_CASE("UnifiedGridFactory: createGrid zero voxel size throws", "[voxelization][unified_grid_factory]") {
    BoundingBox bounds(0.0, 0.0, 0.0, 10.0, 10.0, 10.0);
    CoordinateSystem ref_system;
    UnifiedGridFactory factory;

    REQUIRE_THROWS_AS(
        factory.createGrid(bounds, ref_system, 0.0f),
        std::invalid_argument
    );
}

TEST_CASE("UnifiedGridFactory: createGrid negative voxel size throws", "[voxelization][unified_grid_factory]") {
    BoundingBox bounds(0.0, 0.0, 0.0, 10.0, 10.0, 10.0);
    CoordinateSystem ref_system;
    UnifiedGridFactory factory;

    REQUIRE_THROWS_AS(
        factory.createGrid(bounds, ref_system, -0.1f),
        std::invalid_argument
    );
}

// ---------------------------------------------------------------------------
// createGrids
// ---------------------------------------------------------------------------

TEST_CASE("UnifiedGridFactory: createGrids returns correct count", "[voxelization][unified_grid_factory]") {
    BoundingBox bounds(0.0, 0.0, 0.0, 10.0, 10.0, 10.0);
    CoordinateSystem ref_system;
    UnifiedGridFactory factory;

    std::vector<FloatGridPtr> grids = factory.createGrids(bounds, ref_system, 0.5f, 4);
    REQUIRE(grids.size() == 4);
    for (const auto& g : grids) {
        REQUIRE(g != nullptr);
    }
}

TEST_CASE("UnifiedGridFactory: createGrids zero count returns empty", "[voxelization][unified_grid_factory]") {
    BoundingBox bounds(0.0, 0.0, 0.0, 10.0, 10.0, 10.0);
    CoordinateSystem ref_system;
    UnifiedGridFactory factory;

    std::vector<FloatGridPtr> grids = factory.createGrids(bounds, ref_system, 0.5f, 0);
    REQUIRE(grids.empty());
}

#endif // EIGEN_AVAILABLE
