/**
 * C++ unit tests for SpatialAlignment (grid_spatial_alignment.cpp).
 * Tests align, alignWithTransform, transformsMatch, getTransformMatrix,
 * getWorldBoundingBox, alignToBoundingBox.
 *
 * Aligned with src/am_qadf_native/synchronization/grid_spatial_alignment.hpp
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "am_qadf_native/synchronization/grid_spatial_alignment.hpp"
#include <openvdb/openvdb.h>
#include <openvdb/Grid.h>
#include <openvdb/math/Transform.h>
#include <openvdb/math/Coord.h>
#include <vector>
#include <string>
#include <cmath>

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

FloatGridPtr makeGridWithTransform(float voxel_size, double tx, double ty, double tz,
    int ix, int iy, int iz, float value) {
    auto grid = openvdb::FloatGrid::create(0.0f);
    auto xform = openvdb::math::Transform::createLinearTransform(voxel_size);
    xform->postTranslate(openvdb::Vec3d(tx, ty, tz));
    grid->setTransform(openvdb::math::Transform::Ptr(new openvdb::math::Transform(*xform)));
    grid->tree().setValue(openvdb::Coord(ix, iy, iz), value);
    return grid;
}

} // namespace

// ---------------------------------------------------------------------------
// align
// ---------------------------------------------------------------------------

TEST_CASE("SpatialAlignment: align same transform preserves value", "[synchronization][grid_spatial_alignment]") {
    initOpenVDB();
    SpatialAlignment aligner;
    auto source = makeGrid(1.0f, 0, 0, 0, 42.0f);
    auto target = makeGrid(1.0f, 0, 0, 0, 0.0f);  // same voxel size, same transform
    FloatGridPtr result = aligner.align(source, target, "trilinear");
    REQUIRE(result != nullptr);
    REQUIRE_THAT(result->tree().getValue(openvdb::Coord(0, 0, 0)),
        Catch::Matchers::WithinAbs(42.0f, 1e-5f));
}

TEST_CASE("SpatialAlignment: align with trilinear", "[synchronization][grid_spatial_alignment]") {
    initOpenVDB();
    SpatialAlignment aligner;
    auto source = makeGrid(1.0f, 0, 0, 0, 10.0f);
    auto target = makeGrid(1.0f, 0, 0, 0, 0.0f);
    FloatGridPtr result = aligner.align(source, target, "trilinear");
    REQUIRE(result != nullptr);
    REQUIRE(result->transform().voxelSize()[0] == Catch::Approx(1.0));
}

TEST_CASE("SpatialAlignment: align with nearest", "[synchronization][grid_spatial_alignment]") {
    initOpenVDB();
    SpatialAlignment aligner;
    auto source = makeGrid(1.0f, 1, 1, 1, 7.0f);
    auto target = makeGrid(1.0f, 0, 0, 0, 0.0f);
    FloatGridPtr result = aligner.align(source, target, "nearest");
    REQUIRE(result != nullptr);
}

TEST_CASE("SpatialAlignment: align with triquadratic", "[synchronization][grid_spatial_alignment]") {
    initOpenVDB();
    SpatialAlignment aligner;
    auto source = makeGrid(1.0f, 0, 0, 0, 5.0f);
    auto target = makeGrid(1.0f, 0, 0, 0, 0.0f);
    FloatGridPtr result = aligner.align(source, target, "triquadratic");
    REQUIRE(result != nullptr);
}

// ---------------------------------------------------------------------------
// alignWithTransform
// ---------------------------------------------------------------------------

TEST_CASE("SpatialAlignment: alignWithTransform produces grid with target transform", "[synchronization][grid_spatial_alignment]") {
    initOpenVDB();
    SpatialAlignment aligner;
    auto source = makeGrid(1.0f, 0, 0, 0, 3.0f);
    auto target = makeGrid(1.0f, 0, 0, 0, 0.0f);
    openvdb::Mat4R identity;
    identity.setIdentity();
    FloatGridPtr result = aligner.alignWithTransform(source, target, identity, "trilinear");
    REQUIRE(result != nullptr);
    REQUIRE(aligner.transformsMatch(result, target, 1e-6));
}

// ---------------------------------------------------------------------------
// transformsMatch
// ---------------------------------------------------------------------------

TEST_CASE("SpatialAlignment: transformsMatch same transform returns true", "[synchronization][grid_spatial_alignment]") {
    initOpenVDB();
    SpatialAlignment aligner;
    auto g1 = makeGrid(1.0f, 0, 0, 0, 1.0f);
    auto g2 = makeGrid(1.0f, 0, 0, 0, 2.0f);
    REQUIRE(aligner.transformsMatch(g1, g2, 1e-6));
}

TEST_CASE("SpatialAlignment: transformsMatch different voxel size returns false", "[synchronization][grid_spatial_alignment]") {
    initOpenVDB();
    SpatialAlignment aligner;
    auto g1 = makeGrid(1.0f, 0, 0, 0, 1.0f);
    auto g2 = makeGrid(2.0f, 0, 0, 0, 1.0f);
    REQUIRE_FALSE(aligner.transformsMatch(g1, g2, 1e-6));
}

TEST_CASE("SpatialAlignment: transformsMatch different translation returns false", "[synchronization][grid_spatial_alignment]") {
    initOpenVDB();
    SpatialAlignment aligner;
    auto g1 = makeGridWithTransform(1.0f, 0, 0, 0, 0, 0, 0, 1.0f);
    auto g2 = makeGridWithTransform(1.0f, 1, 0, 0, 0, 0, 0, 1.0f);
    REQUIRE_FALSE(aligner.transformsMatch(g1, g2, 1e-6));
}

// ---------------------------------------------------------------------------
// getTransformMatrix
// ---------------------------------------------------------------------------

TEST_CASE("SpatialAlignment: getTransformMatrix returns 4x4", "[synchronization][grid_spatial_alignment]") {
    initOpenVDB();
    SpatialAlignment aligner;
    auto grid = makeGrid(1.0f, 0, 0, 0, 1.0f);
    auto mat = aligner.getTransformMatrix(grid);
    REQUIRE(mat.size() == 4);
    for (const auto& row : mat) {
        REQUIRE(row.size() == 4);
    }
    REQUIRE_THAT(mat[0][0], Catch::Matchers::WithinAbs(1.0, 1e-6));
    REQUIRE_THAT(mat[3][3], Catch::Matchers::WithinAbs(1.0, 1e-6));
}

// ---------------------------------------------------------------------------
// getWorldBoundingBox
// ---------------------------------------------------------------------------

TEST_CASE("SpatialAlignment: getWorldBoundingBox empty grid returns zeros", "[synchronization][grid_spatial_alignment]") {
    initOpenVDB();
    SpatialAlignment aligner;
    auto grid = openvdb::FloatGrid::create(0.0f);
    grid->setTransform(openvdb::math::Transform::createLinearTransform(1.0f));
    // No active voxels
    auto bbox = aligner.getWorldBoundingBox(grid);
    REQUIRE(bbox.size() == 6);
    REQUIRE(bbox[0] == 0.0);
    REQUIRE(bbox[1] == 0.0);
    REQUIRE(bbox[2] == 0.0);
    REQUIRE(bbox[3] == 0.0);
    REQUIRE(bbox[4] == 0.0);
    REQUIRE(bbox[5] == 0.0);
}

TEST_CASE("SpatialAlignment: getWorldBoundingBox single voxel", "[synchronization][grid_spatial_alignment]") {
    initOpenVDB();
    SpatialAlignment aligner;
    auto grid = makeGrid(1.0f, 2, 3, 4, 1.0f);
    auto bbox = aligner.getWorldBoundingBox(grid);
    REQUIRE(bbox.size() == 6);
    // Index (2,3,4) with voxel size 1 and default transform (no translation) -> world (2,3,4) to (3,4,5)
    REQUIRE(bbox[0] <= bbox[3]);
    REQUIRE(bbox[1] <= bbox[4]);
    REQUIRE(bbox[2] <= bbox[5]);
}

// ---------------------------------------------------------------------------
// alignToBoundingBox
// ---------------------------------------------------------------------------

TEST_CASE("SpatialAlignment: alignToBoundingBox resamples into unified box", "[synchronization][grid_spatial_alignment]") {
    initOpenVDB();
    SpatialAlignment aligner;
    auto source = makeGrid(1.0f, 0, 0, 0, 11.0f);
    auto target = makeGrid(1.0f, 0, 0, 0, 0.0f);
    double umin[] = { -1.0, -1.0, -1.0 };
    double umax[] = { 1.0, 1.0, 1.0 };
    FloatGridPtr result = aligner.alignToBoundingBox(
        source, target,
        umin[0], umin[1], umin[2], umax[0], umax[1], umax[2],
        "trilinear"
    );
    REQUIRE(result != nullptr);
    auto out_bbox = aligner.getWorldBoundingBox(result);
    REQUIRE(out_bbox.size() == 6);
}
