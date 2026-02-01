/**
 * C++ unit tests for GridFusion (grid_fusion.cpp).
 * Tests fuse (weighted_average, max, min, median) and fuseWeighted.
 *
 * Aligned with src/am_qadf_native/fusion/grid_fusion.hpp
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "am_qadf_native/fusion/grid_fusion.hpp"
#include <openvdb/openvdb.h>
#include <openvdb/Grid.h>
#include <openvdb/math/Transform.h>
#include <openvdb/math/Coord.h>
#include <vector>
#include <string>
#include <limits>

using namespace am_qadf_native::fusion;

namespace {

void initOpenVDB() {
    openvdb::initialize();
}

FloatGridPtr makeGridWithValue(float voxel_size, int ix, int iy, int iz, float value) {
    auto grid = openvdb::FloatGrid::create(0.0f);
    grid->setTransform(openvdb::math::Transform::createLinearTransform(voxel_size));
    grid->tree().setValue(openvdb::Coord(ix, iy, iz), value);
    return grid;
}

FloatGridPtr makeGridWithValues(float voxel_size,
    const std::vector<std::tuple<int, int, int, float>>& coord_values) {
    auto grid = openvdb::FloatGrid::create(0.0f);
    grid->setTransform(openvdb::math::Transform::createLinearTransform(voxel_size));
    for (const auto& t : coord_values) {
        grid->tree().setValue(
            openvdb::Coord(std::get<0>(t), std::get<1>(t), std::get<2>(t)),
            std::get<3>(t)
        );
    }
    return grid;
}

} // namespace

// ---------------------------------------------------------------------------
// fuse
// ---------------------------------------------------------------------------

TEST_CASE("GridFusion: fuse empty grids returns empty grid", "[fusion][grid_fusion]") {
    initOpenVDB();
    GridFusion fusion;
    std::vector<FloatGridPtr> grids;
    FloatGridPtr result = fusion.fuse(grids);
    REQUIRE(result != nullptr);
}

TEST_CASE("GridFusion: fuse single grid", "[fusion][grid_fusion]") {
    initOpenVDB();
    GridFusion fusion;
    std::vector<FloatGridPtr> grids = { makeGridWithValue(1.0f, 0, 0, 0, 42.0f) };
    FloatGridPtr result = fusion.fuse(grids, "weighted_average");
    REQUIRE(result != nullptr);
    REQUIRE_THAT(result->tree().getValue(openvdb::Coord(0, 0, 0)),
        Catch::Matchers::WithinAbs(42.0f, 1e-6f));
}

TEST_CASE("GridFusion: fuse weighted_average two grids", "[fusion][grid_fusion]") {
    initOpenVDB();
    GridFusion fusion;
    std::vector<FloatGridPtr> grids = {
        makeGridWithValue(1.0f, 0, 0, 0, 10.0f),
        makeGridWithValue(1.0f, 0, 0, 0, 20.0f)
    };
    FloatGridPtr result = fusion.fuse(grids, "weighted_average");
    REQUIRE(result != nullptr);
    float val = result->tree().getValue(openvdb::Coord(0, 0, 0));
    REQUIRE_THAT(val, Catch::Matchers::WithinAbs(15.0f, 1e-5f));
}

TEST_CASE("GridFusion: fuse max two grids", "[fusion][grid_fusion]") {
    initOpenVDB();
    GridFusion fusion;
    std::vector<FloatGridPtr> grids = {
        makeGridWithValue(1.0f, 0, 0, 0, 10.0f),
        makeGridWithValue(1.0f, 0, 0, 0, 30.0f)
    };
    FloatGridPtr result = fusion.fuse(grids, "max");
    REQUIRE(result != nullptr);
    REQUIRE_THAT(result->tree().getValue(openvdb::Coord(0, 0, 0)),
        Catch::Matchers::WithinAbs(30.0f, 1e-6f));
}

TEST_CASE("GridFusion: fuse min two grids", "[fusion][grid_fusion]") {
    initOpenVDB();
    GridFusion fusion;
    std::vector<FloatGridPtr> grids = {
        makeGridWithValue(1.0f, 0, 0, 0, 10.0f),
        makeGridWithValue(1.0f, 0, 0, 0, 30.0f)
    };
    FloatGridPtr result = fusion.fuse(grids, "min");
    REQUIRE(result != nullptr);
    REQUIRE_THAT(result->tree().getValue(openvdb::Coord(0, 0, 0)),
        Catch::Matchers::WithinAbs(10.0f, 1e-6f));
}

TEST_CASE("GridFusion: fuse median three grids", "[fusion][grid_fusion]") {
    initOpenVDB();
    GridFusion fusion;
    std::vector<FloatGridPtr> grids = {
        makeGridWithValue(1.0f, 0, 0, 0, 10.0f),
        makeGridWithValue(1.0f, 0, 0, 0, 20.0f),
        makeGridWithValue(1.0f, 0, 0, 0, 30.0f)
    };
    FloatGridPtr result = fusion.fuse(grids, "median");
    REQUIRE(result != nullptr);
    REQUIRE_THAT(result->tree().getValue(openvdb::Coord(0, 0, 0)),
        Catch::Matchers::WithinAbs(20.0f, 1e-6f));
}

TEST_CASE("GridFusion: fuse union of coordinates", "[fusion][grid_fusion]") {
    initOpenVDB();
    GridFusion fusion;
    // Grid1: (0,0,0)=10. Grid2: (1,1,1)=20. Union: both coords in output.
    std::vector<FloatGridPtr> grids = {
        makeGridWithValue(1.0f, 0, 0, 0, 10.0f),
        makeGridWithValue(1.0f, 1, 1, 1, 20.0f)
    };
    FloatGridPtr result = fusion.fuse(grids, "weighted_average");
    REQUIRE(result != nullptr);
    REQUIRE_THAT(result->tree().getValue(openvdb::Coord(0, 0, 0)),
        Catch::Matchers::WithinAbs(10.0f, 1e-6f));
    REQUIRE_THAT(result->tree().getValue(openvdb::Coord(1, 1, 1)),
        Catch::Matchers::WithinAbs(20.0f, 1e-6f));
}

// ---------------------------------------------------------------------------
// fuseWeighted
// ---------------------------------------------------------------------------

TEST_CASE("GridFusion: fuseWeighted two grids", "[fusion][grid_fusion]") {
    initOpenVDB();
    GridFusion fusion;
    std::vector<FloatGridPtr> grids = {
        makeGridWithValue(1.0f, 0, 0, 0, 10.0f),
        makeGridWithValue(1.0f, 0, 0, 0, 30.0f)
    };
    std::vector<float> weights = { 0.25f, 0.75f };  // 0.25*10 + 0.75*30 = 25
    FloatGridPtr result = fusion.fuseWeighted(grids, weights);
    REQUIRE(result != nullptr);
    float val = result->tree().getValue(openvdb::Coord(0, 0, 0));
    REQUIRE_THAT(val, Catch::Matchers::WithinAbs(25.0f, 1e-5f));
}

TEST_CASE("GridFusion: fuseWeighted empty grids returns empty grid", "[fusion][grid_fusion]") {
    initOpenVDB();
    GridFusion fusion;
    std::vector<FloatGridPtr> grids;
    std::vector<float> weights;
    FloatGridPtr result = fusion.fuseWeighted(grids, weights);
    REQUIRE(result != nullptr);
}

TEST_CASE("GridFusion: fuseWeighted grids and weights size mismatch throws", "[fusion][grid_fusion]") {
    initOpenVDB();
    GridFusion fusion;
    std::vector<FloatGridPtr> grids = { makeGridWithValue(1.0f, 0, 0, 0, 1.0f) };
    std::vector<float> weights = { 0.5f, 0.5f };
    REQUIRE_THROWS_AS(fusion.fuseWeighted(grids, weights), std::invalid_argument);
}

TEST_CASE("GridFusion: fuseWeighted sum of weights zero throws", "[fusion][grid_fusion]") {
    initOpenVDB();
    GridFusion fusion;
    std::vector<FloatGridPtr> grids = {
        makeGridWithValue(1.0f, 0, 0, 0, 1.0f),
        makeGridWithValue(1.0f, 0, 0, 0, 2.0f)
    };
    std::vector<float> weights = { 0.0f, 0.0f };
    REQUIRE_THROWS_AS(fusion.fuseWeighted(grids, weights), std::invalid_argument);
}

TEST_CASE("GridFusion: fuseWeighted sum of weights negative throws", "[fusion][grid_fusion]") {
    initOpenVDB();
    GridFusion fusion;
    std::vector<FloatGridPtr> grids = {
        makeGridWithValue(1.0f, 0, 0, 0, 1.0f)
    };
    std::vector<float> weights = { -1.0f };
    REQUIRE_THROWS_AS(fusion.fuseWeighted(grids, weights), std::invalid_argument);
}

TEST_CASE("GridFusion: fuseWeighted normalizes weights", "[fusion][grid_fusion]") {
    initOpenVDB();
    GridFusion fusion;
    std::vector<FloatGridPtr> grids = {
        makeGridWithValue(1.0f, 0, 0, 0, 10.0f),
        makeGridWithValue(1.0f, 0, 0, 0, 30.0f)
    };
    std::vector<float> weights = { 2.0f, 6.0f };  // normalized to 0.25, 0.75
    FloatGridPtr result = fusion.fuseWeighted(grids, weights);
    REQUIRE(result != nullptr);
    float val = result->tree().getValue(openvdb::Coord(0, 0, 0));
    REQUIRE_THAT(val, Catch::Matchers::WithinAbs(25.0f, 1e-5f));
}
