/**
 * C++ unit tests for TemporalAlignment (grid_temporal_alignment.cpp).
 * Tests synchronizeTemporal, GridWithMetadata, time bins, aggregateGridsInBin.
 *
 * Aligned with src/am_qadf_native/synchronization/grid_temporal_alignment.hpp
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "am_qadf_native/synchronization/grid_temporal_alignment.hpp"
#include <openvdb/openvdb.h>
#include <openvdb/Grid.h>
#include <openvdb/math/Transform.h>
#include <openvdb/math/Coord.h>
#include <vector>
#include <memory>

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

GridWithMetadata wrapGrid(FloatGridPtr grid, std::vector<float> timestamps, std::vector<int> layer_indices = {}) {
    GridWithMetadata g;
    g.grid = grid;
    g.timestamps = std::move(timestamps);
    g.layer_indices = std::move(layer_indices);
    return g;
}

} // namespace

// ---------------------------------------------------------------------------
// synchronizeTemporal - empty / single / multiple
// ---------------------------------------------------------------------------

TEST_CASE("TemporalAlignment: synchronizeTemporal empty grids returns empty vector", "[synchronization][grid_temporal_alignment]") {
    initOpenVDB();
    TemporalAlignment aligner;
    std::vector<GridWithMetadata> grids;
    auto result = aligner.synchronizeTemporal(grids, 0.1f, 1);
    REQUIRE(result.empty());
}

TEST_CASE("TemporalAlignment: synchronizeTemporal single grid one timestamp", "[synchronization][grid_temporal_alignment]") {
    initOpenVDB();
    TemporalAlignment aligner;
    auto grid = makeGrid(1.0f, 0, 0, 0, 5.0f);
    std::vector<GridWithMetadata> grids = { wrapGrid(grid, { 1.0f }) };
    auto result = aligner.synchronizeTemporal(grids, 0.5f, 1);
    REQUIRE(result.size() >= 1);
    REQUIRE(result[0] != nullptr);
    REQUIRE_THAT(result[0]->tree().getValue(openvdb::Coord(0, 0, 0)),
        Catch::Matchers::WithinAbs(5.0f, 1e-5f));
}

TEST_CASE("TemporalAlignment: synchronizeTemporal two grids same bin", "[synchronization][grid_temporal_alignment]") {
    initOpenVDB();
    TemporalAlignment aligner;
    auto g1 = makeGrid(1.0f, 0, 0, 0, 10.0f);
    auto g2 = makeGrid(1.0f, 0, 0, 0, 20.0f);
    std::vector<GridWithMetadata> grids = {
        wrapGrid(g1, { 1.0f }),
        wrapGrid(g2, { 1.05f })  // same time window (e.g. window 0.5)
    };
    auto result = aligner.synchronizeTemporal(grids, 0.5f, 1);
    REQUIRE(result.size() >= 1);
    REQUIRE(result[0] != nullptr);
    float v = result[0]->tree().getValue(openvdb::Coord(0, 0, 0));
    REQUIRE(v >= 10.0f - 1e-5f);
    REQUIRE(v <= 20.0f + 1e-5f);
}

TEST_CASE("TemporalAlignment: synchronizeTemporal two grids different bins", "[synchronization][grid_temporal_alignment]") {
    initOpenVDB();
    TemporalAlignment aligner;
    auto g1 = makeGrid(1.0f, 0, 0, 0, 10.0f);
    auto g2 = makeGrid(1.0f, 0, 0, 0, 20.0f);
    std::vector<GridWithMetadata> grids = {
        wrapGrid(g1, { 0.0f }),
        wrapGrid(g2, { 2.0f })
    };
    auto result = aligner.synchronizeTemporal(grids, 0.5f, 1);
    REQUIRE(result.size() >= 2);
    REQUIRE(result[0] != nullptr);
    REQUIRE(result[1] != nullptr);
    REQUIRE_THAT(result[0]->tree().getValue(openvdb::Coord(0, 0, 0)),
        Catch::Matchers::WithinAbs(10.0f, 1e-5f));
    REQUIRE_THAT(result[1]->tree().getValue(openvdb::Coord(0, 0, 0)),
        Catch::Matchers::WithinAbs(20.0f, 1e-5f));
}

TEST_CASE("TemporalAlignment: synchronizeTemporal grid with no timestamps excluded from bins", "[synchronization][grid_temporal_alignment]") {
    initOpenVDB();
    TemporalAlignment aligner;
    auto grid = makeGrid(1.0f, 0, 0, 0, 7.0f);
    std::vector<GridWithMetadata> grids = { wrapGrid(grid, {}) };
    auto result = aligner.synchronizeTemporal(grids, 0.1f, 1);
    REQUIRE(result.empty());
}

TEST_CASE("TemporalAlignment: synchronizeTemporal custom temporal_window and layer_tolerance", "[synchronization][grid_temporal_alignment]") {
    initOpenVDB();
    TemporalAlignment aligner;
    auto grid = makeGrid(1.0f, 1, 1, 1, 3.0f);
    std::vector<GridWithMetadata> grids = { wrapGrid(grid, { 0.5f, 1.0f }, { 0, 1 }) };
    auto result = aligner.synchronizeTemporal(grids, 0.2f, 2);
    REQUIRE(result.size() >= 1);
    REQUIRE(result[0] != nullptr);
}
