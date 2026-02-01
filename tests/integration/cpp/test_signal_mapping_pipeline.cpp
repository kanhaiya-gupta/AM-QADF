/**
 * C++ integration tests: signal mapping pipeline.
 * Chains: create grid → NearestNeighborMapper.map(points, values) → verify grid values → optional write/read.
 *
 * Aligned with docs/Tests/Test_New_Plans.md (tests/integration/cpp/).
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "am_qadf_native/voxelization/openvdb_grid.hpp"
#include "am_qadf_native/signal_mapping/nearest_neighbor.hpp"
#include "am_qadf_native/io/vdb_writer.hpp"
#include "am_qadf_native/io/openvdb_reader.hpp"
#include <openvdb/openvdb.h>
#include <openvdb/math/Coord.h>
#include <vector>
#include <cstdio>
#include <string>

using namespace am_qadf_native::voxelization;
using namespace am_qadf_native::signal_mapping;
using namespace am_qadf_native::io;

namespace {

const char* kTempVdb = "test_signal_mapping_temp.vdb";

void initOpenVDB() {
    openvdb::initialize();
}

void removeTemp() {
    std::remove(kTempVdb);
}

} // namespace

TEST_CASE("Signal mapping pipeline: nearest-neighbor map then verify", "[integration][signal_mapping]") {
    initOpenVDB();
    removeTemp();

    float voxelSize = 1.0f;
    UniformVoxelGrid grid(voxelSize, 0.0f, 0.0f, 0.0f);
    grid.setSignalName("mapped");
    openvdb::FloatGrid::Ptr g = grid.getGrid();
    REQUIRE(g);

    std::vector<Point> points = {
        Point(0.5f, 0.5f, 0.5f),
        Point(1.5f, 0.5f, 0.5f),
        Point(0.5f, 1.5f, 0.5f)
    };
    std::vector<float> values = { 10.0f, 20.0f, 30.0f };

    NearestNeighborMapper mapper;
    mapper.map(g, points, values);

    REQUIRE(grid.getValue(0, 0, 0) == Catch::Approx(10.0f));
    REQUIRE(grid.getValue(1, 0, 0) == Catch::Approx(20.0f));
    REQUIRE(grid.getValue(0, 1, 0) == Catch::Approx(30.0f));
}

TEST_CASE("Signal mapping pipeline: map then write-read round-trip", "[integration][signal_mapping]") {
    initOpenVDB();
    removeTemp();

    UniformVoxelGrid grid(1.0f, 0.0f, 0.0f, 0.0f);
    grid.setSignalName("mapped");
    openvdb::FloatGrid::Ptr g = grid.getGrid();
    std::vector<Point> points = { Point(0.5f, 0.5f, 0.5f) };
    std::vector<float> values = { 42.0f };
    NearestNeighborMapper mapper;
    mapper.map(g, points, values);

    VDBWriter writer;
    writer.write(g, kTempVdb);

    OpenVDBReader reader;
    auto grids = reader.loadAllGrids(kTempVdb);
    REQUIRE(!grids.empty());
    openvdb::FloatGrid::Ptr read = grids.begin()->second;
    float v = read->tree().getValue(openvdb::Coord(0, 0, 0));
    REQUIRE(v == Catch::Approx(42.0f));

    removeTemp();
}
