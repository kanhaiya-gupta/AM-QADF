/**
 * C++ integration tests: grid operations pipeline.
 * Chains: UniformVoxelGrid → add points → get grid → write (VDBWriter) → read (OpenVDBReader) → verify.
 *
 * Aligned with docs/Tests/Test_New_Plans.md (tests/integration/cpp/).
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "am_qadf_native/voxelization/openvdb_grid.hpp"
#include "am_qadf_native/io/openvdb_reader.hpp"
#include "am_qadf_native/io/vdb_writer.hpp"
#include <openvdb/openvdb.h>
#include <openvdb/Grid.h>
#include <openvdb/math/Coord.h>
#include <cstdio>
#include <string>

using namespace am_qadf_native::voxelization;
using namespace am_qadf_native::io;

namespace {

const char* kTempVdb = "test_grid_ops_temp.vdb";

void initOpenVDB() {
    openvdb::initialize();
}

void removeTemp() {
    std::remove(kTempVdb);
}

} // namespace

TEST_CASE("Grid operations integration: voxel grid write-read round-trip", "[integration][grid_operations]") {
    initOpenVDB();
    removeTemp();

    float voxelSize = 1.0f;
    UniformVoxelGrid grid(voxelSize, 0.0f, 0.0f, 0.0f);
    grid.setSignalName("signal");
    grid.addPointAtVoxel(0, 0, 0, 5.0f);
    grid.addPointAtVoxel(1, 0, 0, 10.0f);
    grid.addPointAtVoxel(0, 1, 0, 15.0f);

    openvdb::FloatGrid::Ptr g = grid.getGrid();
    REQUIRE(g);
    REQUIRE(grid.getValue(0, 0, 0) == Catch::Approx(5.0f));
    REQUIRE(grid.getValue(1, 0, 0) == Catch::Approx(10.0f));

    VDBWriter writer;
    writer.write(g, kTempVdb);

    OpenVDBReader reader;
    auto grids = reader.loadAllGrids(kTempVdb);
    REQUIRE(!grids.empty());
    auto it = grids.find("signal");
    if (it == grids.end())
        it = grids.begin();
    openvdb::FloatGrid::Ptr read = it->second;
    REQUIRE(read);
    float v0 = read->tree().getValue(openvdb::Coord(0, 0, 0));
    float v1 = read->tree().getValue(openvdb::Coord(1, 0, 0));
    REQUIRE(v0 == Catch::Approx(5.0f));
    REQUIRE(v1 == Catch::Approx(10.0f));

    removeTemp();
}

TEST_CASE("Grid operations integration: addPoint world coords then write", "[integration][grid_operations]") {
    initOpenVDB();
    removeTemp();

    UniformVoxelGrid grid(1.0f, 0.0f, 0.0f, 0.0f);
    grid.setSignalName("world_signal");
    grid.addPoint(0.5f, 0.5f, 0.5f, 7.0f);
    grid.addPoint(1.5f, 0.5f, 0.5f, 8.0f);

    openvdb::FloatGrid::Ptr g = grid.getGrid();
    REQUIRE(g);
    VDBWriter writer;
    writer.write(g, kTempVdb);

    OpenVDBReader reader;
    auto grids = reader.loadAllGrids(kTempVdb);
    REQUIRE(!grids.empty());
    openvdb::FloatGrid::Ptr read = grids.begin()->second;
    REQUIRE(read);
    float v = read->tree().getValue(openvdb::Coord(0, 0, 0));
    REQUIRE(v == Catch::Approx(7.0f));

    removeTemp();
}
