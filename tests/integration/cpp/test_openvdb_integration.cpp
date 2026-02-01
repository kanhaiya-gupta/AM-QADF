/**
 * C++ integration tests: OpenVDB read/write round-trip and grid lifecycle.
 * Chains: create grid → write (VDBWriter) → read (OpenVDBReader) → verify.
 *
 * Aligned with docs/Tests/Test_New_Plans.md (tests/integration/cpp/).
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "am_qadf_native/io/openvdb_reader.hpp"
#include "am_qadf_native/io/vdb_writer.hpp"
#include <openvdb/openvdb.h>
#include <openvdb/Grid.h>
#include <openvdb/math/Transform.h>
#include <openvdb/math/Coord.h>
#include <cstdio>
#include <string>

using namespace am_qadf_native::io;

namespace {

const char* kTempVdb = "test_openvdb_integration_temp.vdb";

void initOpenVDB() {
    openvdb::initialize();
}

openvdb::FloatGrid::Ptr makeSimpleGrid(float voxelSize, float valueAtOrigin) {
    auto grid = openvdb::FloatGrid::create(0.0f);
    grid->setTransform(openvdb::math::Transform::createLinearTransform(voxelSize));
    grid->tree().setValue(openvdb::Coord(0, 0, 0), valueAtOrigin);
    grid->tree().setValue(openvdb::Coord(1, 0, 0), valueAtOrigin * 2.0f);
    grid->setName("signal");
    return grid;
}

void removeTemp() {
    std::remove(kTempVdb);
}

} // namespace

TEST_CASE("OpenVDB integration: write then read round-trip", "[integration][openvdb]") {
    initOpenVDB();
    removeTemp();

    float voxelSize = 1.0f;
    float value = 5.0f;
    openvdb::FloatGrid::Ptr written = makeSimpleGrid(voxelSize, value);

    VDBWriter writer;
    writer.write(written, kTempVdb);

    OpenVDBReader reader;
    auto grids = reader.loadAllGrids(kTempVdb);
    REQUIRE(!grids.empty());
    auto it = grids.find("signal");
    REQUIRE(it != grids.end());
    openvdb::FloatGrid::Ptr read = it->second;
    REQUIRE(read);
    REQUIRE(read->transform().voxelSize()[0] == Catch::Approx(voxelSize));
    float v0 = read->tree().getValue(openvdb::Coord(0, 0, 0));
    float v1 = read->tree().getValue(openvdb::Coord(1, 0, 0));
    REQUIRE(v0 == Catch::Approx(value));
    REQUIRE(v1 == Catch::Approx(value * 2.0f));

    removeTemp();
}

TEST_CASE("OpenVDB integration: loadGridByName after writeMultipleWithNames", "[integration][openvdb]") {
    initOpenVDB();
    removeTemp();

    auto g1 = makeSimpleGrid(1.0f, 1.0f);
    g1->setName("grid_a");
    auto g2 = makeSimpleGrid(0.5f, 2.0f);
    g2->setName("grid_b");

    VDBWriter writer;
    writer.writeMultipleWithNames({g1, g2}, {"grid_a", "grid_b"}, kTempVdb);

    OpenVDBReader reader;
    openvdb::FloatGrid::Ptr a = reader.loadGridByName(kTempVdb, "grid_a");
    openvdb::FloatGrid::Ptr b = reader.loadGridByName(kTempVdb, "grid_b");
    REQUIRE(a);
    REQUIRE(b);
    REQUIRE(a->tree().getValue(openvdb::Coord(0, 0, 0)) == Catch::Approx(1.0f));
    REQUIRE(b->tree().getValue(openvdb::Coord(0, 0, 0)) == Catch::Approx(2.0f));

    removeTemp();
}
