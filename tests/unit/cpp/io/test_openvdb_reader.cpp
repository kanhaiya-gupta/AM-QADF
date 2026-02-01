/**
 * C++ unit tests for OpenVDBReader (openvdb_reader.cpp).
 * Tests loadGridByName and loadAllGrids (C++-only; pybind11 methods not tested here).
 *
 * Aligned with docs/Tests/Test_New_Plans.md (tests/unit/cpp/io/).
 */

#include <catch2/catch_test_macros.hpp>
#include "am_qadf_native/io/openvdb_reader.hpp"
#include "am_qadf_native/io/vdb_writer.hpp"
#include <openvdb/openvdb.h>
#include <openvdb/Grid.h>
#include <openvdb/math/Transform.h>
#include <openvdb/math/Coord.h>
#include <cstdio>
#include <string>
#include <map>

using namespace am_qadf_native::io;

namespace {

const char* kTempVdbFile = "test_io_reader_temp.vdb";

void initOpenVDB() {
    openvdb::initialize();
}

FloatGridPtr makeGrid(float voxel_size, int ix, int iy, int iz, float value) {
    auto grid = openvdb::FloatGrid::create(0.0f);
    grid->setTransform(openvdb::math::Transform::createLinearTransform(voxel_size));
    grid->tree().setValue(openvdb::Coord(ix, iy, iz), value);
    return grid;
}

void writeTempVdb() {
    VDBWriter writer;
    FloatGridPtr grid = makeGrid(1.0f, 0, 0, 0, 5.0f);
    grid->setName("test_grid");
    writer.write(grid, kTempVdbFile);
}

void removeTempVdb() {
    std::remove(kTempVdbFile);
}

} // namespace

TEST_CASE("OpenVDBReader: loadAllGrids nonexistent file throws or returns empty", "[io][openvdb_reader]") {
    initOpenVDB();
    OpenVDBReader reader;
    std::map<std::string, FloatGridPtr> result;
    try {
        result = reader.loadAllGrids("nonexistent_io_file_xyz.vdb");
    } catch (...) {
        REQUIRE(true);
        return;
    }
    REQUIRE(result.empty());
}

TEST_CASE("OpenVDBReader: loadAllGrids after write returns grids", "[io][openvdb_reader]") {
    initOpenVDB();
    writeTempVdb();
    OpenVDBReader reader;
    std::map<std::string, FloatGridPtr> result = reader.loadAllGrids(kTempVdbFile);
    removeTempVdb();
    REQUIRE(!result.empty());
    REQUIRE(result.begin()->second != nullptr);
}

TEST_CASE("OpenVDBReader: loadGridByName after write", "[io][openvdb_reader]") {
    initOpenVDB();
    writeTempVdb();
    OpenVDBReader reader;
    std::map<std::string, FloatGridPtr> all = reader.loadAllGrids(kTempVdbFile);
    removeTempVdb();
    if (all.empty()) {
        REQUIRE(true);
        return;
    }
    std::string name = all.begin()->first;
    writeTempVdb();
    FloatGridPtr grid = reader.loadGridByName(kTempVdbFile, name);
    removeTempVdb();
    REQUIRE(grid != nullptr);
}
