/**
 * C++ unit tests for VDBWriter (vdb_writer.cpp).
 * Tests write, writeMultiple, writeMultipleWithNames, writeCompressed, append.
 *
 * Aligned with docs/Tests/Test_New_Plans.md (tests/unit/cpp/io/).
 */

#include <catch2/catch_test_macros.hpp>
#include "am_qadf_native/io/vdb_writer.hpp"
#include <openvdb/openvdb.h>
#include <openvdb/Grid.h>
#include <openvdb/math/Transform.h>
#include <openvdb/math/Coord.h>
#include <cstdio>
#include <string>
#include <vector>
#include <fstream>

using namespace am_qadf_native::io;

namespace {

const char* kTempVdbFile = "test_io_writer_temp.vdb";

void initOpenVDB() {
    openvdb::initialize();
}

FloatGridPtr makeGrid(float voxel_size, int ix, int iy, int iz, float value) {
    auto grid = openvdb::FloatGrid::create(0.0f);
    grid->setTransform(openvdb::math::Transform::createLinearTransform(voxel_size));
    grid->tree().setValue(openvdb::Coord(ix, iy, iz), value);
    return grid;
}

bool fileExists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

void removeTempVdb() {
    std::remove(kTempVdbFile);
}

} // namespace

TEST_CASE("VDBWriter: write creates file", "[io][vdb_writer]") {
    initOpenVDB();
    VDBWriter writer;
    FloatGridPtr grid = makeGrid(1.0f, 0, 0, 0, 7.0f);
    writer.write(grid, kTempVdbFile);
    REQUIRE(fileExists(kTempVdbFile));
    removeTempVdb();
}

TEST_CASE("VDBWriter: writeMultiple creates file", "[io][vdb_writer]") {
    initOpenVDB();
    VDBWriter writer;
    std::vector<FloatGridPtr> grids = {
        makeGrid(1.0f, 0, 0, 0, 1.0f),
        makeGrid(1.0f, 1, 0, 0, 2.0f)
    };
    writer.writeMultiple(grids, kTempVdbFile);
    REQUIRE(fileExists(kTempVdbFile));
    removeTempVdb();
}

TEST_CASE("VDBWriter: writeMultipleWithNames size mismatch throws", "[io][vdb_writer]") {
    initOpenVDB();
    VDBWriter writer;
    std::vector<FloatGridPtr> grids = { makeGrid(1.0f, 0, 0, 0, 1.0f) };
    std::vector<std::string> names = { "a", "b" };
    REQUIRE_THROWS_AS(writer.writeMultipleWithNames(grids, names, kTempVdbFile), std::invalid_argument);
}

TEST_CASE("VDBWriter: writeMultipleWithNames empty does not throw", "[io][vdb_writer]") {
    initOpenVDB();
    VDBWriter writer;
    std::vector<FloatGridPtr> grids;
    std::vector<std::string> names;
    writer.writeMultipleWithNames(grids, names, kTempVdbFile);
    REQUIRE(true);
}

TEST_CASE("VDBWriter: writeMultipleWithNames creates file", "[io][vdb_writer]") {
    initOpenVDB();
    VDBWriter writer;
    std::vector<FloatGridPtr> grids = {
        makeGrid(1.0f, 0, 0, 0, 1.0f),
        makeGrid(1.0f, 1, 0, 0, 2.0f)
    };
    std::vector<std::string> names = { "grid1", "grid2" };
    writer.writeMultipleWithNames(grids, names, kTempVdbFile);
    REQUIRE(fileExists(kTempVdbFile));
    removeTempVdb();
}

TEST_CASE("VDBWriter: writeCompressed creates file", "[io][vdb_writer]") {
    initOpenVDB();
    VDBWriter writer;
    FloatGridPtr grid = makeGrid(1.0f, 0, 0, 0, 3.0f);
    writer.writeCompressed(grid, kTempVdbFile, 6);
    REQUIRE(fileExists(kTempVdbFile));
    removeTempVdb();
}

TEST_CASE("VDBWriter: append creates or extends file", "[io][vdb_writer]") {
    initOpenVDB();
    VDBWriter writer;
    FloatGridPtr grid = makeGrid(1.0f, 0, 0, 0, 4.0f);
    writer.write(grid, kTempVdbFile);
    REQUIRE(fileExists(kTempVdbFile));
    FloatGridPtr grid2 = makeGrid(1.0f, 1, 1, 1, 5.0f);
    grid2->setName("second");
    writer.append(grid2, kTempVdbFile);
    removeTempVdb();
    REQUIRE(true);
}
