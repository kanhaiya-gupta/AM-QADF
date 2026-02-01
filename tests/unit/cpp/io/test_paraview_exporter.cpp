/**
 * C++ unit tests for ParaViewExporter (paraview_exporter.cpp).
 * Tests exportToParaView, exportMultipleToParaView, exportWithMetadata.
 *
 * Aligned with docs/Tests/Test_New_Plans.md (tests/unit/cpp/io/).
 */

#include <catch2/catch_test_macros.hpp>
#include "am_qadf_native/io/paraview_exporter.hpp"
#include <openvdb/openvdb.h>
#include <openvdb/Grid.h>
#include <openvdb/math/Transform.h>
#include <openvdb/math/Coord.h>
#include <cstdio>
#include <string>
#include <vector>
#include <map>
#include <fstream>

using namespace am_qadf_native::io;

namespace {

const char* kTempVdbFile = "test_io_paraview_temp.vdb";

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

TEST_CASE("ParaViewExporter: exportToParaView creates file", "[io][paraview_exporter]") {
    initOpenVDB();
    ParaViewExporter exporter;
    FloatGridPtr grid = makeGrid(1.0f, 0, 0, 0, 10.0f);
    exporter.exportToParaView(grid, kTempVdbFile);
    REQUIRE(fileExists(kTempVdbFile));
    removeTempVdb();
}

TEST_CASE("ParaViewExporter: exportMultipleToParaView creates file", "[io][paraview_exporter]") {
    initOpenVDB();
    ParaViewExporter exporter;
    std::vector<FloatGridPtr> grids = {
        makeGrid(1.0f, 0, 0, 0, 1.0f),
        makeGrid(1.0f, 1, 0, 0, 2.0f)
    };
    exporter.exportMultipleToParaView(grids, kTempVdbFile);
    REQUIRE(fileExists(kTempVdbFile));
    removeTempVdb();
}

TEST_CASE("ParaViewExporter: exportWithMetadata creates file", "[io][paraview_exporter]") {
    initOpenVDB();
    ParaViewExporter exporter;
    FloatGridPtr grid = makeGrid(1.0f, 0, 0, 0, 5.0f);
    std::map<std::string, std::string> metadata;
    metadata["source"] = "test";
    metadata["version"] = "1.0";
    exporter.exportWithMetadata(grid, kTempVdbFile, metadata);
    REQUIRE(fileExists(kTempVdbFile));
    removeTempVdb();
}
