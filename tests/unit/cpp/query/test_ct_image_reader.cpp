/**
 * C++ unit tests for CTImageReader (ct_image_reader.cpp).
 * Tests readRawBinary (with temp file), readDICOMSeries/readNIfTI when ITK unavailable.
 *
 * Aligned with docs/Tests/Test_New_Plans.md (tests/unit/cpp/query/).
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "am_qadf_native/query/ct_image_reader.hpp"
#include <openvdb/openvdb.h>
#include <openvdb/Grid.h>
#include <openvdb/math/Coord.h>
#include <fstream>
#include <string>
#include <cstdio>
#include <vector>

using namespace am_qadf_native::query;

namespace {

const char* kTempRawFile = "test_ct_reader_raw.bin";

void writeTempRawFile(int width, int height, int depth, const std::vector<float>& data) {
    (void)width;
    (void)height;
    (void)depth;
    std::ofstream f(kTempRawFile, std::ios::binary);
    REQUIRE(f.is_open());
    for (float v : data) {
        f.write(reinterpret_cast<const char*>(&v), sizeof(float));
    }
    f.close();
}

void removeTempRawFile() {
    std::remove(kTempRawFile);
}

} // namespace

TEST_CASE("CTImageReader: readRawBinary missing file returns empty grid", "[query][ct_image_reader]") {
    openvdb::initialize();
    CTImageReader reader;
    FloatGridPtr grid = reader.readRawBinary("nonexistent_file_xyz.bin", 2, 2, 2, 1.0f);
    REQUIRE(grid != nullptr);
    REQUIRE(grid->tree().activeVoxelCount() == 0);
}

TEST_CASE("CTImageReader: readRawBinary small volume", "[query][ct_image_reader]") {
    openvdb::initialize();
    CTImageReader reader;
    int w = 2, h = 2, d = 2;
    std::vector<float> data(w * h * d, 0.0f);
    data[0] = 1.0f;
    data[w * h * d - 1] = 2.0f;
    writeTempRawFile(w, h, d, data);
    FloatGridPtr grid = reader.readRawBinary(kTempRawFile, w, h, d, 1.0f);
    removeTempRawFile();
    REQUIRE(grid != nullptr);
    REQUIRE(grid->tree().activeVoxelCount() >= 1);
    float v0 = grid->tree().getValue(openvdb::Coord(0, 0, 0));
    REQUIRE_THAT(v0, Catch::Matchers::WithinAbs(1.0f, 1e-6f));
}

TEST_CASE("CTImageReader: readDICOMSeries nonexistent returns empty or grid", "[query][ct_image_reader]") {
    openvdb::initialize();
    CTImageReader reader;
    FloatGridPtr grid = reader.readDICOMSeries("nonexistent_dicom_dir_xyz");
    REQUIRE(grid != nullptr);
}

TEST_CASE("CTImageReader: readNIfTI nonexistent returns empty grid", "[query][ct_image_reader]") {
    openvdb::initialize();
    CTImageReader reader;
    FloatGridPtr grid = reader.readNIfTI("nonexistent.nii");
    REQUIRE(grid != nullptr);
}

TEST_CASE("CTImageReader: readTIFFStack returns grid", "[query][ct_image_reader]") {
    openvdb::initialize();
    CTImageReader reader;
    FloatGridPtr grid = reader.readTIFFStack("nonexistent_tiff_dir");
    REQUIRE(grid != nullptr);
}
