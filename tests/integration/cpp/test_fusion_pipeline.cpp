/**
 * C++ integration tests: fusion pipeline.
 * Chains: create two grids → GridFusion.fuse → verify → optional write/read.
 *
 * Aligned with docs/Tests/Test_New_Plans.md (tests/integration/cpp/).
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "am_qadf_native/voxelization/openvdb_grid.hpp"
#include "am_qadf_native/fusion/grid_fusion.hpp"
#include "am_qadf_native/io/vdb_writer.hpp"
#include "am_qadf_native/io/openvdb_reader.hpp"
#include <openvdb/openvdb.h>
#include <openvdb/math/Coord.h>
#include <vector>
#include <cstdio>
#include <string>

using namespace am_qadf_native::voxelization;
using namespace am_qadf_native::fusion;
using namespace am_qadf_native::io;

namespace {

const char* kTempVdb = "test_fusion_temp.vdb";

void initOpenVDB() {
    openvdb::initialize();
}

void removeTemp() {
    std::remove(kTempVdb);
}

} // namespace

TEST_CASE("Fusion pipeline: fuse two grids weighted_average", "[integration][fusion]") {
    initOpenVDB();
    removeTemp();

    UniformVoxelGrid g1(1.0f, 0.0f, 0.0f, 0.0f);
    g1.addPointAtVoxel(0, 0, 0, 10.0f);
    g1.addPointAtVoxel(1, 0, 0, 20.0f);
    UniformVoxelGrid g2(1.0f, 0.0f, 0.0f, 0.0f);
    g2.addPointAtVoxel(0, 0, 0, 30.0f);
    g2.addPointAtVoxel(1, 0, 0, 40.0f);

    std::vector<openvdb::FloatGrid::Ptr> grids = { g1.getGrid(), g2.getGrid() };
    GridFusion fusion;
    openvdb::FloatGrid::Ptr fused = fusion.fuse(grids, "weighted_average");
    REQUIRE(fused);

    openvdb::FloatGrid::ValueOnIter it = fused->beginValueOn();
    float v00 = fused->tree().getValue(openvdb::Coord(0, 0, 0));
    float v10 = fused->tree().getValue(openvdb::Coord(1, 0, 0));
    REQUIRE(v00 == Catch::Approx(20.0f));  // (10+30)/2
    REQUIRE(v10 == Catch::Approx(30.0f)); // (20+40)/2
}

TEST_CASE("Fusion pipeline: fuse then write-read", "[integration][fusion]") {
    initOpenVDB();
    removeTemp();

    UniformVoxelGrid g1(1.0f, 0.0f, 0.0f, 0.0f);
    g1.addPointAtVoxel(0, 0, 0, 1.0f);
    UniformVoxelGrid g2(1.0f, 0.0f, 0.0f, 0.0f);
    g2.addPointAtVoxel(0, 0, 0, 3.0f);

    GridFusion fusion;
    openvdb::FloatGrid::Ptr fused = fusion.fuse({ g1.getGrid(), g2.getGrid() }, "weighted_average");
    fused->setName("fused");
    VDBWriter writer;
    writer.write(fused, kTempVdb);

    OpenVDBReader reader;
    auto grids = reader.loadAllGrids(kTempVdb);
    REQUIRE(!grids.empty());
    openvdb::FloatGrid::Ptr read = grids.begin()->second;
    float v = read->tree().getValue(openvdb::Coord(0, 0, 0));
    REQUIRE(v == Catch::Approx(2.0f));

    removeTemp();
}
