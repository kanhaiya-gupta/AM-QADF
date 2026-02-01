/**
 * C++ integration tests: synchronization pipeline.
 * Chains: create source + reference grids → GridSynchronizer.synchronizeSpatial → verify output count.
 *
 * Aligned with docs/Tests/Test_New_Plans.md (tests/integration/cpp/).
 */

#include <catch2/catch_test_macros.hpp>
#include "am_qadf_native/voxelization/openvdb_grid.hpp"
#include "am_qadf_native/synchronization/grid_synchronizer.hpp"
#include <openvdb/openvdb.h>
#include <openvdb/math/Coord.h>
#include <vector>

using namespace am_qadf_native::voxelization;
using namespace am_qadf_native::synchronization;

namespace {

void initOpenVDB() {
    openvdb::initialize();
}

} // namespace

TEST_CASE("Synchronization pipeline: spatial sync two grids to reference", "[integration][synchronization]") {
    initOpenVDB();

    UniformVoxelGrid ref(1.0f, 0.0f, 0.0f, 0.0f);
    ref.addPointAtVoxel(0, 0, 0, 1.0f);
    openvdb::FloatGrid::Ptr reference = ref.getGrid();
    REQUIRE(reference);

    UniformVoxelGrid s1(1.0f, 0.0f, 0.0f, 0.0f);
    s1.addPointAtVoxel(0, 0, 0, 10.0f);
    UniformVoxelGrid s2(1.0f, 0.0f, 0.0f, 0.0f);
    s2.addPointAtVoxel(1, 0, 0, 20.0f);
    std::vector<openvdb::FloatGrid::Ptr> sources = { s1.getGrid(), s2.getGrid() };

    GridSynchronizer sync;
    std::vector<openvdb::FloatGrid::Ptr> out = sync.synchronizeSpatial(sources, reference, "trilinear");
    REQUIRE(out.size() == sources.size());
    REQUIRE(out[0]);
    REQUIRE(out[1]);
}

TEST_CASE("Synchronization pipeline: spatial sync single grid", "[integration][synchronization]") {
    initOpenVDB();

    UniformVoxelGrid ref(1.0f, 0.0f, 0.0f, 0.0f);
    ref.addPointAtVoxel(0, 0, 0, 0.0f);
    UniformVoxelGrid src(1.0f, 0.0f, 0.0f, 0.0f);
    src.addPointAtVoxel(0, 0, 0, 5.0f);

    GridSynchronizer sync;
    std::vector<openvdb::FloatGrid::Ptr> out = sync.synchronizeSpatial({ src.getGrid() }, ref.getGrid(), "trilinear");
    REQUIRE(out.size() == 1u);
    REQUIRE(out[0]);
}
