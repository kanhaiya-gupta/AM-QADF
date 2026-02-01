#ifndef AM_QADF_NATIVE_VOXELIZATION_STL_VOXELIZER_HPP
#define AM_QADF_NATIVE_VOXELIZATION_STL_VOXELIZER_HPP

#include <openvdb/openvdb.h>
#include <openvdb/Grid.h>
#include <openvdb/tools/MeshToVolume.h>
#include <openvdb/tools/VolumeToMesh.h>
#include <string>
#include <vector>
#include <memory>

namespace am_qadf_native {
namespace voxelization {

using FloatGrid = openvdb::FloatGrid;
using FloatGridPtr = FloatGrid::Ptr;

// STL Voxelizer using OpenVDB.
// STL voxelization is for geometry/bounding only (not the core task); coarse voxel sizes
// (e.g. >= 0.25f) and modest half_width are sufficient and avoid large memory use.
class STLVoxelizer {
public:
    // Voxelize STL file to OpenVDB FloatGrid
    // Creates a level set (signed distance field) from the STL mesh
    FloatGridPtr voxelizeSTL(
        const std::string& stl_path,
        float voxel_size,
        float half_width = 3.0f,  // Half-width of narrow band; use modest values (e.g. 1–2) for coarse grids
        bool unsigned_distance = false  // Use unsigned distance field
    );
    
    // Voxelize STL and fill with signal values
    // Creates voxel grid and fills voxels that intersect with geometry
    FloatGridPtr voxelizeSTLWithSignals(
        const std::string& stl_path,
        float voxel_size,
        const std::vector<std::array<float, 3>>& points,
        const std::vector<float>& signal_values,
        float half_width = 3.0f
    );
    
    // Get bounding box from STL file
    void getSTLBoundingBox(
        const std::string& stl_path,
        std::array<float, 3>& bbox_min,
        std::array<float, 3>& bbox_max
    );
    
private:
    // Helper: Load STL mesh and convert to OpenVDB format
    // Returns vertices and faces for meshToLevelSet
    bool loadSTLMesh(
        const std::string& stl_path,
        std::vector<openvdb::Vec3s>& points,
        std::vector<openvdb::Vec3I>& triangles
    );
    
    // Helper: Convert level set to binary occupancy grid
    FloatGridPtr levelSetToOccupancy(FloatGridPtr level_set);
};

} // namespace voxelization
} // namespace am_qadf_native

#endif // AM_QADF_NATIVE_VOXELIZATION_STL_VOXELIZER_HPP
