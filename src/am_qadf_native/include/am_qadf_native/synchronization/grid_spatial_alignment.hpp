#ifndef AM_QADF_NATIVE_SYNCHRONIZATION_GRID_SPATIAL_ALIGNMENT_HPP
#define AM_QADF_NATIVE_SYNCHRONIZATION_GRID_SPATIAL_ALIGNMENT_HPP

#include <openvdb/openvdb.h>
#include <openvdb/Grid.h>
#include <openvdb/tools/GridTransformer.h>
#include <string>
#include <memory>

namespace am_qadf_native {
namespace synchronization {

using FloatGrid = openvdb::FloatGrid;
using FloatGridPtr = FloatGrid::Ptr;

// Spatial alignment using OpenVDB's GridTransformer
class SpatialAlignment {
public:
    // Align source grid to target grid's coordinate system
    FloatGridPtr align(
        FloatGridPtr source_grid,
        FloatGridPtr target_grid,
        const std::string& method = "trilinear"  // "nearest", "trilinear", "triquadratic"
    );
    
    // Align with custom transformation matrix
    FloatGridPtr alignWithTransform(
        FloatGridPtr source_grid,
        FloatGridPtr target_grid,
        const openvdb::Mat4R& transform,
        const std::string& method = "trilinear"
    );
    
    // Verify if two grids have the same transform (within tolerance)
    bool transformsMatch(
        FloatGridPtr grid1,
        FloatGridPtr grid2,
        double tolerance = 1e-6
    ) const;
    
    // Get transform matrix as a 4x4 array (for Python access)
    std::vector<std::vector<double>> getTransformMatrix(FloatGridPtr grid) const;
    
    // Get world-space bounding box of a grid
    // Returns: {min_x, min_y, min_z, max_x, max_y, max_z}
    std::vector<double> getWorldBoundingBox(FloatGridPtr grid) const;
    
    // Align source grid to target transform AND resample into unified bounding box
    // This ensures all grids occupy the same world-space region
    FloatGridPtr alignToBoundingBox(
        FloatGridPtr source_grid,
        FloatGridPtr target_grid,
        double unified_min_x, double unified_min_y, double unified_min_z,
        double unified_max_x, double unified_max_y, double unified_max_z,
        const std::string& method = "trilinear"
    );
    
private:
    // Helper: Create transformation matrix from source to target
    openvdb::Mat4R computeTransform(
        FloatGridPtr source_grid,
        FloatGridPtr target_grid
    ) const;
};

} // namespace synchronization
} // namespace am_qadf_native

#endif // AM_QADF_NATIVE_SYNCHRONIZATION_GRID_SPATIAL_ALIGNMENT_HPP
