#ifndef AM_QADF_NATIVE_VOXELIZATION_UNIFIED_GRID_FACTORY_HPP
#define AM_QADF_NATIVE_VOXELIZATION_UNIFIED_GRID_FACTORY_HPP

#ifdef EIGEN_AVAILABLE
#include <Eigen/Dense>
#include <openvdb/openvdb.h>
#include <openvdb/Grid.h>
#include <openvdb/math/Transform.h>
#include "am_qadf_native/synchronization/point_bounds.hpp"
#include "am_qadf_native/synchronization/point_coordinate_transform.hpp"
#include <vector>
#include <memory>

namespace am_qadf_native {
namespace voxelization {

// Type aliases
using FloatGrid = openvdb::FloatGrid;
using FloatGridPtr = FloatGrid::Ptr;

// Types are fully defined in included headers:
// - synchronization::BoundingBox (from unified_bounds_computer.hpp)
// - synchronization::CoordinateSystem (from coordinate_transformer.hpp)

// UnifiedGridFactory: Creates grids with unified bounds and reference transform
// All grids share the same coordinate system and bounds for direct voxel-to-voxel operations
class UnifiedGridFactory {
public:
    // Create grid with unified bounds and reference transform
    // The grid will be created with a transform that maps from index space to world space
    FloatGridPtr createGrid(
        const synchronization::BoundingBox& unified_bounds,
        const synchronization::CoordinateSystem& reference_system,
        float voxel_size,
        float background_value = 0.0f
    );
    
    // Create multiple grids with same bounds and transform
    // Useful for creating grids for multiple signals (thermal, acoustic, etc.)
    std::vector<FloatGridPtr> createGrids(
        const synchronization::BoundingBox& unified_bounds,
        const synchronization::CoordinateSystem& reference_system,
        float voxel_size,
        size_t count,
        float background_value = 0.0f
    );
    
    // Create grid from existing grid's transform (for compatibility)
    // Uses transform from reference grid but with new bounds
    FloatGridPtr createFromReference(
        FloatGridPtr reference_grid,
        const synchronization::BoundingBox& unified_bounds,
        float background_value = 0.0f
    );

private:
    // Helper: Build OpenVDB transform from bounds and voxel size
    openvdb::math::Transform::Ptr buildTransform(
        const synchronization::BoundingBox& bounds,
        float voxel_size
    );
    
    // Helper: Build transform from coordinate system
    openvdb::math::Transform::Ptr buildTransformFromCoordinateSystem(
        const synchronization::CoordinateSystem& system,
        float voxel_size
    );
};

} // namespace voxelization
} // namespace am_qadf_native

#endif // EIGEN_AVAILABLE
#endif // AM_QADF_NATIVE_VOXELIZATION_UNIFIED_GRID_FACTORY_HPP
