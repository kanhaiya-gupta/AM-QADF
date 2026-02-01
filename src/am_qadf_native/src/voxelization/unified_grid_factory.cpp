#ifdef EIGEN_AVAILABLE

#include "am_qadf_native/voxelization/unified_grid_factory.hpp"
#include "am_qadf_native/synchronization/point_bounds.hpp"
#include "am_qadf_native/synchronization/point_coordinate_transform.hpp"

using namespace am_qadf_native::synchronization;
#include <openvdb/math/Transform.h>
#include <stdexcept>

namespace am_qadf_native {
namespace voxelization {

// Helper: Build OpenVDB transform from bounds and voxel size
openvdb::math::Transform::Ptr UnifiedGridFactory::buildTransform(
    const BoundingBox& bounds,
    float voxel_size
) {
    if (!bounds.isValid()) {
        throw std::invalid_argument("Bounding box is invalid");
    }
    
    if (voxel_size <= 0.0f) {
        throw std::invalid_argument("Voxel size must be positive");
    }
    
    // Create linear transform: maps index space to world space
    // Transform: world = index * voxel_size + offset
    // Offset is the minimum corner of the bounding box
    openvdb::Vec3d offset(
        bounds.min_x,
        bounds.min_y,
        bounds.min_z
    );
    
    // Create uniform scale transform
    openvdb::math::Transform::Ptr transform = 
        openvdb::math::Transform::createLinearTransform(voxel_size);
    
    // Set translation (offset)
    transform->postTranslate(offset);
    
    return transform;
}

// Helper: Build transform from coordinate system
openvdb::math::Transform::Ptr UnifiedGridFactory::buildTransformFromCoordinateSystem(
    const synchronization::CoordinateSystem& system,
    float voxel_size
) {
    if (voxel_size <= 0.0f) {
        throw std::invalid_argument("Voxel size must be positive");
    }
    
    // Use CoordinateTransformer to build transformation matrix
    synchronization::CoordinateTransformer transformer;
    Eigen::Matrix4d eigen_transform = transformer.buildTransformMatrix(system);
    
    // Apply voxel_size scaling to the rotation/scale part
    eigen_transform.block<3, 3>(0, 0) *= voxel_size;
    
    // Convert Eigen::Matrix4d to OpenVDB Mat4d
    openvdb::math::Mat4d mat4d(
        eigen_transform(0, 0), eigen_transform(0, 1), eigen_transform(0, 2), eigen_transform(0, 3),
        eigen_transform(1, 0), eigen_transform(1, 1), eigen_transform(1, 2), eigen_transform(1, 3),
        eigen_transform(2, 0), eigen_transform(2, 1), eigen_transform(2, 2), eigen_transform(2, 3),
        eigen_transform(3, 0), eigen_transform(3, 1), eigen_transform(3, 2), eigen_transform(3, 3)
    );
    
    // Create transform from 4x4 matrix
    openvdb::math::Transform::Ptr transform = 
        openvdb::math::Transform::createLinearTransform(mat4d);
    
    return transform;
}

// Create grid with unified bounds and reference transform
FloatGridPtr UnifiedGridFactory::createGrid(
    const synchronization::BoundingBox& unified_bounds,
    const synchronization::CoordinateSystem& reference_system,
    float voxel_size,
    float background_value
) {
    if (!unified_bounds.isValid()) {
        throw std::invalid_argument("Bounding box is invalid");
    }
    // Build transform from coordinate system
    openvdb::math::Transform::Ptr transform = 
        buildTransformFromCoordinateSystem(reference_system, voxel_size);
    
    // Create grid with transform
    FloatGridPtr grid = FloatGrid::create(background_value);
    grid->setTransform(transform);
    
    // Set grid name (optional, for identification)
    grid->setName("unified_grid");
    
    return grid;
}

// Create multiple grids with same bounds and transform
std::vector<FloatGridPtr> UnifiedGridFactory::createGrids(
    const synchronization::BoundingBox& unified_bounds,
    const synchronization::CoordinateSystem& reference_system,
    float voxel_size,
    size_t count,
    float background_value
) {
    std::vector<FloatGridPtr> grids;
    grids.reserve(count);
    
    // Build transform once (shared by all grids)
    openvdb::math::Transform::Ptr transform = 
        buildTransformFromCoordinateSystem(reference_system, voxel_size);
    
    for (size_t i = 0; i < count; ++i) {
        FloatGridPtr grid = FloatGrid::create(background_value);
        grid->setTransform(transform);
        grid->setName("unified_grid_" + std::to_string(i));
        grids.push_back(grid);
    }
    
    return grids;
}

// Create grid from existing grid's transform (for compatibility)
FloatGridPtr UnifiedGridFactory::createFromReference(
    FloatGridPtr reference_grid,
    const synchronization::BoundingBox& unified_bounds,
    float background_value
) {
    if (!reference_grid) {
        throw std::invalid_argument("Reference grid cannot be null");
    }
    
    // Get transform from reference grid
    openvdb::math::Transform::Ptr transform = reference_grid->transform().copy();
    
    // Create new grid with same transform
    FloatGridPtr grid = FloatGrid::create(background_value);
    grid->setTransform(transform);
    grid->setName("unified_grid_from_reference");
    
    return grid;
}

} // namespace voxelization
} // namespace am_qadf_native

#endif // EIGEN_AVAILABLE
