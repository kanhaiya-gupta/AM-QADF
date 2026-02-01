#include "am_qadf_native/synchronization/grid_spatial_alignment.hpp"
#include <openvdb/tools/GridTransformer.h>
#include <openvdb/tools/Interpolation.h>
#include <openvdb/math/Mat4.h>
#include <openvdb/math/BBox.h>
#include <string>
#include <memory>
#include <cmath>
#include <vector>
#include <limits>

namespace am_qadf_native {
namespace synchronization {

FloatGridPtr SpatialAlignment::align(
    FloatGridPtr source_grid,
    FloatGridPtr target_grid,
    const std::string& method
) {
    // STEP 1: Transform source grid to reference coordinate system (target_grid's transform)
    // This transforms BOTH:
    //   1. The grid itself (its transform is changed to target_grid's transform)
    //   2. All coordinates/data (all voxels are resampled to the new coordinate system)
    // 
    // OpenVDB's resampleToMatch handles this correctly:
    //   - For each active voxel in source_grid:
    //     * Convert source index → world coordinates (using source transform)
    //     * Convert world coordinates → target index (using target transform)
    //     * Sample source grid and set in aligned_grid
    //   - This moves the complete grid and all its data to the reference coordinate system
    
    // Get target transform (this is the reference coordinate system we want)
    const auto& target_xform = target_grid->transform();
    
    // Create output grid with target's transform (this changes the grid's coordinate system)
    auto aligned_grid = FloatGrid::create(source_grid->background());
    aligned_grid->setTransform(openvdb::math::Transform::Ptr(new openvdb::math::Transform(target_xform)));
    
    // Use OpenVDB's resampleToMatch to resample ALL source grid data into target's coordinate system
    // This transforms all coordinates/voxels to the reference coordinate system
    if (method == "nearest") {
        openvdb::tools::resampleToMatch<openvdb::tools::PointSampler>(*source_grid, *aligned_grid);
    } else if (method == "trilinear") {
        openvdb::tools::resampleToMatch<openvdb::tools::BoxSampler>(*source_grid, *aligned_grid);
    } else {
        // triquadratic
        openvdb::tools::resampleToMatch<openvdb::tools::QuadraticSampler>(*source_grid, *aligned_grid);
    }
    
    // Prune for optimal sparsity
    aligned_grid->tree().prune();
    
    return aligned_grid;
}

FloatGridPtr SpatialAlignment::alignWithTransform(
    FloatGridPtr source_grid,
    FloatGridPtr target_grid,
    const openvdb::Mat4R& transform,
    const std::string& method
) {
    // Create transformer with custom transform
    openvdb::tools::GridTransformer transformer(transform);
    
    // Create output grid
    auto aligned_grid = FloatGrid::create();
    // setTransform requires a Transform::Ptr, create a copy using the Transform copy constructor
    const auto& target_transform = target_grid->transform();
    aligned_grid->setTransform(openvdb::math::Transform::Ptr(new openvdb::math::Transform(target_transform)));
    
    // Transform
    if (method == "nearest") {
        transformer.transformGrid<openvdb::tools::PointSampler, FloatGrid>(
            *source_grid, *aligned_grid);
    } else {
        transformer.transformGrid<openvdb::tools::BoxSampler, FloatGrid>(
            *source_grid, *aligned_grid);
    }
    
    aligned_grid->tree().prune();
    
    return aligned_grid;
}

openvdb::Mat4R SpatialAlignment::computeTransform(
    FloatGridPtr source_grid,
    FloatGridPtr target_grid
) const {
    const auto& source_xform = source_grid->transform();
    const auto& target_xform = target_grid->transform();
    
    // Compute transformation matrix from source index space to target index space
    // target_index = target_xform.inverse() * source_xform * source_index
    return target_xform.baseMap()->getAffineMap()->getMat4().inverse() *
           source_xform.baseMap()->getAffineMap()->getMat4();
}

bool SpatialAlignment::transformsMatch(
    FloatGridPtr grid1,
    FloatGridPtr grid2,
    double tolerance
) const {
    const auto& xform1 = grid1->transform();
    const auto& xform2 = grid2->transform();
    
    // Compare transform matrices
    const auto& mat1 = xform1.baseMap()->getAffineMap()->getMat4();
    const auto& mat2 = xform2.baseMap()->getAffineMap()->getMat4();
    
    // Check if matrices are approximately equal
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            if (std::abs(mat1(i, j) - mat2(i, j)) > tolerance) {
                return false;
            }
        }
    }
    
    return true;
}

std::vector<std::vector<double>> SpatialAlignment::getTransformMatrix(FloatGridPtr grid) const {
    const auto& xform = grid->transform();
    const auto& mat = xform.baseMap()->getAffineMap()->getMat4();
    
    std::vector<std::vector<double>> result(4, std::vector<double>(4));
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            result[i][j] = static_cast<double>(mat(i, j));
        }
    }
    
    return result;
}

std::vector<double> SpatialAlignment::getWorldBoundingBox(FloatGridPtr grid) const {
    // Get index-space bounding box of active voxels
    openvdb::CoordBBox index_bbox;
    if (!grid->tree().evalActiveVoxelBoundingBox(index_bbox)) {
        // Empty grid - return zeros
        return {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    }
    
    // Get transform
    const auto& xform = grid->transform();
    
    // Convert index-space corners to world space
    openvdb::Coord min_coord = index_bbox.min();
    openvdb::Coord max_coord = index_bbox.max();
    
    // Get world coordinates of all 8 corners
    openvdb::Vec3d world_min = xform.indexToWorld(openvdb::Vec3d(min_coord.x(), min_coord.y(), min_coord.z()));
    openvdb::Vec3d world_max = xform.indexToWorld(openvdb::Vec3d(max_coord.x(), max_coord.y(), max_coord.z()));
    
    // Also check other corners to get true bounding box
    openvdb::Vec3d corners[8] = {
        xform.indexToWorld(openvdb::Vec3d(min_coord.x(), min_coord.y(), min_coord.z())),
        xform.indexToWorld(openvdb::Vec3d(max_coord.x(), min_coord.y(), min_coord.z())),
        xform.indexToWorld(openvdb::Vec3d(min_coord.x(), max_coord.y(), min_coord.z())),
        xform.indexToWorld(openvdb::Vec3d(max_coord.x(), max_coord.y(), min_coord.z())),
        xform.indexToWorld(openvdb::Vec3d(min_coord.x(), min_coord.y(), max_coord.z())),
        xform.indexToWorld(openvdb::Vec3d(max_coord.x(), min_coord.y(), max_coord.z())),
        xform.indexToWorld(openvdb::Vec3d(min_coord.x(), max_coord.y(), max_coord.z())),
        xform.indexToWorld(openvdb::Vec3d(max_coord.x(), max_coord.y(), max_coord.z()))
    };
    
    // Find min/max across all corners
    double min_x = std::numeric_limits<double>::max();
    double min_y = std::numeric_limits<double>::max();
    double min_z = std::numeric_limits<double>::max();
    double max_x = std::numeric_limits<double>::lowest();
    double max_y = std::numeric_limits<double>::lowest();
    double max_z = std::numeric_limits<double>::lowest();
    
    for (int i = 0; i < 8; ++i) {
        min_x = std::min(min_x, corners[i].x());
        min_y = std::min(min_y, corners[i].y());
        min_z = std::min(min_z, corners[i].z());
        max_x = std::max(max_x, corners[i].x());
        max_y = std::max(max_y, corners[i].y());
        max_z = std::max(max_z, corners[i].z());
    }
    
    return {min_x, min_y, min_z, max_x, max_y, max_z};
}

FloatGridPtr SpatialAlignment::alignToBoundingBox(
    FloatGridPtr source_grid,
    FloatGridPtr target_grid,
    double unified_min_x, double unified_min_y, double unified_min_z,
    double unified_max_x, double unified_max_y, double unified_max_z,
    const std::string& method
) {
    // Get target transform (this is what we want all grids to have)
    const auto& target_xform = target_grid->transform();
    const auto& source_xform = source_grid->transform();
    
    // Create output grid with background value
    // Transform will be set after computing centered bounding box
    auto aligned_grid = FloatGrid::create(source_grid->background());
    
    // Convert unified bounding box from world space
    openvdb::Vec3d unified_world_min(unified_min_x, unified_min_y, unified_min_z);
    openvdb::Vec3d unified_world_max(unified_max_x, unified_max_y, unified_max_z);
    
    // CENTER-BASED APPROACH: Compute center and center the unified bounding box at origin
    // This ensures all grids appear centered and aligned
    openvdb::Vec3d unified_center = (unified_world_min + unified_world_max) * 0.5;
    
    // Get the voxel size from the target transform
    double voxel_size = target_xform.voxelSize()[0];
    
    // Create a new transform that includes the centering translation
    // The transform maps: index → world_coordinates
    // Formula: world = index * voxel_size + unified_center
    // This centers the grid at unified_center (so index (0,0,0) maps to unified_center)
    openvdb::math::Transform::Ptr centered_transform = 
        openvdb::math::Transform::createLinearTransform(voxel_size);
    centered_transform->postTranslate(unified_center);
    
    // Use the centered transform for the aligned grid
    aligned_grid->setTransform(centered_transform);
    
    // Compute bounding box in index space using the original unified_world_min/max
    // The transform maps: world = index * voxel_size + unified_center
    // So: index = (world - unified_center) / voxel_size
    openvdb::Coord unified_min_coord = centered_transform->worldToIndexNodeCentered(unified_world_min);
    openvdb::Coord unified_max_coord = centered_transform->worldToIndexNodeCentered(unified_world_max);
    
    // Ensure min < max
    openvdb::Coord actual_min(
        std::min(unified_min_coord.x(), unified_max_coord.x()),
        std::min(unified_min_coord.y(), unified_max_coord.y()),
        std::min(unified_min_coord.z(), unified_max_coord.z())
    );
    openvdb::Coord actual_max(
        std::max(unified_min_coord.x(), unified_max_coord.x()),
        std::max(unified_min_coord.y(), unified_max_coord.y()),
        std::max(unified_min_coord.z(), unified_max_coord.z())
    );
    
    openvdb::CoordBBox unified_bbox(actual_min, actual_max);
    
    // Resample the source grid into the unified bounding box
    // Strategy: Iterate over source grid's active voxels and map them to unified bounding box
    // This moves the source data to the unified bounding box region (like moving a car)
    // All grids will occupy the same spatial region and overlap
    
    // Map source grid's active voxels to unified bounding box
    // This is the core "moving" operation - we take each source voxel and place it
    // at the corresponding position in the unified bounding box
    for (auto iter = source_grid->tree().cbeginValueOn(); iter; ++iter) {
        openvdb::Coord source_coord = iter.getCoord();
        
        // Convert source index to world coordinates (original position)
        openvdb::Vec3d source_world_pos = source_xform.indexToWorld(source_coord);
        
        // Convert world coordinates to target index space (unified bounding box)
        // This maps the source data to the unified bounding box region
        openvdb::Coord target_coord = centered_transform->worldToIndexNodeCentered(source_world_pos);
        
        // The unified bounding box should encompass all source data, so this should always pass
        // But we check anyway to be safe
        if (unified_bbox.isInside(target_coord)) {
            // Get the value from source grid
            float value = iter.getValue();
            
            // Set in aligned grid at the target coordinate
            // This moves the data to the unified bounding box region
            aligned_grid->tree().setValue(target_coord, value);
        }
    }
    
    // Also sample the unified bounding box region for interpolation
    // This ensures smooth data and handles any gaps
    // We sample at each position in the unified bounding box to interpolate values
    for (auto iter = unified_bbox.begin(); iter; ++iter) {
        openvdb::Coord target_coord = *iter;
        
        // Skip if already set from active voxel mapping above
        if (aligned_grid->tree().isValueOn(target_coord)) {
            continue;
        }
        
        // Convert target index to world coordinates (using centered transform)
        openvdb::Vec3d target_world_pos = centered_transform->indexToWorld(target_coord);
        
        // Convert world coordinates to source index space (to sample source grid)
        openvdb::Vec3d source_index = source_xform.worldToIndex(target_world_pos);
        
        // Sample source grid at this world position (interpolation)
        float value = source_grid->background();
        if (method == "nearest") {
            value = openvdb::tools::PointSampler::sample(source_grid->tree(), source_index);
        } else if (method == "trilinear") {
            value = openvdb::tools::BoxSampler::sample(source_grid->tree(), source_index);
        } else {
            // triquadratic
            value = openvdb::tools::QuadraticSampler::sample(source_grid->tree(), source_index);
        }
        
        // Only set if value is not background (to maintain sparsity)
        if (value != source_grid->background()) {
            aligned_grid->tree().setValue(target_coord, value);
        }
    }
    
    // Prune for optimal sparsity
    aligned_grid->tree().prune();
    
    return aligned_grid;
}

} // namespace synchronization
} // namespace am_qadf_native
