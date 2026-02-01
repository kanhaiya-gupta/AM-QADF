#include "am_qadf_native/correction/geometric_correction.hpp"
#include <openvdb/tools/GridTransformer.h>
#include <openvdb/math/Mat4.h>
#include <openvdb/math/Coord.h>
#include <openvdb/math/Vec3.h>
#include <cmath>
#include <vector>
#include <array>
#include <string>
#include <memory>
#include <algorithm>
#include <map>

namespace am_qadf_native {
namespace correction {

FloatGridPtr GeometricCorrection::correctDistortions(
    FloatGridPtr grid,
    const DistortionMap& distortion_map
) {
    if (distortion_map.reference_points.empty() || 
        distortion_map.distortion_vectors.size() != distortion_map.reference_points.size()) {
        // Invalid distortion map - return original grid
        return grid;
    }
    
    auto corrected_grid = FloatGrid::create();
    const auto& source_transform = grid->transform();
    corrected_grid->setTransform(openvdb::math::Transform::Ptr(new openvdb::math::Transform(source_transform)));
    
    auto& output_tree = corrected_grid->tree();
    const auto& input_tree = grid->tree();
    
    // Iterate over all active voxels in input grid
    for (auto iter = grid->beginValueOn(); iter; ++iter) {
        openvdb::Coord coord = iter.getCoord();
        float value = iter.getValue();
        
        // Convert voxel coordinate to world coordinate
        openvdb::Vec3R world_pos = source_transform.indexToWorld(coord);
        
        // Apply distortion correction
        std::array<float, 3> point = {
            static_cast<float>(world_pos.x()),
            static_cast<float>(world_pos.y()),
            static_cast<float>(world_pos.z())
        };
        
        std::array<float, 3> corrected_point = applyDistortionCorrection(point, distortion_map);
        
        // Convert corrected world coordinate back to voxel coordinate
        openvdb::Vec3R corrected_world(corrected_point[0], corrected_point[1], corrected_point[2]);
        openvdb::Coord corrected_coord = source_transform.worldToIndexCellCentered(corrected_world);
        
        // Set value at corrected coordinate
        output_tree.setValue(corrected_coord, value);
    }
    
    return corrected_grid;
}

FloatGridPtr GeometricCorrection::correctLensDistortion(
    FloatGridPtr grid,
    const std::vector<float>& distortion_coefficients
) {
    // Lens distortion model: k1, k2, p1, p2, k3
    // Radial distortion: r^2 = x^2 + y^2, correction = (1 + k1*r^2 + k2*r^4 + k3*r^6)
    // Tangential distortion: p1, p2 terms
    
    if (distortion_coefficients.size() < 5) {
        // Not enough coefficients - return original grid
        return grid;
    }
    
    float k1 = distortion_coefficients[0];  // Radial distortion coefficient 1
    float k2 = distortion_coefficients[1];  // Radial distortion coefficient 2
    float p1 = distortion_coefficients[2];  // Tangential distortion coefficient 1
    float p2 = distortion_coefficients[3];  // Tangential distortion coefficient 2
    float k3 = distortion_coefficients.size() > 4 ? distortion_coefficients[4] : 0.0f;  // Radial distortion coefficient 3
    
    auto corrected_grid = FloatGrid::create();
    const auto& source_transform = grid->transform();
    corrected_grid->setTransform(openvdb::math::Transform::Ptr(new openvdb::math::Transform(source_transform)));
    
    auto& output_tree = corrected_grid->tree();
    const auto& input_tree = grid->tree();
    
    // Get principal point (center of distortion) - assume center of grid
    openvdb::CoordBBox bbox = grid->evalActiveVoxelBoundingBox();
    openvdb::Vec3R center_world = source_transform.indexToWorld(
        openvdb::Vec3R(
            (bbox.min().x() + bbox.max().x()) / 2.0,
            (bbox.min().y() + bbox.max().y()) / 2.0,
            (bbox.min().z() + bbox.max().z()) / 2.0
        )
    );
    
    // Iterate over all active voxels
    for (auto iter = grid->beginValueOn(); iter; ++iter) {
        openvdb::Coord coord = iter.getCoord();
        float value = iter.getValue();
        
        // Convert to world coordinates
        openvdb::Vec3R world_pos = source_transform.indexToWorld(coord);
        
        // Compute relative position from center (for 2D distortion, use x-y plane)
        float x = static_cast<float>(world_pos.x() - center_world.x());
        float y = static_cast<float>(world_pos.y() - center_world.y());
        
        // Compute radial distance squared
        float r_sq = x * x + y * y;
        float r_4 = r_sq * r_sq;
        float r_6 = r_4 * r_sq;
        
        // Radial distortion correction factor
        float radial_correction = 1.0f + k1 * r_sq + k2 * r_4 + k3 * r_6;
        
        // Tangential distortion
        float x_tangential = 2.0f * p1 * x * y + p2 * (r_sq + 2.0f * x * x);
        float y_tangential = p1 * (r_sq + 2.0f * y * y) + 2.0f * p2 * x * y;
        
        // Apply corrections
        float x_corrected = x * radial_correction + x_tangential;
        float y_corrected = y * radial_correction + y_tangential;
        
        // Convert back to world coordinates
        openvdb::Vec3R corrected_world(
            center_world.x() + x_corrected,
            center_world.y() + y_corrected,
            world_pos.z()  // Z coordinate unchanged for 2D lens distortion
        );
        
        // Convert to voxel coordinates
        openvdb::Coord corrected_coord = source_transform.worldToIndexCellCentered(corrected_world);
        
        // Set value at corrected coordinate
        output_tree.setValue(corrected_coord, value);
    }
    
    return corrected_grid;
}

FloatGridPtr GeometricCorrection::correctSensorMisalignment(
    FloatGridPtr grid,
    const std::array<float, 3>& translation,
    const std::array<float, 3>& rotation
) {
    // Rotation is in Euler angles (roll, pitch, yaw) in radians
    // Translation is in world coordinates
    
    auto corrected_grid = FloatGrid::create();
    const auto& source_transform = grid->transform();
    corrected_grid->setTransform(openvdb::math::Transform::Ptr(new openvdb::math::Transform(source_transform)));
    
    // Build rotation matrix from Euler angles (ZYX convention)
    float roll = rotation[0];   // Rotation around X-axis
    float pitch = rotation[1];  // Rotation around Y-axis
    float yaw = rotation[2];    // Rotation around Z-axis
    
    double cos_r = std::cos(static_cast<double>(roll));
    double sin_r = std::sin(static_cast<double>(roll));
    double cos_p = std::cos(static_cast<double>(pitch));
    double sin_p = std::sin(static_cast<double>(pitch));
    double cos_y = std::cos(static_cast<double>(yaw));
    double sin_y = std::sin(static_cast<double>(yaw));
    
    // Rotation matrix (ZYX Euler angles)
    // R = R_z(yaw) * R_y(pitch) * R_x(roll)
    openvdb::math::Mat4d rotation_matrix(
        cos_y * cos_p,  cos_y * sin_p * sin_r - sin_y * cos_r,  cos_y * sin_p * cos_r + sin_y * sin_r,  0.0,
        sin_y * cos_p,  sin_y * sin_p * sin_r + cos_y * cos_r,  sin_y * sin_p * cos_r - cos_y * sin_r,  0.0,
        -sin_p,         cos_p * sin_r,                          cos_p * cos_r,                          0.0,
        0.0,            0.0,                                    0.0,                                     1.0
    );
    
    // Translation vector
    openvdb::Vec3d translation_vec(translation[0], translation[1], translation[2]);
    
    auto& output_tree = corrected_grid->tree();
    
    // Iterate over all active voxels
    for (auto iter = grid->beginValueOn(); iter; ++iter) {
        openvdb::Coord coord = iter.getCoord();
        float value = iter.getValue();
        
        // Convert to world coordinates
        openvdb::Vec3R world_pos = source_transform.indexToWorld(coord);
        
        // Apply translation first (subtract misalignment translation to correct)
        openvdb::Vec3d corrected_pos(
            world_pos.x() - translation_vec.x(),
            world_pos.y() - translation_vec.y(),
            world_pos.z() - translation_vec.z()
        );
        
        // Apply inverse rotation (to correct misalignment, we apply inverse rotation)
        // For inverse rotation, we use transpose (since rotation matrix is orthogonal)
        openvdb::Vec3d rotated_pos(
            rotation_matrix(0, 0) * corrected_pos.x() + rotation_matrix(1, 0) * corrected_pos.y() + rotation_matrix(2, 0) * corrected_pos.z(),
            rotation_matrix(0, 1) * corrected_pos.x() + rotation_matrix(1, 1) * corrected_pos.y() + rotation_matrix(2, 1) * corrected_pos.z(),
            rotation_matrix(0, 2) * corrected_pos.x() + rotation_matrix(1, 2) * corrected_pos.y() + rotation_matrix(2, 2) * corrected_pos.z()
        );
        
        // Convert back to voxel coordinates
        openvdb::Vec3R corrected_world(rotated_pos.x(), rotated_pos.y(), rotated_pos.z());
        openvdb::Coord corrected_coord = source_transform.worldToIndexCellCentered(corrected_world);
        
        // Set value at corrected coordinate
        output_tree.setValue(corrected_coord, value);
    }
    
    return corrected_grid;
}

std::array<float, 3> GeometricCorrection::applyDistortionCorrection(
    const std::array<float, 3>& point,
    const DistortionMap& distortion_map
) const {
    if (distortion_map.reference_points.empty()) {
        return point;
    }
    
    // Find nearest reference point using Euclidean distance
    float min_dist_sq = std::numeric_limits<float>::max();
    size_t nearest_idx = 0;
    
    for (size_t i = 0; i < distortion_map.reference_points.size(); ++i) {
        const auto& ref_point = distortion_map.reference_points[i];
        float dx = point[0] - ref_point[0];
        float dy = point[1] - ref_point[1];
        float dz = point[2] - ref_point[2];
        float dist_sq = dx * dx + dy * dy + dz * dz;
        
        if (dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
            nearest_idx = i;
        }
    }
    
    // Apply correction vector from nearest reference point
    if (nearest_idx < distortion_map.distortion_vectors.size()) {
        const auto& correction = distortion_map.distortion_vectors[nearest_idx];
        std::array<float, 3> corrected_point = {
            point[0] + correction[0],
            point[1] + correction[1],
            point[2] + correction[2]
        };
        return corrected_point;
    }
    
    return point;
}

} // namespace correction
} // namespace am_qadf_native
