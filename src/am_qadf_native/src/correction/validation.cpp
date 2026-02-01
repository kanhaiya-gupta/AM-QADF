#include "am_qadf_native/correction/validation.hpp"
#include <openvdb/openvdb.h>
#include <openvdb/tools/GridOperators.h>
#include <openvdb/tools/Statistics.h>
#include <openvdb/math/Coord.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>
#include <array>
#include <map>
#include <string>

namespace am_qadf_native {
namespace correction {

ValidationResult Validation::validateGrid(FloatGridPtr grid) {
    ValidationResult result;
    result.is_valid = true;
    
    // Check for invalid values
    if (hasInvalidValues(grid)) {
        result.is_valid = false;
        result.errors.push_back("Grid contains NaN or Inf values");
    }
    
    // Check bounds
    if (!checkBounds(grid)) {
        result.is_valid = false;
        result.errors.push_back("Grid bounds are invalid");
    }
    
    // Compute quality metrics
    float min_val, max_val, mean_val, std_val;
    computeStatistics(grid, min_val, max_val, mean_val, std_val);
    result.metrics["min"] = min_val;
    result.metrics["max"] = max_val;
    result.metrics["mean"] = mean_val;
    result.metrics["std"] = std_val;
    
    return result;
}

ValidationResult Validation::validateSignalData(
    const std::vector<float>& values,
    float min_value,
    float max_value
) {
    ValidationResult result;
    result.is_valid = true;
    
    for (float v : values) {
        if (std::isnan(v) || std::isinf(v)) {
            result.is_valid = false;
            result.errors.push_back("Signal contains NaN or Inf values");
            break;
        }
        
        if (v < min_value || v > max_value) {
            result.warnings.push_back("Signal value out of expected range");
        }
    }
    
    return result;
}

ValidationResult Validation::validateCoordinates(
    const std::vector<std::array<float, 3>>& points,
    const std::array<float, 3>& bbox_min,
    const std::array<float, 3>& bbox_max
) {
    ValidationResult result;
    result.is_valid = true;
    
    for (const auto& point : points) {
        if (point[0] < bbox_min[0] || point[0] > bbox_max[0] ||
            point[1] < bbox_min[1] || point[1] > bbox_max[1] ||
            point[2] < bbox_min[2] || point[2] > bbox_max[2]) {
            result.warnings.push_back("Point outside bounding box");
        }
    }
    
    return result;
}

bool Validation::checkConsistency(
    FloatGridPtr grid1,
    FloatGridPtr grid2,
    float tolerance
) {
    if (!grid1 || !grid2) {
        return false;
    }
    
    // Check transforms (voxel size)
    // voxelSize() returns Vec3<double>, extract x component (typically uniform)
    float voxel_size1 = static_cast<float>(grid1->voxelSize().x());
    float voxel_size2 = static_cast<float>(grid2->voxelSize().x());
    if (std::abs(voxel_size1 - voxel_size2) > tolerance) {
        return false;  // Different voxel sizes
    }
    
    // Check bounding boxes (if both have active voxels)
    openvdb::CoordBBox bbox1, bbox2;
    bool has_bbox1 = grid1->tree().evalActiveVoxelBoundingBox(bbox1);
    bool has_bbox2 = grid2->tree().evalActiveVoxelBoundingBox(bbox2);
    
    if (has_bbox1 && has_bbox2) {
        // Check if bounding boxes are similar
        openvdb::Vec3d size1 = bbox1.max().asVec3d() - bbox1.min().asVec3d();
        openvdb::Vec3d size2 = bbox2.max().asVec3d() - bbox2.min().asVec3d();
        
        if (std::abs(size1.x() - size2.x()) > tolerance ||
            std::abs(size1.y() - size2.y()) > tolerance ||
            std::abs(size1.z() - size2.z()) > tolerance) {
            return false;  // Different dimensions
        }
    }
    
    // Check values at common coordinates
    int common_voxels = 0;
    int matching_voxels = 0;
    
    for (auto iter = grid1->beginValueOn(); iter; ++iter) {
        openvdb::Coord coord = iter.getCoord();
        if (grid2->tree().isValueOn(coord)) {
            common_voxels++;
            float val1 = *iter;
            float val2 = grid2->tree().getValue(coord);
            if (std::abs(val1 - val2) <= tolerance) {
                matching_voxels++;
            }
        }
    }
    
    // Consider consistent if most common voxels match
    if (common_voxels > 0) {
        float match_ratio = static_cast<float>(matching_voxels) / common_voxels;
        return match_ratio >= 0.9f;  // 90% match threshold
    }
    
    return true;  // No common voxels - considered consistent
}

bool Validation::hasInvalidValues(FloatGridPtr grid) const {
    if (!grid) {
        return false;
    }
    
    // Iterate over all active voxels and check for NaN or Inf
    for (auto iter = grid->beginValueOn(); iter; ++iter) {
        float val = *iter;
        if (std::isnan(val) || std::isinf(val)) {
            return true;
        }
    }
    
    return false;
}

bool Validation::checkBounds(FloatGridPtr grid) const {
    // Check grid bounds
    // Verify transform exists
    if (!grid || !grid->transform().hasUniformScale()) {
        return false;
    }
    
    // Check if grid has any active voxels
    return grid->activeVoxelCount() > 0;
}

void Validation::computeStatistics(
    FloatGridPtr grid,
    float& min_val,
    float& max_val,
    float& mean_val,
    float& std_val
) const {
    // TODO: Compute statistics using OpenVDB tools
    // Use openvdb::tools::Statistics or iterate manually
    
    min_val = std::numeric_limits<float>::max();
    max_val = std::numeric_limits<float>::lowest();
    double sum = 0.0;
    size_t count = 0;
    
    for (auto iter = grid->beginValueOn(); iter; ++iter) {
        float val = *iter;
        min_val = std::min(min_val, val);
        max_val = std::max(max_val, val);
        sum += val;
        count++;
    }
    
    mean_val = count > 0 ? static_cast<float>(sum / count) : 0.0f;
    
    // Compute standard deviation
    double sum_sq_diff = 0.0;
    for (auto iter = grid->beginValueOn(); iter; ++iter) {
        float diff = *iter - mean_val;
        sum_sq_diff += diff * diff;
    }
    std_val = count > 0 ? static_cast<float>(std::sqrt(sum_sq_diff / count)) : 0.0f;
}

} // namespace correction
} // namespace am_qadf_native
