#include "am_qadf_native/signal_mapping/idw_interpolation.hpp"
#include "am_qadf_native/signal_mapping/interpolation_base.hpp"
#include <openvdb/openvdb.h>
#include <openvdb/math/Coord.h>
#include <openvdb/math/Vec3.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <utility>
#include <limits>

namespace am_qadf_native {
namespace signal_mapping {

IDWMapper::IDWMapper(float power, int k_neighbors)
    : power_(power), k_neighbors_(k_neighbors) {
    if (power_ <= 0.0f) {
        throw std::invalid_argument("IDW power must be positive");
    }
    if (k_neighbors_ <= 0) {
        throw std::invalid_argument("Number of neighbors must be positive");
    }
}

void IDWMapper::setPower(float power) {
    if (power <= 0.0f) {
        throw std::invalid_argument("IDW power must be positive");
    }
    power_ = power;
}

void IDWMapper::setKNeighbors(int k) {
    if (k <= 0) {
        throw std::invalid_argument("Number of neighbors must be positive");
    }
    k_neighbors_ = k;
}

void IDWMapper::map(
    FloatGridPtr grid,
    const std::vector<Point>& points,
    const std::vector<float>& values
) {
    if (points.size() != values.size()) {
        throw std::invalid_argument("Points and values size mismatch");
    }
    if (points.empty()) {
        return;  // No points to interpolate
    }
    
    // Get grid bounding box to determine voxel range
    openvdb::CoordBBox bbox = grid->evalActiveVoxelBoundingBox();
    if (bbox.empty()) {
        // If no active voxels, compute bounding box from points
        float min_x = points[0].x, max_x = points[0].x;
        float min_y = points[0].y, max_y = points[0].y;
        float min_z = points[0].z, max_z = points[0].z;
        
        for (const auto& pt : points) {
            min_x = std::min(min_x, pt.x);
            max_x = std::max(max_x, pt.x);
            min_y = std::min(min_y, pt.y);
            max_y = std::max(max_y, pt.y);
            min_z = std::min(min_z, pt.z);
            max_z = std::max(max_z, pt.z);
        }
        
        // Expand bbox slightly
        // voxelSize() returns Vec3<double>, extract x component (typically uniform)
        float margin = static_cast<float>(grid->voxelSize().x()) * 2.0f;
        min_x -= margin; max_x += margin;
        min_y -= margin; max_y += margin;
        min_z -= margin; max_z += margin;
        
        // Use worldToIndex (float) then floor/ceil so voxel (0,0,0) is included for point (0.5,0.5,0.5)
        openvdb::Vec3R min_idx = grid->transform().worldToIndex(openvdb::Vec3R(min_x, min_y, min_z));
        openvdb::Vec3R max_idx = grid->transform().worldToIndex(openvdb::Vec3R(max_x, max_y, max_z));
        bbox = openvdb::CoordBBox(
            openvdb::Coord(
                static_cast<int>(std::floor(min_idx.x())),
                static_cast<int>(std::floor(min_idx.y())),
                static_cast<int>(std::floor(min_idx.z()))
            ),
            openvdb::Coord(
                static_cast<int>(std::ceil(max_idx.x())),
                static_cast<int>(std::ceil(max_idx.y())),
                static_cast<int>(std::ceil(max_idx.z()))
            )
        );
    }
    
    // Helper function to compute squared distance
    auto dist_sq = [](const openvdb::Vec3R& a, const openvdb::Vec3R& b) -> double {
        openvdb::Vec3R diff = a - b;
        return diff.x() * diff.x() + diff.y() * diff.y() + diff.z() * diff.z();
    };
    
    // Iterate over all voxels in bounding box
    auto& tree = grid->tree();
    const auto& transform = grid->transform();
    
    for (auto iter = bbox.begin(); iter; ++iter) {
        openvdb::Coord voxel_coord = *iter;
        
        // Convert voxel coordinate to world coordinate
        openvdb::Vec3R world_pos = transform.indexToWorld(voxel_coord);
        
        // Find k nearest points
        std::vector<std::pair<double, size_t>> distances;  // (distance^2, point_index)
        distances.reserve(points.size());
        
        for (size_t i = 0; i < points.size(); ++i) {
            openvdb::Vec3R pt_pos(points[i].x, points[i].y, points[i].z);
            double d_sq = dist_sq(world_pos, pt_pos);
            distances.emplace_back(d_sq, i);
        }
        
        // Sort by distance and take k nearest
        std::partial_sort(
            distances.begin(),
            distances.begin() + std::min(static_cast<size_t>(k_neighbors_), distances.size()),
            distances.end(),
            [](const std::pair<double, size_t>& a, const std::pair<double, size_t>& b) {
                return a.first < b.first;
            }
        );
        
        // Compute IDW weighted average
        double weight_sum = 0.0;
        double weighted_sum = 0.0;
        bool exact_match = false;
        const size_t k_actual = std::min(static_cast<size_t>(k_neighbors_), distances.size());
        
        for (size_t i = 0; i < k_actual; ++i) {
            double dist_sq_val = distances[i].first;
            size_t pt_idx = distances[i].second;
            
            // Avoid division by zero for exact matches
            if (dist_sq_val < std::numeric_limits<double>::epsilon()) {
                tree.setValue(voxel_coord, values[pt_idx]);
                exact_match = true;
                break;
            }
            
            double dist = std::sqrt(dist_sq_val);
            double weight = 1.0 / std::pow(dist, static_cast<double>(power_));
            weight_sum += weight;
            weighted_sum += weight * static_cast<double>(values[pt_idx]);
        }
        
        // Set interpolated value only when we did not already set via exact match
        if (!exact_match && weight_sum > 0.0) {
            float interpolated_value = static_cast<float>(weighted_sum / weight_sum);
            tree.setValue(voxel_coord, interpolated_value);
        }
    }
}

} // namespace signal_mapping
} // namespace am_qadf_native
