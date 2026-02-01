#include "am_qadf_native/signal_mapping/kde_interpolation.hpp"
#include "am_qadf_native/signal_mapping/interpolation_base.hpp"
#include <openvdb/openvdb.h>
#include <openvdb/math/Coord.h>
#include <openvdb/math/Vec3.h>
#include <cmath>
#include <vector>
#include <string>
#include <stdexcept>

namespace am_qadf_native {
namespace signal_mapping {

KDEMapper::KDEMapper(float bandwidth, const std::string& kernel_type)
    : bandwidth_(bandwidth), kernel_type_(kernel_type) {
}

void KDEMapper::setBandwidth(float bandwidth) {
    bandwidth_ = bandwidth;
}

void KDEMapper::setKernelType(const std::string& kernel_type) {
    kernel_type_ = kernel_type;
}

float KDEMapper::kernelFunction(float distance) const {
    if (kernel_type_ == "gaussian") {
        return std::exp(-0.5f * (distance * distance) / (bandwidth_ * bandwidth_));
    } else if (kernel_type_ == "epanechnikov") {
        float u = distance / bandwidth_;
        if (std::abs(u) <= 1.0f) {
            return 0.75f * (1.0f - u * u);
        }
        return 0.0f;
    }
    // Default: Gaussian
    return std::exp(-0.5f * (distance * distance) / (bandwidth_ * bandwidth_));
}

void KDEMapper::map(
    FloatGridPtr grid,
    const std::vector<Point>& points,
    const std::vector<float>& values
) {
    if (points.size() != values.size()) {
        throw std::invalid_argument("Points and values size mismatch");
    }
    
    // Get bounding box of points to determine grid region
    openvdb::CoordBBox bbox;
    if (grid->tree().evalActiveVoxelBoundingBox(bbox)) {
        // For each voxel in bounding box, compute KDE weighted average
        for (auto iter = bbox.begin(); iter; ++iter) {
            openvdb::Coord coord = *iter;
            auto world_coord = grid->transform().indexToWorld(coord);
            
            float weighted_sum = 0.0f;
            float weight_sum = 0.0f;
            
            // Compute KDE weighted average from all points
            for (size_t i = 0; i < points.size(); ++i) {
                float dx = world_coord.x() - points[i].x;
                float dy = world_coord.y() - points[i].y;
                float dz = world_coord.z() - points[i].z;
                float distance = std::sqrt(dx*dx + dy*dy + dz*dz);
                
                float weight = kernelFunction(distance);
                weighted_sum += values[i] * weight;
                weight_sum += weight;
            }
            
            if (weight_sum > 0.0f) {
                grid->tree().setValue(coord, weighted_sum / weight_sum);
            }
        }
    }
}

} // namespace signal_mapping
} // namespace am_qadf_native
