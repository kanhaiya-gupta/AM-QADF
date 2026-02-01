#include "am_qadf_native/correction/spatial_noise_filtering.hpp"
#include <openvdb/tools/Filter.h>
#include <openvdb/tools/GridOperators.h>
#include <openvdb/math/Coord.h>
#include <openvdb/math/Vec3.h>
#include <algorithm>
#include <cmath>
#include <vector>
#include <string>
#include <memory>

namespace am_qadf_native {
namespace correction {

FloatGridPtr SpatialNoiseFilter::apply(
    FloatGridPtr grid,
    const std::string& method,
    int kernel_size,
    float sigma_spatial,
    float sigma_color
) {
    if (method == "median") {
        return applyMedianFilter(grid, kernel_size);
    } else if (method == "bilateral") {
        return applyBilateralFilter(grid, sigma_spatial, sigma_color);
    } else if (method == "gaussian") {
        return applyGaussianFilter(grid, sigma_spatial);
    }
    
    return grid;
}

FloatGridPtr SpatialNoiseFilter::applyMedianFilter(
    FloatGridPtr grid,
    int kernel_size
) {
    auto filtered_grid = FloatGrid::create();
    const auto& source_transform = grid->transform();
    filtered_grid->setTransform(openvdb::math::Transform::Ptr(new openvdb::math::Transform(source_transform)));
    
    int radius = kernel_size / 2;
    
    // For each active voxel in input grid
    for (auto iter = grid->beginValueOn(); iter; ++iter) {
        openvdb::Coord coord = iter.getCoord();
        auto neighborhood = getNeighborhoodValues(grid, coord, radius);
        
        if (!neighborhood.empty()) {
            // Compute median
            std::sort(neighborhood.begin(), neighborhood.end());
            size_t n = neighborhood.size();
            float median = (n % 2 == 0) 
                ? (neighborhood[n/2 - 1] + neighborhood[n/2]) / 2.0f
                : neighborhood[n/2];
            
            filtered_grid->tree().setValue(coord, median);
        }
    }
    
    return filtered_grid;
}

FloatGridPtr SpatialNoiseFilter::applyBilateralFilter(
    FloatGridPtr grid,
    float sigma_spatial,
    float sigma_color
) {
    auto filtered_grid = FloatGrid::create();
    const auto& source_transform = grid->transform();
    filtered_grid->setTransform(openvdb::math::Transform::Ptr(new openvdb::math::Transform(source_transform)));
    
    int radius = static_cast<int>(3.0f * sigma_spatial);
    
    // For each active voxel
    for (auto iter = grid->beginValueOn(); iter; ++iter) {
        openvdb::Coord coord = iter.getCoord();
        float center_value = *iter;
        
        float weighted_sum = 0.0f;
        float weight_sum = 0.0f;
        
        // Compute bilateral weighted average
        for (int dx = -radius; dx <= radius; ++dx) {
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dz = -radius; dz <= radius; ++dz) {
                    openvdb::Coord neighbor(coord.x() + dx, coord.y() + dy, coord.z() + dz);
                    if (grid->tree().isValueOn(neighbor)) {
                        float neighbor_value = grid->tree().getValue(neighbor);
                        
                        // Spatial weight (Gaussian)
                        float spatial_dist = std::sqrt(dx*dx + dy*dy + dz*dz);
                        float spatial_weight = std::exp(-0.5f * spatial_dist * spatial_dist / (sigma_spatial * sigma_spatial));
                        
                        // Color weight (Gaussian)
                        float color_diff = neighbor_value - center_value;
                        float color_weight = std::exp(-0.5f * color_diff * color_diff / (sigma_color * sigma_color));
                        
                        float weight = spatial_weight * color_weight;
                        weighted_sum += neighbor_value * weight;
                        weight_sum += weight;
                    }
                }
            }
        }
        
        if (weight_sum > 0.0f) {
            filtered_grid->tree().setValue(coord, weighted_sum / weight_sum);
        }
    }
    
    return filtered_grid;
}

FloatGridPtr SpatialNoiseFilter::applyGaussianFilter(
    FloatGridPtr grid,
    float sigma
) {
    auto filtered_grid = FloatGrid::create();
    const auto& source_transform = grid->transform();
    filtered_grid->setTransform(openvdb::math::Transform::Ptr(new openvdb::math::Transform(source_transform)));
    
    int radius = static_cast<int>(3.0f * sigma);
    
    // For each active voxel
    for (auto iter = grid->beginValueOn(); iter; ++iter) {
        openvdb::Coord coord = iter.getCoord();
        
        float weighted_sum = 0.0f;
        float weight_sum = 0.0f;
        
        // Compute Gaussian weighted average
        for (int dx = -radius; dx <= radius; ++dx) {
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dz = -radius; dz <= radius; ++dz) {
                    openvdb::Coord neighbor(coord.x() + dx, coord.y() + dy, coord.z() + dz);
                    if (grid->tree().isValueOn(neighbor)) {
                        float neighbor_value = grid->tree().getValue(neighbor);
                        
                        // Gaussian weight
                        float dist = std::sqrt(dx*dx + dy*dy + dz*dz);
                        float weight = std::exp(-0.5f * dist * dist / (sigma * sigma));
                        
                        weighted_sum += neighbor_value * weight;
                        weight_sum += weight;
                    }
                }
            }
        }
        
        if (weight_sum > 0.0f) {
            filtered_grid->tree().setValue(coord, weighted_sum / weight_sum);
        }
    }
    
    return filtered_grid;
}

std::vector<float> SpatialNoiseFilter::getNeighborhoodValues(
    FloatGridPtr grid,
    const openvdb::Coord& coord,
    int radius
) const {
    std::vector<float> values;
    
    // Get values from neighborhood
    for (int dx = -radius; dx <= radius; ++dx) {
        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dz = -radius; dz <= radius; ++dz) {
                openvdb::Coord neighbor(coord.x() + dx, coord.y() + dy, coord.z() + dz);
                if (grid->tree().isValueOn(neighbor)) {
                    values.push_back(grid->tree().getValue(neighbor));
                }
            }
        }
    }
    
    return values;
}

} // namespace correction
} // namespace am_qadf_native
