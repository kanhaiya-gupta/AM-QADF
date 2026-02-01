#include "am_qadf_native/fusion/grid_fusion.hpp"
#include <openvdb/openvdb.h>
#include <openvdb/tools/GridOperators.h>
#include <openvdb/math/Coord.h>
#include <algorithm>
#include <numeric>
#include <vector>
#include <string>
#include <set>
#include <stdexcept>

namespace am_qadf_native {
namespace fusion {

FloatGridPtr GridFusion::fuse(
    const std::vector<FloatGridPtr>& grids,
    const std::string& strategy
) {
    if (grids.empty()) {
        return FloatGrid::create();
    }
    
    // Create output grid (use first grid's transform as reference)
    auto fused_grid = FloatGrid::create();
    const auto& source_transform = grids[0]->transform();
    fused_grid->setTransform(openvdb::math::Transform::Ptr(new openvdb::math::Transform(source_transform)));
    
    if (strategy == "weighted_average") {
        fuseWeightedAverage(grids, fused_grid);
    } else if (strategy == "max") {
        fuseMax(grids, fused_grid);
    } else if (strategy == "min") {
        fuseMin(grids, fused_grid);
    } else if (strategy == "median") {
        fuseMedian(grids, fused_grid);
    }
    
    return fused_grid;
}

FloatGridPtr GridFusion::fuseWeighted(
    const std::vector<FloatGridPtr>& grids,
    const std::vector<float>& weights
) {
    if (grids.size() != weights.size()) {
        throw std::invalid_argument("Grids and weights size mismatch");
    }
    
    if (grids.empty()) {
        return FloatGrid::create();
    }
    
    // Normalize weights
    float sum_weights = std::accumulate(weights.begin(), weights.end(), 0.0f);
    if (sum_weights <= 0.0f) {
        throw std::invalid_argument("Sum of weights must be positive");
    }
    
    std::vector<float> normalized_weights = weights;
    for (auto& w : normalized_weights) {
        w /= sum_weights;
    }
    
    // Create output grid
    auto fused_grid = FloatGrid::create();
    const auto& source_transform = grids[0]->transform();
    fused_grid->setTransform(openvdb::math::Transform::Ptr(new openvdb::math::Transform(source_transform)));
    
    // Collect all active coordinates
    std::set<openvdb::Coord> all_coords;
    for (auto grid : grids) {
        for (auto iter = grid->beginValueOn(); iter; ++iter) {
            all_coords.insert(iter.getCoord());
        }
    }
    
    // For each coordinate, compute weighted average
    auto& output_tree = fused_grid->tree();
    for (const auto& coord : all_coords) {
        float weighted_sum = 0.0f;
        float weight_sum = 0.0f;
        
        for (size_t i = 0; i < grids.size(); ++i) {
            if (grids[i]->tree().isValueOn(coord)) {
                float value = grids[i]->tree().getValue(coord);
                float weight = normalized_weights[i];
                weighted_sum += value * weight;
                weight_sum += weight;
            }
        }
        
        if (weight_sum > 0.0f) {
            // Normalize by actual weight sum (handles cases where not all grids have value at coord)
            float normalized_value = weighted_sum / weight_sum;
            output_tree.setValue(coord, normalized_value);
        }
    }
    
    return fused_grid;
}

void GridFusion::fuseWeightedAverage(
    const std::vector<FloatGridPtr>& grids,
    FloatGridPtr output
) {
    if (grids.empty()) return;
    
    // Use first grid's structure as base
    const auto& source_transform = grids[0]->transform();
    output->setTransform(openvdb::math::Transform::Ptr(new openvdb::math::Transform(source_transform)));
    
    // Collect all active coordinates
    std::set<openvdb::Coord> all_coords;
    for (auto grid : grids) {
        for (auto iter = grid->beginValueOn(); iter; ++iter) {
            all_coords.insert(iter.getCoord());
        }
    }
    
    // For each coordinate, average only over grids that have a value at that coord
    for (const auto& coord : all_coords) {
        float sum = 0.0f;
        int count = 0;
        
        for (auto grid : grids) {
            if (grid->tree().isValueOn(coord)) {
                sum += grid->tree().getValue(coord);
                count++;
            }
        }
        
        if (count > 0) {
            output->tree().setValue(coord, sum / static_cast<float>(count));
        }
    }
}

void GridFusion::fuseMax(
    const std::vector<FloatGridPtr>& grids,
    FloatGridPtr output
) {
    if (grids.empty()) return;
    
    const auto& source_transform = grids[0]->transform();
    output->setTransform(openvdb::math::Transform::Ptr(new openvdb::math::Transform(source_transform)));
    
    std::set<openvdb::Coord> all_coords;
    for (auto grid : grids) {
        for (auto iter = grid->beginValueOn(); iter; ++iter) {
            all_coords.insert(iter.getCoord());
        }
    }
    
    for (const auto& coord : all_coords) {
        float max_val = std::numeric_limits<float>::lowest();
        bool has_value = false;
        
        for (auto grid : grids) {
            if (grid->tree().isValueOn(coord)) {
                max_val = std::max(max_val, grid->tree().getValue(coord));
                has_value = true;
            }
        }
        
        if (has_value) {
            output->tree().setValue(coord, max_val);
        }
    }
}

void GridFusion::fuseMin(
    const std::vector<FloatGridPtr>& grids,
    FloatGridPtr output
) {
    if (grids.empty()) return;
    
    const auto& source_transform = grids[0]->transform();
    output->setTransform(openvdb::math::Transform::Ptr(new openvdb::math::Transform(source_transform)));
    
    std::set<openvdb::Coord> all_coords;
    for (auto grid : grids) {
        for (auto iter = grid->beginValueOn(); iter; ++iter) {
            all_coords.insert(iter.getCoord());
        }
    }
    
    for (const auto& coord : all_coords) {
        float min_val = std::numeric_limits<float>::max();
        bool has_value = false;
        
        for (auto grid : grids) {
            if (grid->tree().isValueOn(coord)) {
                min_val = std::min(min_val, grid->tree().getValue(coord));
                has_value = true;
            }
        }
        
        if (has_value) {
            output->tree().setValue(coord, min_val);
        }
    }
}

void GridFusion::fuseMedian(
    const std::vector<FloatGridPtr>& grids,
    FloatGridPtr output
) {
    if (grids.empty()) return;
    
    const auto& source_transform = grids[0]->transform();
    output->setTransform(openvdb::math::Transform::Ptr(new openvdb::math::Transform(source_transform)));
    
    std::set<openvdb::Coord> all_coords;
    for (auto grid : grids) {
        for (auto iter = grid->beginValueOn(); iter; ++iter) {
            all_coords.insert(iter.getCoord());
        }
    }
    
    for (const auto& coord : all_coords) {
        std::vector<float> values;
        
        for (auto grid : grids) {
            if (grid->tree().isValueOn(coord)) {
                values.push_back(grid->tree().getValue(coord));
            }
        }
        
        if (!values.empty()) {
            std::sort(values.begin(), values.end());
            size_t n = values.size();
            float median = (n % 2 == 0)
                ? (values[n/2 - 1] + values[n/2]) / 2.0f
                : values[n/2];
            output->tree().setValue(coord, median);
        }
    }
}

} // namespace fusion
} // namespace am_qadf_native
