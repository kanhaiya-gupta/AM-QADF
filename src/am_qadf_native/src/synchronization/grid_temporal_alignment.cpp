#include "am_qadf_native/synchronization/grid_temporal_alignment.hpp"
#include <openvdb/openvdb.h>
#include <openvdb/math/Coord.h>
#include <algorithm>
#include <limits>
#include <vector>
#include <set>
#include <memory>
#include <cmath>

namespace am_qadf_native {
namespace synchronization {

std::vector<FloatGridPtr> TemporalAlignment::synchronizeTemporal(
    const std::vector<GridWithMetadata>& grids,
    float temporal_window,
    int layer_tolerance
) {
    // Create synchronized time bins
    auto time_bins = createTimeBins(grids, temporal_window);
    
    // For each time bin, create synchronized grid
    std::vector<FloatGridPtr> synchronized_grids;
    for (const auto& bin : time_bins) {
        auto synced_grid = aggregateGridsInBin(bin, layer_tolerance);
        synchronized_grids.push_back(synced_grid);
    }
    
    return synchronized_grids;
}

std::vector<TemporalAlignment::TimeBin> TemporalAlignment::createTimeBins(
    const std::vector<GridWithMetadata>& grids,
    float window
) const {
    // Find time range across all grids
    float min_time = std::numeric_limits<float>::max();
    float max_time = std::numeric_limits<float>::lowest();
    
    for (const auto& g : grids) {
        if (!g.timestamps.empty()) {
            min_time = std::min(min_time, *std::min_element(g.timestamps.begin(), g.timestamps.end()));
            max_time = std::max(max_time, *std::max_element(g.timestamps.begin(), g.timestamps.end()));
        }
    }
    
    // Create time bins
    std::vector<TimeBin> bins;
    for (float t = min_time; t <= max_time; t += window) {
        TimeBin bin;
        bin.time_start = t;
        bin.time_end = t + window;
        
        // Find grids with data in this time window
        for (const auto& g : grids) {
            bool in_window = false;
            for (float ts : g.timestamps) {
                if (ts >= bin.time_start && ts < bin.time_end) {
                    in_window = true;
                    break;
                }
            }
            if (in_window) {
                bin.grids_in_bin.push_back(g);
            }
        }
        
        if (!bin.grids_in_bin.empty()) {
            bins.push_back(bin);
        }
    }
    
    return bins;
}

FloatGridPtr TemporalAlignment::aggregateGridsInBin(
    const TimeBin& bin,
    int /*layer_tolerance*/
) const {
    if (bin.grids_in_bin.empty()) {
        return FloatGrid::create();
    }
    
    // Create output grid (use first grid's transform as reference)
    auto output_grid = FloatGrid::create();
    const auto& source_transform = bin.grids_in_bin[0].grid->transform();
    output_grid->setTransform(openvdb::math::Transform::Ptr(new openvdb::math::Transform(source_transform)));
    
    // Collect all active coordinates
    std::set<openvdb::Coord> all_coords;
    for (const auto& g : bin.grids_in_bin) {
        for (auto iter = g.grid->beginValueOn(); iter; ++iter) {
            all_coords.insert(iter.getCoord());
        }
    }
    
    // Aggregate values using weighted average based on temporal proximity
    for (const auto& coord : all_coords) {
        float weighted_sum = 0.0f;
        float weight_sum = 0.0f;
        
        for (const auto& g : bin.grids_in_bin) {
            if (g.grid->tree().isValueOn(coord)) {
                float value = g.grid->tree().getValue(coord);
                
                // Compute temporal weight (closer to bin center = higher weight)
                float bin_center = (bin.time_start + bin.time_end) / 2.0f;
                float min_time_dist = std::numeric_limits<float>::max();
                
                for (float ts : g.timestamps) {
                    float time_dist = std::abs(ts - bin_center);
                    min_time_dist = std::min(min_time_dist, time_dist);
                }
                
                float time_weight = std::exp(-min_time_dist);
                weighted_sum += value * time_weight;
                weight_sum += time_weight;
            }
        }
        
        if (weight_sum > 0.0f) {
            output_grid->tree().setValue(coord, weighted_sum / weight_sum);
        }
    }
    
    return output_grid;
}

} // namespace synchronization
} // namespace am_qadf_native
