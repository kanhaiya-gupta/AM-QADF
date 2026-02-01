#include "am_qadf_native/synchronization/grid_synchronizer.hpp"
#include <vector>
#include <memory>
#include <string>

namespace am_qadf_native {
namespace synchronization {

std::vector<FloatGridPtr> GridSynchronizer::synchronize(
    const std::vector<FloatGridPtr>& source_grids,
    FloatGridPtr reference_grid,
    const std::vector<std::vector<float>>& timestamps,
    const std::vector<std::vector<int>>& layer_indices,
    float temporal_window,
    int layer_tolerance
) {
    // Step 1: Spatial alignment (all grids to reference coordinate system)
    std::vector<FloatGridPtr> spatially_aligned;
    for (auto grid : source_grids) {
        auto aligned = spatial_aligner_.align(grid, reference_grid, "trilinear");
        spatially_aligned.push_back(aligned);
    }
    
    // Step 2: Temporal alignment
    std::vector<GridWithMetadata> grids_with_metadata;
    for (size_t i = 0; i < spatially_aligned.size(); ++i) {
        GridWithMetadata g;
        g.grid = spatially_aligned[i];
        if (i < timestamps.size()) {
            g.timestamps = timestamps[i];
        }
        if (i < layer_indices.size()) {
            g.layer_indices = layer_indices[i];
        }
        grids_with_metadata.push_back(g);
    }
    
    auto synchronized = temporal_aligner_.synchronizeTemporal(
        grids_with_metadata,
        temporal_window,
        layer_tolerance
    );
    
    return synchronized;
}

std::vector<FloatGridPtr> GridSynchronizer::synchronizeSpatial(
    const std::vector<FloatGridPtr>& source_grids,
    FloatGridPtr reference_grid,
    const std::string& method
) {
    std::vector<FloatGridPtr> aligned;
    for (auto grid : source_grids) {
        aligned.push_back(spatial_aligner_.align(grid, reference_grid, method));
    }
    return aligned;
}

std::vector<FloatGridPtr> GridSynchronizer::synchronizeTemporal(
    const std::vector<FloatGridPtr>& grids,
    const std::vector<std::vector<float>>& timestamps,
    const std::vector<std::vector<int>>& layer_indices,
    float temporal_window,
    int layer_tolerance
) {
    std::vector<GridWithMetadata> grids_with_metadata;
    for (size_t i = 0; i < grids.size(); ++i) {
        GridWithMetadata g;
        g.grid = grids[i];
        if (i < timestamps.size()) {
            g.timestamps = timestamps[i];
        }
        if (i < layer_indices.size()) {
            g.layer_indices = layer_indices[i];
        }
        grids_with_metadata.push_back(g);
    }
    
    return temporal_aligner_.synchronizeTemporal(
        grids_with_metadata,
        temporal_window,
        layer_tolerance
    );
}

} // namespace synchronization
} // namespace am_qadf_native
