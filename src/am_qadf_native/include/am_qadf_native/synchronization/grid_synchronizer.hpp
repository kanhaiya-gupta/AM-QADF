#ifndef AM_QADF_NATIVE_SYNCHRONIZATION_GRID_SYNCHRONIZER_HPP
#define AM_QADF_NATIVE_SYNCHRONIZATION_GRID_SYNCHRONIZER_HPP

#include "grid_spatial_alignment.hpp"
#include "grid_temporal_alignment.hpp"
#include <openvdb/openvdb.h>
#include <openvdb/Grid.h>
#include <vector>
#include <memory>

namespace am_qadf_native {
namespace synchronization {

using FloatGrid = openvdb::FloatGrid;
using FloatGridPtr = FloatGrid::Ptr;

// Combined spatial + temporal synchronization
class GridSynchronizer {
private:
    SpatialAlignment spatial_aligner_;
    TemporalAlignment temporal_aligner_;
    
public:
    // Combined spatial + temporal synchronization
    std::vector<FloatGridPtr> synchronize(
        const std::vector<FloatGridPtr>& source_grids,
        FloatGridPtr reference_grid,  // Build platform grid (spatial reference)
        const std::vector<std::vector<float>>& timestamps,
        const std::vector<std::vector<int>>& layer_indices,
        float temporal_window = 0.1,
        int layer_tolerance = 1
    );
    
    // Spatial-only synchronization
    std::vector<FloatGridPtr> synchronizeSpatial(
        const std::vector<FloatGridPtr>& source_grids,
        FloatGridPtr reference_grid,
        const std::string& method = "trilinear"
    );
    
    // Temporal-only synchronization
    std::vector<FloatGridPtr> synchronizeTemporal(
        const std::vector<FloatGridPtr>& grids,
        const std::vector<std::vector<float>>& timestamps,
        const std::vector<std::vector<int>>& layer_indices,
        float temporal_window = 0.1,
        int layer_tolerance = 1
    );
};

} // namespace synchronization
} // namespace am_qadf_native

#endif // AM_QADF_NATIVE_SYNCHRONIZATION_GRID_SYNCHRONIZER_HPP
