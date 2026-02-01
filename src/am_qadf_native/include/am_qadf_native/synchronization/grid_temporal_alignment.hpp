#ifndef AM_QADF_NATIVE_SYNCHRONIZATION_GRID_TEMPORAL_ALIGNMENT_HPP
#define AM_QADF_NATIVE_SYNCHRONIZATION_GRID_TEMPORAL_ALIGNMENT_HPP

#include <openvdb/openvdb.h>
#include <openvdb/Grid.h>
#include <vector>
#include <memory>

namespace am_qadf_native {
namespace synchronization {

using FloatGrid = openvdb::FloatGrid;
using FloatGridPtr = FloatGrid::Ptr;

// Grid with temporal metadata
struct GridWithMetadata {
    FloatGridPtr grid;
    std::vector<float> timestamps;
    std::vector<int> layer_indices;
};

// Temporal alignment based on timestamps and layer indices
class TemporalAlignment {
public:
    // Align grids based on timestamps and layer indices
    std::vector<FloatGridPtr> synchronizeTemporal(
        const std::vector<GridWithMetadata>& grids,
        float temporal_window = 0.1,  // Time window for alignment (seconds)
        int layer_tolerance = 1        // Layer tolerance (±layers)
    );
    
private:
    // Time bin structure
    struct TimeBin {
        float time_start;
        float time_end;
        std::vector<GridWithMetadata> grids_in_bin;
    };
    
    // Create time bins from grids
    std::vector<TimeBin> createTimeBins(
        const std::vector<GridWithMetadata>& grids,
        float window
    ) const;
    
    // Aggregate grids within a time bin
    FloatGridPtr aggregateGridsInBin(
        const TimeBin& bin,
        int layer_tolerance
    ) const;
};

} // namespace synchronization
} // namespace am_qadf_native

#endif // AM_QADF_NATIVE_SYNCHRONIZATION_GRID_TEMPORAL_ALIGNMENT_HPP
