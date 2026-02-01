#ifndef AM_QADF_NATIVE_FUSION_GRID_FUSION_HPP
#define AM_QADF_NATIVE_FUSION_GRID_FUSION_HPP

#include <openvdb/openvdb.h>
#include <openvdb/Grid.h>
#include <vector>
#include <string>
#include <memory>

namespace am_qadf_native {
namespace fusion {

using FloatGrid = openvdb::FloatGrid;
using FloatGridPtr = FloatGrid::Ptr;

// Grid fusion operations
class GridFusion {
public:
    // Fuse multiple grids into a single grid
    FloatGridPtr fuse(
        const std::vector<FloatGridPtr>& grids,
        const std::string& strategy = "weighted_average"
    );
    
    // Fuse with custom weights
    FloatGridPtr fuseWeighted(
        const std::vector<FloatGridPtr>& grids,
        const std::vector<float>& weights
    );
    
private:
    // Fusion strategies
    void fuseWeightedAverage(const std::vector<FloatGridPtr>& grids,
                            FloatGridPtr output);
    void fuseMax(const std::vector<FloatGridPtr>& grids, FloatGridPtr output);
    void fuseMin(const std::vector<FloatGridPtr>& grids, FloatGridPtr output);
    void fuseMedian(const std::vector<FloatGridPtr>& grids, FloatGridPtr output);
};

} // namespace fusion
} // namespace am_qadf_native

#endif // AM_QADF_NATIVE_FUSION_GRID_FUSION_HPP
