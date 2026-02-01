#ifndef AM_QADF_NATIVE_CORRECTION_SPATIAL_NOISE_FILTERING_HPP
#define AM_QADF_NATIVE_CORRECTION_SPATIAL_NOISE_FILTERING_HPP

#include <openvdb/Grid.h>
#include <openvdb/tools/Filter.h>
#include <string>
#include <memory>

namespace am_qadf_native {
namespace correction {

using FloatGrid = openvdb::FloatGrid;
using FloatGridPtr = FloatGrid::Ptr;

// Spatial noise filtering (after mapping to voxel grids)
class SpatialNoiseFilter {
public:
    // Apply noise filtering on voxel grids
    FloatGridPtr apply(
        FloatGridPtr grid,
        const std::string& method,  // 'median', 'bilateral', 'gaussian'
        int kernel_size = 3,
        float sigma_spatial = 1.0f,
        float sigma_color = 0.1f
    );
    
    // Median filter
    FloatGridPtr applyMedianFilter(
        FloatGridPtr grid,
        int kernel_size = 3
    );
    
    // Bilateral filter (preserves edges)
    FloatGridPtr applyBilateralFilter(
        FloatGridPtr grid,
        float sigma_spatial = 1.0f,
        float sigma_color = 0.1f
    );
    
    // Gaussian filter
    FloatGridPtr applyGaussianFilter(
        FloatGridPtr grid,
        float sigma = 1.0f
    );
    
private:
    // Helper: Get neighborhood values
    std::vector<float> getNeighborhoodValues(
        FloatGridPtr grid,
        const openvdb::Coord& coord,
        int radius
    ) const;
};

} // namespace correction
} // namespace am_qadf_native

#endif // AM_QADF_NATIVE_CORRECTION_SPATIAL_NOISE_FILTERING_HPP
