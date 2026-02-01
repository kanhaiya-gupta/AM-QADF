#ifndef AM_QADF_NATIVE_SIGNAL_MAPPING_NEAREST_NEIGHBOR_HPP
#define AM_QADF_NATIVE_SIGNAL_MAPPING_NEAREST_NEIGHBOR_HPP

#include "interpolation_base.hpp"

namespace am_qadf_native {
namespace signal_mapping {

// Nearest Neighbor interpolation (using PointSampler logic)
class NearestNeighborMapper : public InterpolationBase {
public:
    void map(
        FloatGridPtr grid,
        const std::vector<Point>& points,
        const std::vector<float>& values
    ) override;
};

} // namespace signal_mapping
} // namespace am_qadf_native

#endif // AM_QADF_NATIVE_SIGNAL_MAPPING_NEAREST_NEIGHBOR_HPP
