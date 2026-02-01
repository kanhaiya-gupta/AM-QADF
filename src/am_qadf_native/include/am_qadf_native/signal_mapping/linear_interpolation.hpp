#ifndef AM_QADF_NATIVE_SIGNAL_MAPPING_LINEAR_INTERPOLATION_HPP
#define AM_QADF_NATIVE_SIGNAL_MAPPING_LINEAR_INTERPOLATION_HPP

#include "interpolation_base.hpp"

namespace am_qadf_native {
namespace signal_mapping {

// Linear/Trilinear interpolation (using BoxSampler logic)
class LinearMapper : public InterpolationBase {
public:
    void map(
        FloatGridPtr grid,
        const std::vector<Point>& points,
        const std::vector<float>& values
    ) override;
    
private:
    // Distribute value to 8 neighboring voxels using trilinear weights
    void distributeTrilinear(FloatGridPtr grid, const openvdb::Vec3R& coord, float value);
};

} // namespace signal_mapping
} // namespace am_qadf_native

#endif // AM_QADF_NATIVE_SIGNAL_MAPPING_LINEAR_INTERPOLATION_HPP
