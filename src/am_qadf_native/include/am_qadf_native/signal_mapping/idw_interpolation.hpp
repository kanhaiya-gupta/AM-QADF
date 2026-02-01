#ifndef AM_QADF_NATIVE_SIGNAL_MAPPING_IDW_INTERPOLATION_HPP
#define AM_QADF_NATIVE_SIGNAL_MAPPING_IDW_INTERPOLATION_HPP

#include "interpolation_base.hpp"

namespace am_qadf_native {
namespace signal_mapping {

// Inverse Distance Weighting (IDW) interpolation
class IDWMapper : public InterpolationBase {
private:
    float power_;  // IDW power parameter
    int k_neighbors_;  // Number of nearest neighbors to use
    
public:
    IDWMapper(float power = 2.0f, int k_neighbors = 10);
    
    void map(
        FloatGridPtr grid,
        const std::vector<Point>& points,
        const std::vector<float>& values
    ) override;
    
    void setPower(float power);
    void setKNeighbors(int k);
};

} // namespace signal_mapping
} // namespace am_qadf_native

#endif // AM_QADF_NATIVE_SIGNAL_MAPPING_IDW_INTERPOLATION_HPP
