#ifndef AM_QADF_NATIVE_SIGNAL_MAPPING_RBF_INTERPOLATION_HPP
#define AM_QADF_NATIVE_SIGNAL_MAPPING_RBF_INTERPOLATION_HPP

#include "interpolation_base.hpp"
#include <string>

namespace am_qadf_native {
namespace signal_mapping {

// Radial Basis Function (RBF) interpolation
class RBFMapper : public InterpolationBase {
private:
    std::string kernel_type_;  // "gaussian", "multiquadric", "thin_plate", etc.
    float epsilon_;  // RBF shape parameter
    
public:
    RBFMapper(const std::string& kernel_type = "gaussian", float epsilon = 1.0f);
    
    void map(
        FloatGridPtr grid,
        const std::vector<Point>& points,
        const std::vector<float>& values
    ) override;
    
    void setKernelType(const std::string& kernel_type);
    void setEpsilon(float epsilon);
    
private:
    float rbfKernel(float distance) const;
    void solveRBFSystem(const std::vector<Point>& points, 
                       const std::vector<float>& values,
                       std::vector<float>& weights);
};

} // namespace signal_mapping
} // namespace am_qadf_native

#endif // AM_QADF_NATIVE_SIGNAL_MAPPING_RBF_INTERPOLATION_HPP
