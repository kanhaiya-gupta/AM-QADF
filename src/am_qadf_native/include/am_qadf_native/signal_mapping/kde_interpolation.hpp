#ifndef AM_QADF_NATIVE_SIGNAL_MAPPING_KDE_INTERPOLATION_HPP
#define AM_QADF_NATIVE_SIGNAL_MAPPING_KDE_INTERPOLATION_HPP

#include "interpolation_base.hpp"

namespace am_qadf_native {
namespace signal_mapping {

// Kernel Density Estimation (KDE) interpolation
class KDEMapper : public InterpolationBase {
private:
    float bandwidth_;  // KDE bandwidth
    std::string kernel_type_;  // "gaussian", "epanechnikov", etc.
    
public:
    KDEMapper(float bandwidth = 1.0f, const std::string& kernel_type = "gaussian");
    
    void map(
        FloatGridPtr grid,
        const std::vector<Point>& points,
        const std::vector<float>& values
    ) override;
    
    void setBandwidth(float bandwidth);
    void setKernelType(const std::string& kernel_type);
    
private:
    float kernelFunction(float distance) const;
};

} // namespace signal_mapping
} // namespace am_qadf_native

#endif // AM_QADF_NATIVE_SIGNAL_MAPPING_KDE_INTERPOLATION_HPP
