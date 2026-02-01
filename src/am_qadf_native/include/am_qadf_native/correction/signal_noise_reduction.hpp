#ifndef AM_QADF_NATIVE_CORRECTION_SIGNAL_NOISE_REDUCTION_HPP
#define AM_QADF_NATIVE_CORRECTION_SIGNAL_NOISE_REDUCTION_HPP

#include "../query/query_result.hpp"
#include <string>

namespace am_qadf_native {
namespace correction {

using QueryResult = query::QueryResult;

// Signal-level noise reduction (before mapping to voxels)
class SignalNoiseReduction {
public:
    // Reduce noise in raw signal data
    QueryResult reduceNoise(
        const QueryResult& raw_data,
        const std::string& method,  // 'gaussian', 'savitzky_golay', 'outlier_removal'
        float sigma = 1.0
    );
    
    // Gaussian filter
    QueryResult applyGaussianFilter(
        const QueryResult& data,
        float sigma
    );
    
    // Savitzky-Golay filter
    QueryResult applySavitzkyGolay(
        const QueryResult& data,
        int window_size = 5,
        int polynomial_order = 3
    );
    
    // Outlier removal
    QueryResult removeOutliers(
        const QueryResult& data,
        float threshold = 3.0f  // Standard deviations
    );
    
private:
    // Helper methods
    float computeMean(const std::vector<float>& values) const;
    float computeStdDev(const std::vector<float>& values) const;
};

} // namespace correction
} // namespace am_qadf_native

#endif // AM_QADF_NATIVE_CORRECTION_SIGNAL_NOISE_REDUCTION_HPP
