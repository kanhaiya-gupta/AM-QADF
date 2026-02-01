#ifndef AM_QADF_NATIVE_PROCESSING_SIGNAL_GENERATION_HPP
#define AM_QADF_NATIVE_PROCESSING_SIGNAL_GENERATION_HPP

#include <vector>
#include <string>
#include <array>

namespace am_qadf_native {
namespace processing {

// Signal generation utilities
class SignalGeneration {
public:
    // Generate synthetic signal data
    std::vector<float> generateSynthetic(
        const std::vector<std::array<float, 3>>& points,
        const std::string& signal_type,  // "gaussian", "sine", "random", etc.
        float amplitude = 1.0f,
        float frequency = 1.0f
    );
    
    // Generate Gaussian signal
    std::vector<float> generateGaussian(
        const std::vector<std::array<float, 3>>& points,
        const std::array<float, 3>& center,
        float amplitude = 1.0f,
        float sigma = 1.0f
    );
    
    // Generate sine wave signal
    std::vector<float> generateSineWave(
        const std::vector<std::array<float, 3>>& points,
        float amplitude = 1.0f,
        float frequency = 1.0f,
        const std::array<float, 3>& direction = {1.0f, 0.0f, 0.0f}
    );
    
    // Generate random signal
    std::vector<float> generateRandom(
        size_t num_points,
        float min_value = 0.0f,
        float max_value = 1.0f,
        unsigned int seed = 0
    );
    
    // Generate signal from mathematical expression
    std::vector<float> generateFromExpression(
        const std::vector<std::array<float, 3>>& points,
        const std::string& expression  // e.g., "x^2 + y^2 + z^2"
    );
};

} // namespace processing
} // namespace am_qadf_native

#endif // AM_QADF_NATIVE_PROCESSING_SIGNAL_GENERATION_HPP
