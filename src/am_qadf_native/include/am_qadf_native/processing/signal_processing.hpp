#ifndef AM_QADF_NATIVE_PROCESSING_SIGNAL_PROCESSING_HPP
#define AM_QADF_NATIVE_PROCESSING_SIGNAL_PROCESSING_HPP

#include <vector>
#include <string>
#include <complex>

namespace am_qadf_native {
namespace processing {

// Signal processing utilities
class SignalProcessing {
public:
    // Normalize signal values
    std::vector<float> normalize(
        const std::vector<float>& values,
        float min_value = 0.0f,
        float max_value = 1.0f
    );
    
    // Apply moving average
    std::vector<float> movingAverage(
        const std::vector<float>& values,
        int window_size = 5
    );
    
    // Apply derivative
    std::vector<float> derivative(const std::vector<float>& values);
    
    // Apply integral
    std::vector<float> integral(const std::vector<float>& values);
    
    // Apply FFT
    std::vector<std::complex<float>> fft(const std::vector<float>& values);
    
    // Apply inverse FFT
    std::vector<float> ifft(const std::vector<std::complex<float>>& spectrum);
    
    // Filter frequency domain
    std::vector<float> frequencyFilter(
        const std::vector<float>& values,
        float cutoff_frequency,
        const std::string& filter_type = "lowpass"  // "lowpass", "highpass", "bandpass"
    );
};

} // namespace processing
} // namespace am_qadf_native

#endif // AM_QADF_NATIVE_PROCESSING_SIGNAL_PROCESSING_HPP
