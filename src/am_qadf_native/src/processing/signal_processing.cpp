#include "am_qadf_native/processing/signal_processing.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <complex>
#include <vector>
#include <string>
#include <functional>
#include <stdexcept>

#ifdef KFR_AVAILABLE
#include <kfr/kfr.h>
#include <kfr/dft.hpp>
#include <kfr/dsp.hpp>
#endif

namespace am_qadf_native {
namespace processing {

std::vector<float> SignalProcessing::normalize(
    const std::vector<float>& values,
    float min_value,
    float max_value
) {
    if (values.empty()) return values;
    
    float data_min = *std::min_element(values.begin(), values.end());
    float data_max = *std::max_element(values.begin(), values.end());
    float data_range = data_max - data_min;
    if (data_range == 0.0f) return values;
    
    float out_range = max_value - min_value;
    if (out_range == 0.0f) return values;
    
    // Map data range [data_min, data_max] to output [min_value, max_value]
    std::vector<float> normalized;
    normalized.reserve(values.size());
    for (float v : values) {
        float t = (v - data_min) / data_range;
        normalized.push_back(t * out_range + min_value);
    }
    return normalized;
}

std::vector<float> SignalProcessing::movingAverage(
    const std::vector<float>& values,
    int window_size
) {
    if (values.empty() || window_size <= 0) return values;
    
    std::vector<float> smoothed;
    smoothed.reserve(values.size());
    
    int half_window = window_size / 2;
    
    for (size_t i = 0; i < values.size(); ++i) {
        int start = std::max(0, static_cast<int>(i) - half_window);
        int end = std::min(static_cast<int>(values.size()), static_cast<int>(i) + half_window + 1);
        
        float sum = 0.0f;
        int count = 0;
        for (int j = start; j < end; ++j) {
            sum += values[j];
            count++;
        }
        
        smoothed.push_back(count > 0 ? sum / count : values[i]);
    }
    
    return smoothed;
}

std::vector<float> SignalProcessing::derivative(const std::vector<float>& values) {
    if (values.size() < 2) return std::vector<float>(values.size(), 0.0f);
    
    std::vector<float> deriv;
    deriv.reserve(values.size());
    
    deriv.push_back(0.0f);  // First point
    
    for (size_t i = 1; i < values.size(); ++i) {
        deriv.push_back(values[i] - values[i-1]);
    }
    
    return deriv;
}

std::vector<float> SignalProcessing::integral(const std::vector<float>& values) {
    if (values.empty()) return values;
    
    std::vector<float> integ;
    integ.reserve(values.size());
    
    float sum = 0.0f;
    for (float v : values) {
        sum += v;
        integ.push_back(sum);
    }
    
    return integ;
}

std::vector<std::complex<float>> SignalProcessing::fft(const std::vector<float>& values) {
    if (values.empty()) {
        return std::vector<std::complex<float>>();
    }
    
#ifdef KFR_AVAILABLE
    // Use KFR for FFT
    size_t n = values.size();
    
    // KFR requires size to be power of 2 or use zero-padding
    // Find next power of 2
    size_t fft_size = 1;
    while (fft_size < n) {
        fft_size <<= 1;
    }
    
    // Create KFR univector for input (zero-padded)
    kfr::univector<kfr::fbase> input(fft_size);
    std::copy(values.begin(), values.end(), input.begin());
    std::fill(input.begin() + n, input.end(), 0.0f);
    
    // Create DFT plan
    kfr::dft_plan<kfr::fbase> plan(fft_size);
    
    // Create output buffer
    kfr::univector<kfr::complex<kfr::fbase>> output(fft_size);
    
    // Execute FFT
    plan.execute(output, input);
    
    // Convert to std::complex<float> vector (only return original size)
    std::vector<std::complex<float>> spectrum(n);
    for (size_t i = 0; i < n; ++i) {
        spectrum[i] = std::complex<float>(
            static_cast<float>(output[i].real()),
            static_cast<float>(output[i].imag())
        );
    }
    
    return spectrum;
#else
    // Fallback: Return zeros if KFR not available
    return std::vector<std::complex<float>>(values.size(), std::complex<float>(0.0f, 0.0f));
#endif
}

std::vector<float> SignalProcessing::ifft(const std::vector<std::complex<float>>& spectrum) {
    if (spectrum.empty()) {
        return std::vector<float>();
    }
    
#ifdef KFR_AVAILABLE
    // Use KFR for inverse FFT
    size_t n = spectrum.size();
    
    // Find next power of 2
    size_t fft_size = 1;
    while (fft_size < n) {
        fft_size <<= 1;
    }
    
    // Create KFR univector for input (zero-padded)
    kfr::univector<kfr::complex<kfr::fbase>> input(fft_size);
    for (size_t i = 0; i < n; ++i) {
        input[i] = kfr::complex<kfr::fbase>(
            spectrum[i].real(),
            spectrum[i].imag()
        );
    }
    std::fill(input.begin() + n, input.end(), kfr::complex<kfr::fbase>(0.0f, 0.0f));
    
    // Create inverse DFT plan
    kfr::dft_plan<kfr::fbase> plan(fft_size);
    plan.set_inverse(true);
    
    // Create output buffer
    kfr::univector<kfr::fbase> output(fft_size);
    
    // Execute inverse FFT
    plan.execute(output, input);
    
    // Normalize by FFT size (IFFT requires normalization)
    float scale = 1.0f / static_cast<float>(fft_size);
    
    // Convert to float vector (only return original size)
    std::vector<float> values(n);
    for (size_t i = 0; i < n; ++i) {
        values[i] = static_cast<float>(output[i]) * scale;
    }
    
    return values;
#else
    // Fallback: Return zeros if KFR not available
    return std::vector<float>(spectrum.size(), 0.0f);
#endif
}

std::vector<float> SignalProcessing::frequencyFilter(
    const std::vector<float>& values,
    float cutoff_frequency,
    const std::string& filter_type
) {
    if (values.empty() || cutoff_frequency <= 0.0f) {
        return values;
    }
    
    // Apply FFT
    std::vector<std::complex<float>> spectrum = fft(values);
    
    if (spectrum.empty()) {
        return values;
    }
    
    // Filter in frequency domain
    size_t n = spectrum.size();
    float nyquist_freq = 0.5f;  // Normalized frequency (0.5 = Nyquist)
    float normalized_cutoff = cutoff_frequency;  // Assuming cutoff is already normalized [0, 0.5]
    
    // Clamp normalized cutoff
    if (normalized_cutoff > nyquist_freq) {
        normalized_cutoff = nyquist_freq;
    }
    
    // Compute frequency bin for cutoff
    size_t cutoff_bin = static_cast<size_t>(normalized_cutoff * n);
    
    // Apply filter based on type
    if (filter_type == "lowpass") {
        // Zero out frequencies above cutoff
        for (size_t i = cutoff_bin; i < n - cutoff_bin; ++i) {
            spectrum[i] = std::complex<float>(0.0f, 0.0f);
        }
        // Also handle negative frequencies (second half)
        if (n > 1) {
            for (size_t i = n - cutoff_bin; i < n; ++i) {
                spectrum[i] = std::complex<float>(0.0f, 0.0f);
            }
        }
    } else if (filter_type == "highpass") {
        // Zero out frequencies below cutoff
        for (size_t i = 0; i < cutoff_bin; ++i) {
            spectrum[i] = std::complex<float>(0.0f, 0.0f);
        }
        // Also handle negative frequencies (second half)
        if (n > 1) {
            for (size_t i = n - cutoff_bin; i < n; ++i) {
                spectrum[i] = std::complex<float>(0.0f, 0.0f);
            }
        }
    } else if (filter_type == "bandpass") {
        // For bandpass, we need two cutoff frequencies
        // For now, implement as lowpass with second cutoff at Nyquist
        float high_cutoff = nyquist_freq;
        size_t high_cutoff_bin = static_cast<size_t>(high_cutoff * n);
        
        // Zero out frequencies outside band [cutoff_bin, high_cutoff_bin]
        for (size_t i = 0; i < cutoff_bin; ++i) {
            spectrum[i] = std::complex<float>(0.0f, 0.0f);
        }
        for (size_t i = high_cutoff_bin; i < n - high_cutoff_bin; ++i) {
            spectrum[i] = std::complex<float>(0.0f, 0.0f);
        }
        if (n > 1) {
            for (size_t i = n - high_cutoff_bin; i < n; ++i) {
                spectrum[i] = std::complex<float>(0.0f, 0.0f);
            }
        }
    }
    // else: no filtering (allpass)
    
    // Apply inverse FFT
    return ifft(spectrum);
}

} // namespace processing
} // namespace am_qadf_native
