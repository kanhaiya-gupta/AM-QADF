#include "am_qadf_native/correction/signal_noise_reduction.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <vector>
#include <array>
#include <string>
#include <stdexcept>

#ifdef EIGEN_AVAILABLE
#include <Eigen/Dense>
#endif

namespace am_qadf_native {
namespace correction {

QueryResult SignalNoiseReduction::reduceNoise(
    const QueryResult& raw_data,
    const std::string& method,
    float sigma
) {
    if (method == "gaussian") {
        return applyGaussianFilter(raw_data, sigma);
    } else if (method == "savitzky_golay") {
        return applySavitzkyGolay(raw_data);
    } else if (method == "outlier_removal") {
        return removeOutliers(raw_data);
    }
    
    // Default: return original data
    return raw_data;
}

QueryResult SignalNoiseReduction::applyGaussianFilter(
    const QueryResult& data,
    float sigma
) {
    QueryResult filtered = data;
    
    if (data.values.empty() || sigma <= 0.0f) {
        return filtered;
    }
    
    // Compute Gaussian kernel
    // Kernel size: 6*sigma (covers 99.7% of distribution)
    int kernel_size = static_cast<int>(std::ceil(6.0f * sigma));
    if (kernel_size % 2 == 0) {
        kernel_size += 1;  // Make odd
    }
    if (kernel_size < 3) {
        kernel_size = 3;  // Minimum kernel size
    }
    
    int half_kernel = kernel_size / 2;
    std::vector<float> kernel(kernel_size);
    float sum = 0.0f;
    
    // Build Gaussian kernel
    float two_sigma_sq = 2.0f * sigma * sigma;
    for (int i = 0; i < kernel_size; ++i) {
        int offset = i - half_kernel;
        float value = std::exp(-(offset * offset) / two_sigma_sq);
        kernel[i] = value;
        sum += value;
    }
    
    // Normalize kernel
    for (auto& k : kernel) {
        k /= sum;
    }
    
    // Apply convolution
    std::vector<float> filtered_values;
    filtered_values.reserve(data.values.size());
    
    for (size_t i = 0; i < data.values.size(); ++i) {
        float filtered_value = 0.0f;
        float weight_sum = 0.0f;
        
        for (int j = 0; j < kernel_size; ++j) {
            int idx = static_cast<int>(i) + j - half_kernel;
            if (idx >= 0 && idx < static_cast<int>(data.values.size())) {
                filtered_value += data.values[idx] * kernel[j];
                weight_sum += kernel[j];
            }
        }
        
        if (weight_sum > 0.0f) {
            filtered_values.push_back(filtered_value / weight_sum);
        } else {
            filtered_values.push_back(data.values[i]);
        }
    }
    
    filtered.values = filtered_values;
    return filtered;
}

QueryResult SignalNoiseReduction::applySavitzkyGolay(
    const QueryResult& data,
    int window_size,
    int polynomial_order
) {
    QueryResult filtered = data;
    
    if (data.values.empty() || window_size < 3 || window_size % 2 == 0) {
        return filtered;
    }
    
    if (polynomial_order >= window_size) {
        polynomial_order = window_size - 1;  // Polynomial order must be less than window size
    }
    
#ifdef EIGEN_AVAILABLE
    // Compute Savitzky-Golay filter coefficients using Eigen
    int half_window = window_size / 2;
    
    // Build Vandermonde matrix A: each row is [1, x, x^2, x^3, ...] for x in [-half_window, half_window]
    Eigen::MatrixXf A(window_size, polynomial_order + 1);
    for (int i = 0; i < window_size; ++i) {
        int x = i - half_window;  // x coordinate: -half_window to +half_window
        float x_power = 1.0f;
        for (int j = 0; j <= polynomial_order; ++j) {
            A(i, j) = x_power;
            x_power *= x;
        }
    }
    
    // Compute pseudo-inverse: coefficients = (A^T * A)^(-1) * A^T
    // For Savitzky-Golay, we need the center row of A (evaluating polynomial at x=0).
    // RHS must be a column vector of size (polynomial_order+1), i.e. A.row(half_window).transpose().
    Eigen::MatrixXf ATA = A.transpose() * A;
    Eigen::VectorXf center_row = ATA.colPivHouseholderQr().solve(A.row(half_window).transpose());
    
    // Extract filter coefficients (for center point evaluation)
    std::vector<float> coefficients(window_size);
    for (int i = 0; i < window_size; ++i) {
        // Coefficient is dot product of center_row with i-th row of A
        float coeff = 0.0f;
        for (int j = 0; j <= polynomial_order; ++j) {
            coeff += center_row(j) * A(i, j);
        }
        coefficients[i] = coeff;
    }
    
    // Apply filter
    std::vector<float> filtered_values;
    filtered_values.reserve(data.values.size());
    
    for (size_t i = 0; i < data.values.size(); ++i) {
        float filtered_value = 0.0f;
        
        for (int j = 0; j < window_size; ++j) {
            int idx = static_cast<int>(i) + j - half_window;
            if (idx >= 0 && idx < static_cast<int>(data.values.size())) {
                filtered_value += data.values[idx] * coefficients[j];
            } else {
                // Handle boundaries: use nearest valid value
                int clamped_idx = std::max(0, std::min(static_cast<int>(data.values.size()) - 1, idx));
                filtered_value += data.values[clamped_idx] * coefficients[j];
            }
        }
        
        filtered_values.push_back(filtered_value);
    }
    
    filtered.values = filtered_values;
    return filtered;
    
#else
    // Fallback: Moving average if Eigen not available
    int half_window = window_size / 2;
    std::vector<float> filtered_values;
    filtered_values.reserve(data.values.size());
    
    for (size_t i = 0; i < data.values.size(); ++i) {
        int start = std::max(0, static_cast<int>(i) - half_window);
        int end = std::min(static_cast<int>(data.values.size()), static_cast<int>(i) + half_window + 1);
        
        float sum = 0.0f;
        int count = 0;
        for (int j = start; j < end; ++j) {
            sum += data.values[j];
            count++;
        }
        
        filtered_values.push_back(count > 0 ? sum / count : data.values[i]);
    }
    
    filtered.values = filtered_values;
    return filtered;
#endif
}

QueryResult SignalNoiseReduction::removeOutliers(
    const QueryResult& data,
    float threshold
) {
    QueryResult cleaned = data;
    
    // Compute mean and standard deviation
    float mean_val = computeMean(data.values);
    float std_val = computeStdDev(data.values);
    
    // Remove outliers
    std::vector<std::array<float, 3>> filtered_points;
    std::vector<float> filtered_values;
    std::vector<float> filtered_timestamps;
    std::vector<int> filtered_layers;
    
    for (size_t i = 0; i < data.values.size(); ++i) {
        float z_score = std::abs((data.values[i] - mean_val) / std_val);
        if (z_score <= threshold) {
            filtered_points.push_back(data.points[i]);
            filtered_values.push_back(data.values[i]);
            if (i < data.timestamps.size()) {
                filtered_timestamps.push_back(data.timestamps[i]);
            }
            if (i < data.layers.size()) {
                filtered_layers.push_back(data.layers[i]);
            }
        }
    }
    
    cleaned.points = filtered_points;
    cleaned.values = filtered_values;
    cleaned.timestamps = filtered_timestamps;
    cleaned.layers = filtered_layers;
    
    return cleaned;
}

float SignalNoiseReduction::computeMean(const std::vector<float>& values) const {
    if (values.empty()) return 0.0f;
    float sum = std::accumulate(values.begin(), values.end(), 0.0f);
    return sum / static_cast<float>(values.size());
}

float SignalNoiseReduction::computeStdDev(const std::vector<float>& values) const {
    if (values.empty()) return 0.0f;
    float mean_val = computeMean(values);
    float sum_sq_diff = 0.0f;
    for (float v : values) {
        float diff = v - mean_val;
        sum_sq_diff += diff * diff;
    }
    return std::sqrt(sum_sq_diff / static_cast<float>(values.size()));
}

} // namespace correction
} // namespace am_qadf_native
