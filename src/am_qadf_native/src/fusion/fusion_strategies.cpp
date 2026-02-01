#include "am_qadf_native/fusion/fusion_strategies.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <vector>
#include <stdexcept>

namespace am_qadf_native {
namespace fusion {

WeightedAverageStrategy::WeightedAverageStrategy(const std::vector<float>& weights)
    : weights_(weights) {
}

float WeightedAverageStrategy::fuseValues(const std::vector<float>& values) {
    if (values.size() != weights_.size()) {
        throw std::invalid_argument("Values and weights size mismatch");
    }
    
    float sum = 0.0f;
    float sum_weights = 0.0f;
    for (size_t i = 0; i < values.size(); ++i) {
        sum += values[i] * weights_[i];
        sum_weights += weights_[i];
    }
    
    return sum_weights > 0.0f ? sum / sum_weights : 0.0f;
}

float MaxStrategy::fuseValues(const std::vector<float>& values) {
    if (values.empty()) return 0.0f;
    return *std::max_element(values.begin(), values.end());
}

float MinStrategy::fuseValues(const std::vector<float>& values) {
    if (values.empty()) return 0.0f;
    return *std::min_element(values.begin(), values.end());
}

float MedianStrategy::fuseValues(const std::vector<float>& values) {
    if (values.empty()) return 0.0f;
    
    std::vector<float> sorted_values = values;
    std::sort(sorted_values.begin(), sorted_values.end());
    
    size_t n = sorted_values.size();
    if (n % 2 == 0) {
        return (sorted_values[n/2 - 1] + sorted_values[n/2]) / 2.0f;
    } else {
        return sorted_values[n/2];
    }
}

} // namespace fusion
} // namespace am_qadf_native
