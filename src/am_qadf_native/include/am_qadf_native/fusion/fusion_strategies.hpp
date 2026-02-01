#ifndef AM_QADF_NATIVE_FUSION_FUSION_STRATEGIES_HPP
#define AM_QADF_NATIVE_FUSION_FUSION_STRATEGIES_HPP

#include "grid_fusion.hpp"
#include <string>

namespace am_qadf_native {
namespace fusion {

// Fusion strategy interface
class FusionStrategy {
public:
    virtual ~FusionStrategy() = default;
    virtual float fuseValues(const std::vector<float>& values) = 0;
};

// Weighted average strategy
class WeightedAverageStrategy : public FusionStrategy {
private:
    std::vector<float> weights_;
    
public:
    WeightedAverageStrategy(const std::vector<float>& weights);
    float fuseValues(const std::vector<float>& values) override;
};

// Maximum strategy
class MaxStrategy : public FusionStrategy {
public:
    float fuseValues(const std::vector<float>& values) override;
};

// Minimum strategy
class MinStrategy : public FusionStrategy {
public:
    float fuseValues(const std::vector<float>& values) override;
};

// Median strategy
class MedianStrategy : public FusionStrategy {
public:
    float fuseValues(const std::vector<float>& values) override;
};

} // namespace fusion
} // namespace am_qadf_native

#endif // AM_QADF_NATIVE_FUSION_FUSION_STRATEGIES_HPP
