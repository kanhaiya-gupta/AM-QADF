#ifndef AM_QADF_NATIVE_FUSION_FUSION_QUALITY_HPP
#define AM_QADF_NATIVE_FUSION_FUSION_QUALITY_HPP

#include <openvdb/openvdb.h>
#include <openvdb/Grid.h>
#include <vector>
#include <string>
#include <map>
#include <memory>

namespace am_qadf_native {
namespace fusion {

using FloatGrid = openvdb::FloatGrid;
using FloatGridPtr = FloatGrid::Ptr;

/// Result of fusion quality assessment (matches Python FusionQualityMetrics).
struct FusionQualityResult {
    float fusion_accuracy = 0.0f;
    float signal_consistency = 0.0f;
    float fusion_completeness = 0.0f;
    float quality_score = 0.0f;
    float coverage_ratio = 0.0f;
    std::map<std::string, float> per_signal_accuracy;
    /// Optional residual summary (mean, max, std) when requested
    float residual_mean = 0.0f;
    float residual_max = 0.0f;
    float residual_std = 0.0f;
    bool has_residual_summary = false;
};

/// Assesses quality of fused voxel signals (C++ core for scale).
class FusionQualityAssessor {
public:
    FusionQualityAssessor() = default;

    /// Assess quality: fused grid vs source grids, optional per-source weights.
    /// Uses OpenVDB iterators (sparse traversal, no dense conversion).
    FusionQualityResult assess(
        FloatGridPtr fused_grid,
        const std::map<std::string, FloatGridPtr>& source_grids,
        const std::map<std::string, float>& weights = {}
    ) const;

private:
    /// Valid value: active and not zero (match Python: valid = not NaN and != 0).
    static bool isValid(float v);
    /// Pearson correlation between two value streams (same coords).
    static float correlation(
        const std::vector<float>& a,
        const std::vector<float>& b
    );
    /// Normalized RMSE: max(0, 1 - rmse(normalized_a, normalized_b)).
    static float consistencyScore(
        const std::vector<float>& fused_vals,
        const std::vector<float>& source_vals
    );
};

} // namespace fusion
} // namespace am_qadf_native

#endif // AM_QADF_NATIVE_FUSION_FUSION_QUALITY_HPP
