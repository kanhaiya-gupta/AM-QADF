#include "am_qadf_native/fusion/fusion_quality.hpp"
#include <openvdb/openvdb.h>
#include <openvdb/tools/Statistics.h>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace am_qadf_native {
namespace fusion {

namespace {
constexpr float EPS = 1e-10f;
}

bool FusionQualityAssessor::isValid(float v) {
    return std::isfinite(v) && std::abs(v) > EPS;
}

float FusionQualityAssessor::correlation(
    const std::vector<float>& a,
    const std::vector<float>& b
) {
    if (a.size() != b.size() || a.size() < 2u) return 0.0f;
    size_t n = a.size();
    float sum_a = 0.0f, sum_b = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum_a += a[i];
        sum_b += b[i];
    }
    float mean_a = sum_a / static_cast<float>(n);
    float mean_b = sum_b / static_cast<float>(n);
    float var_a = 0.0f, var_b = 0.0f, cov = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float da = a[i] - mean_a, db = b[i] - mean_b;
        var_a += da * da;
        var_b += db * db;
        cov += da * db;
    }
    float denom = std::sqrt(var_a * var_b);
    if (denom < EPS) return 0.0f;
    float corr = cov / denom;
    return std::max(-1.0f, std::min(1.0f, corr));
}

float FusionQualityAssessor::consistencyScore(
    const std::vector<float>& fused_vals,
    const std::vector<float>& source_vals
) {
    if (fused_vals.size() != source_vals.size() || fused_vals.empty()) return 0.0f;
    float max_f = *std::max_element(fused_vals.begin(), fused_vals.end());
    float max_s = *std::max_element(source_vals.begin(), source_vals.end());
    if (max_f < EPS || max_s < EPS) return 0.0f;
    float sum_sq = 0.0f;
    size_t n = fused_vals.size();
    for (size_t i = 0; i < n; ++i) {
        float fn = fused_vals[i] / max_f;
        float sn = source_vals[i] / max_s;
        float d = fn - sn;
        sum_sq += d * d;
    }
    float rmse = std::sqrt(sum_sq / static_cast<float>(n));
    return std::max(0.0f, 1.0f - rmse);
}

FusionQualityResult FusionQualityAssessor::assess(
    FloatGridPtr fused_grid,
    const std::map<std::string, FloatGridPtr>& source_grids,
    const std::map<std::string, float>& weights
) const {
    FusionQualityResult result;
    if (!fused_grid) return result;

    openvdb::CoordBBox bbox = fused_grid->evalActiveVoxelBoundingBox();
    openvdb::Coord dim = bbox.max() - bbox.min() + openvdb::Coord(1, 1, 1);
    openvdb::Index64 bbox_volume = static_cast<openvdb::Index64>(dim.x()) * static_cast<openvdb::Index64>(dim.y()) * static_cast<openvdb::Index64>(dim.z());
    if (bbox_volume == 0) bbox_volume = 1;
    openvdb::Index64 active_count = fused_grid->activeVoxelCount();
    result.coverage_ratio = static_cast<float>(active_count) / static_cast<float>(bbox_volume);
    result.fusion_completeness = result.coverage_ratio;

    if (source_grids.empty()) {
        result.quality_score = 0.3f * result.coverage_ratio;
        return result;
    }

    std::vector<float> consistency_scores;
    std::vector<float> residual_vals;

    for (const auto& kv : source_grids) {
        const std::string& name = kv.first;
        FloatGridPtr src = kv.second;
        if (!src) {
            result.per_signal_accuracy[name] = 0.0f;
            continue;
        }

        std::vector<float> fused_vals, source_vals;
        float weight = 1.0f;
        auto wit = weights.find(name);
        if (wit != weights.end()) weight = wit->second;

        for (auto iter = fused_grid->cbeginValueOn(); iter; ++iter) {
            openvdb::Coord coord = iter.getCoord();
            float fv = iter.getValue();
            if (!isValid(fv)) continue;
            if (!src->tree().isValueOn(coord)) continue;
            float sv = src->tree().getValue(coord);
            if (!isValid(sv)) continue;
            fused_vals.push_back(fv);
            source_vals.push_back(sv);
            if (result.has_residual_summary || true) {
                residual_vals.push_back(weight * std::abs(fv - sv));
            }
        }

        if (fused_vals.size() < 2u) {
            result.per_signal_accuracy[name] = fused_vals.size() == 1u ? 1.0f : 0.0f;
            consistency_scores.push_back(result.per_signal_accuracy[name]);
            continue;
        }

        float corr = correlation(fused_vals, source_vals);
        result.per_signal_accuracy[name] = std::abs(corr);
        float cons = consistencyScore(fused_vals, source_vals);
        consistency_scores.push_back(cons);
    }

    float sum_acc = 0.0f;
    for (const auto& kv : result.per_signal_accuracy) {
        sum_acc += kv.second;
    }
    result.fusion_accuracy = result.per_signal_accuracy.empty()
        ? 0.0f
        : sum_acc / static_cast<float>(result.per_signal_accuracy.size());

    result.signal_consistency = consistency_scores.empty()
        ? 0.0f
        : std::accumulate(consistency_scores.begin(), consistency_scores.end(), 0.0f) / static_cast<float>(consistency_scores.size());

    result.quality_score = 0.4f * result.fusion_accuracy + 0.3f * result.signal_consistency + 0.3f * result.coverage_ratio;

    if (!residual_vals.empty()) {
        result.has_residual_summary = true;
        float sum = std::accumulate(residual_vals.begin(), residual_vals.end(), 0.0f);
        result.residual_mean = sum / static_cast<float>(residual_vals.size());
        result.residual_max = *std::max_element(residual_vals.begin(), residual_vals.end());
        float mean = result.residual_mean;
        float var = 0.0f;
        for (float v : residual_vals) {
            float d = v - mean;
            var += d * d;
        }
        result.residual_std = std::sqrt(var / static_cast<float>(residual_vals.size()));
    }

    return result;
}

} // namespace fusion
} // namespace am_qadf_native
