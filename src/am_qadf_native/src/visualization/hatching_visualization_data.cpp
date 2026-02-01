#include "am_qadf_native/visualization/hatching_visualization_data.hpp"
#include "am_qadf_native/query/mongodb_query_client.hpp"
#include "am_qadf_native/query/query_result.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace am_qadf_native {
namespace visualization {

namespace {

using Vector = query::QueryResult::Vector;
using VectorData = query::QueryResult::VectorData;

const std::array<float, 3> kNoBboxMin{std::numeric_limits<float>::lowest(),
                                       std::numeric_limits<float>::lowest(),
                                       std::numeric_limits<float>::lowest()};
const std::array<float, 3> kNoBboxMax{std::numeric_limits<float>::max(),
                                     std::numeric_limits<float>::max(),
                                     std::numeric_limits<float>::max()};

bool is_contour_type(const std::string& type) {
    if (type.empty()) return false;
    std::string t = type;
    std::transform(t.begin(), t.end(), t.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return (t == "outer" || t == "inner" || t == "contour");
}

float segment_length(const Vector& v) {
    float dx = v.x2 - v.x1;
    float dy = v.y2 - v.y1;
    return std::sqrt(dx * dx + dy * dy);
}

}  // namespace

HatchingVisualizationResult get_hatching_visualization_data(
    const std::string& model_id,
    int layer_start,
    int layer_end,
    const std::string& scalar_name,
    const std::string& uri,
    const std::string& db_name
) {
    HatchingVisualizationResult out;
    out.active_scalar_name = "laser_power";

    query::MongoDBQueryClient client(uri, db_name);
    query::QueryResult result = client.queryHatchingData(model_id, layer_start, layer_end, kNoBboxMin, kNoBboxMax);

    if (result.format != "vector-based" || result.vectors.empty()) {
        return out;
    }

    std::map<int, VectorData> di_to_vd;
    for (const auto& vd : result.vectordata) {
        if (vd.dataindex >= 0) {
            di_to_vd[vd.dataindex] = vd;
        }
    }

    std::map<int, std::vector<std::pair<Vector, VectorData>>> by_layer;
    for (const Vector& vec : result.vectors) {
        auto it = di_to_vd.find(vec.dataindex);
        if (it == di_to_vd.end()) continue;
        int layer_index = it->second.layer_index;
        by_layer[layer_index].emplace_back(vec, it->second);
    }

    std::vector<std::pair<Vector, VectorData>> ordered;
    std::vector<int> ordered_types;
    ordered.reserve(result.vectors.size());
    ordered_types.reserve(result.vectors.size());

    for (const auto& kv : by_layer) {
        std::vector<std::pair<Vector, VectorData>> hatch_pairs;
        std::vector<std::pair<Vector, VectorData>> contour_pairs;
        for (const auto& p : kv.second) {
            if (is_contour_type(p.second.type))
                contour_pairs.push_back(p);
            else
                hatch_pairs.push_back(p);
        }

        std::sort(contour_pairs.begin(), contour_pairs.end(),
                  [](const std::pair<Vector, VectorData>& a, const std::pair<Vector, VectorData>& b) {
                      return a.first.dataindex < b.first.dataindex;
                  });

        if (!hatch_pairs.empty()) {
            float dx_sum = 0.0f, dy_sum = 0.0f;
            for (const auto& p : hatch_pairs) {
                dx_sum += p.first.x2 - p.first.x1;
                dy_sum += p.first.y2 - p.first.y1;
            }
            size_t n_hatch = hatch_pairs.size();
            float ax = dx_sum / static_cast<float>(n_hatch);
            float ay = dy_sum / static_cast<float>(n_hatch);
            float len = std::sqrt(ax * ax + ay * ay);
            if (len > 1e-10f) {
                float perp_x = -ay / len;
                float perp_y = ax / len;
                std::sort(hatch_pairs.begin(), hatch_pairs.end(),
                          [perp_x, perp_y](const std::pair<Vector, VectorData>& a, const std::pair<Vector, VectorData>& b) {
                              float mx_a = (a.first.x1 + a.first.x2) * 0.5f;
                              float my_a = (a.first.y1 + a.first.y2) * 0.5f;
                              float mx_b = (b.first.x1 + b.first.x2) * 0.5f;
                              float my_b = (b.first.y1 + b.first.y2) * 0.5f;
                              float key_a = mx_a * perp_x + my_a * perp_y;
                              float key_b = mx_b * perp_x + my_b * perp_y;
                              if (key_a != key_b) return key_a < key_b;
                              if (mx_a != mx_b) return mx_a < mx_b;
                              return my_a < my_b;
                          });
            } else {
                std::sort(hatch_pairs.begin(), hatch_pairs.end(),
                          [](const std::pair<Vector, VectorData>& a, const std::pair<Vector, VectorData>& b) {
                              float mx_a = (a.first.x1 + a.first.x2) * 0.5f;
                              float my_a = (a.first.y1 + a.first.y2) * 0.5f;
                              float mx_b = (b.first.x1 + b.first.x2) * 0.5f;
                              float my_b = (b.first.y1 + b.first.y2) * 0.5f;
                              if (my_a != my_b) return my_a < my_b;
                              return mx_a < mx_b;
                          });
            }
        }

        for (const auto& p : hatch_pairs) {
            ordered.push_back(p);
            ordered_types.push_back(0);  /* Hatch */
        }
        for (const auto& p : contour_pairs) {
            ordered.push_back(p);
            ordered_types.push_back(1);  /* Contour */
        }
    }

    std::string scalar_lower = scalar_name;
    std::transform(scalar_lower.begin(), scalar_lower.end(), scalar_lower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    bool path_only = (scalar_lower == "path" || scalar_lower == "none" || scalar_lower.empty());
    bool use_laser = (scalar_lower == "laser_power" || scalar_lower == "laserpower");
    bool use_speed = (scalar_lower == "scan_speed" || scalar_lower == "scannerspeed");
    bool use_length = (scalar_lower == "length" || scalar_lower == "segment_length");
    if (path_only)
        use_length = true;

    if (use_laser)
        out.active_scalar_name = "laser_power";
    else if (use_speed)
        out.active_scalar_name = "scan_speed";
    else if (use_length)
        out.active_scalar_name = "length";
    else
        use_laser = true;

    const size_t n = ordered.size();
    out.positions.reserve(6 * n);
    out.scalars.reserve(n);
    out.segment_types.reserve(n);

    for (size_t i = 0; i < n; ++i) {
        const auto& p = ordered[i];
        const Vector& v = p.first;
        const VectorData& vd = p.second;
        out.positions.push_back(v.x1);
        out.positions.push_back(v.y1);
        out.positions.push_back(v.z);
        out.positions.push_back(v.x2);
        out.positions.push_back(v.y2);
        out.positions.push_back(v.z);
        if (use_length)
            out.scalars.push_back(segment_length(v));
        else if (use_speed)
            out.scalars.push_back(vd.scannerspeed);
        else
            out.scalars.push_back(vd.laserpower);
        out.segment_types.push_back(ordered_types[i]);
    }

    // Per-vertex RGB: contour = grey metallic, hatch path = pink, hatch signal = heatmap
    const float cR = 0.52f, cG = 0.54f, cB = 0.58f;
    const float pR = 0.9f, pG = 0.4f, pB = 0.6f;
    float v_min = std::numeric_limits<float>::max();
    float v_max = std::numeric_limits<float>::lowest();
    if (!path_only && n > 0) {
        for (size_t i = 0; i < n; ++i) {
            if (out.segment_types[i] == 0) {
                float s = out.scalars[i];
                if (s < v_min) v_min = s;
                if (s > v_max) v_max = s;
            }
        }
        if (v_max <= v_min) v_max = v_min + 1.f;
    }
    if (path_only || n == 0) {
        out.scalar_bar_min = std::numeric_limits<float>::quiet_NaN();
        out.scalar_bar_max = std::numeric_limits<float>::quiet_NaN();
    } else {
        out.scalar_bar_min = v_min;
        out.scalar_bar_max = v_max;
    }

    out.vertex_colors_rgb.reserve(6 * n);
    for (size_t i = 0; i < n; ++i) {
        int st = out.segment_types[i];
        float r, g, b;
        if (st == 1) {
            r = cR; g = cG; b = cB;
        } else if (path_only) {
            r = pR; g = pG; b = pB;
        } else {
            float t = (out.scalars[i] - v_min) / (v_max - v_min);
            r = 0.12f + 0.36f * t;
            g = 0.35f + 0.42f * t;
            b = 0.58f + 0.42f * t;
        }
        out.vertex_colors_rgb.push_back(r);
        out.vertex_colors_rgb.push_back(g);
        out.vertex_colors_rgb.push_back(b);
        out.vertex_colors_rgb.push_back(r);
        out.vertex_colors_rgb.push_back(g);
        out.vertex_colors_rgb.push_back(b);
    }

    return out;
}

}  // namespace visualization
}  // namespace am_qadf_native
