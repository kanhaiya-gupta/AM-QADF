#ifndef AM_QADF_NATIVE_VISUALIZATION_HATCHING_VISUALIZATION_DATA_HPP
#define AM_QADF_NATIVE_VISUALIZATION_HATCHING_VISUALIZATION_DATA_HPP

#include <string>
#include <vector>

namespace am_qadf_native {
namespace visualization {

/** Segment type: 0 = hatch (inside), 1 = contour (outer/inner boundary). */
enum SegmentType { Hatch = 0, Contour = 1 };

/** Ready-to-visualize buffers for hatching line plot (C++ only; no Python fallback). */
struct HatchingVisualizationResult {
    /** Flat positions: [x1,y1,z1, x2,y2,z2, ...] per segment; length = 6 * n_segments. */
    std::vector<float> positions;
    /** One scalar per segment (e.g. laser_power, scan_speed, length); length = n_segments. */
    std::vector<float> scalars;
    /** One type per segment: 0 = hatch, 1 = contour; length = n_segments. */
    std::vector<int> segment_types;
    /** Name of the scalar used (e.g. "laser_power", "scan_speed", "length"). */
    std::string active_scalar_name;
    /** Per-vertex RGB: length = 6 * n_segments (r,g,b for each segment vertex). */
    std::vector<float> vertex_colors_rgb;
    /** Heatmap scalar range min; NaN when path-only or empty. */
    float scalar_bar_min = 0.f;
    /** Heatmap scalar range max; NaN when path-only or empty. */
    float scalar_bar_max = 0.f;
};

/**
 * Build visualization buffers from hatching query (C++ to C++).
 * Calls query::MongoDBQueryClient::queryHatchingData, then sorts by type and spatial order,
 * and fills positions[] and scalars[]. Python only uses these arrays to build PyVista mesh.
 *
 * @param model_id Model UUID
 * @param layer_start First layer index (-1 = no filter)
 * @param layer_end Last layer index (-1 = no filter)
 * @param scalar_name "laser_power", "scan_speed", "length", or "path"/"none" for path-only
 * @param uri MongoDB connection URI
 * @param db_name MongoDB database name
 * @return positions, scalars, segment_types, vertex_colors_rgb, active_scalar_name, scalar_bar_min/max
 */
HatchingVisualizationResult get_hatching_visualization_data(
    const std::string& model_id,
    int layer_start,
    int layer_end,
    const std::string& scalar_name,
    const std::string& uri,
    const std::string& db_name
);

}  // namespace visualization
}  // namespace am_qadf_native

#endif  // AM_QADF_NATIVE_VISUALIZATION_HATCHING_VISUALIZATION_DATA_HPP
