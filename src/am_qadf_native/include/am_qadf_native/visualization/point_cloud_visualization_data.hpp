#ifndef AM_QADF_NATIVE_VISUALIZATION_POINT_CLOUD_VISUALIZATION_DATA_HPP
#define AM_QADF_NATIVE_VISUALIZATION_POINT_CLOUD_VISUALIZATION_DATA_HPP

#include <string>
#include <vector>

namespace am_qadf_native {
namespace visualization {

/** Ready-to-visualize buffers for point cloud (laser_monitoring, ISPM): points + per-point scalar. */
struct PointCloudVisualizationResult {
    /** Flat positions: [x,y,z, x,y,z, ...]; length = 3 * n_points. */
    std::vector<float> positions;
    /** One scalar per point; length = n_points. */
    std::vector<float> scalars;
    /** Name of the scalar used (e.g. "actual_power", "melt_pool_temperature"). */
    std::string active_scalar_name;
    /** Per-point RGB for heatmap; length = 3 * n_points. */
    std::vector<float> vertex_colors_rgb;
    /** Heatmap scalar range min; NaN when empty. */
    float scalar_bar_min = 0.f;
    /** Heatmap scalar range max; NaN when empty. */
    float scalar_bar_max = 0.f;
};

/**
 * Build visualization buffers from laser_monitoring query (C++ to C++).
 * Calls query::MongoDBQueryClient::queryLaserMonitoringData, then fills positions[] and scalars[]
 * from points and laser_temporal_data. Python only builds point mesh and exports.
 *
 * @param model_id Model UUID
 * @param layer_start First layer index (-1 = no filter)
 * @param layer_end Last layer index (-1 = no filter)
 * @param scalar_name "actual_power", "commanded_power", "power_error", "timestamp", etc.
 * @param uri MongoDB connection URI
 * @param db_name MongoDB database name
 */
PointCloudVisualizationResult get_laser_monitoring_visualization_data(
    const std::string& model_id,
    int layer_start,
    int layer_end,
    const std::string& scalar_name,
    const std::string& uri,
    const std::string& db_name
);

/**
 * Build visualization buffers from ISPM query (C++ to C++).
 * Calls the appropriate query (queryISPMThermal, queryISPMOptical, etc.), then fills
 * positions[] and scalars[] from points and typed data. Python only builds point mesh and exports.
 *
 * @param model_id Model UUID
 * @param layer_start First layer index (-1 = no filter)
 * @param layer_end Last layer index (-1 = no filter)
 * @param source_type "ispm_thermal", "ispm_optical", "ispm_acoustic", "ispm_strain", "ispm_plume"
 * @param scalar_name Signal field name (e.g. "melt_pool_temperature", "photodiode_intensity")
 * @param uri MongoDB connection URI
 * @param db_name MongoDB database name
 */
PointCloudVisualizationResult get_ispm_visualization_data(
    const std::string& model_id,
    int layer_start,
    int layer_end,
    const std::string& source_type,
    const std::string& scalar_name,
    const std::string& uri,
    const std::string& db_name
);

}  // namespace visualization
}  // namespace am_qadf_native

#endif  // AM_QADF_NATIVE_VISUALIZATION_POINT_CLOUD_VISUALIZATION_DATA_HPP
