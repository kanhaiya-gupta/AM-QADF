#include "am_qadf_native/visualization/point_cloud_visualization_data.hpp"
#include "am_qadf_native/query/mongodb_query_client.hpp"
#include "am_qadf_native/query/query_result.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace am_qadf_native {
namespace visualization {

namespace {

const std::array<float, 3> kNoBboxMin{std::numeric_limits<float>::lowest(),
                                     std::numeric_limits<float>::lowest(),
                                     std::numeric_limits<float>::lowest()};
const std::array<float, 3> kNoBboxMax{std::numeric_limits<float>::max(),
                                     std::numeric_limits<float>::max(),
                                     std::numeric_limits<float>::max()};

float getLaserScalar(const query::QueryResult::LaserTemporalData& d, const std::string& name) {
    if (name == "actual_power") return d.actual_power;
    if (name == "commanded_power") return d.commanded_power;
    if (name == "power_setpoint") return d.power_setpoint;
    if (name == "power_error") return d.power_error;
    if (name == "commanded_scan_speed") return d.commanded_scan_speed;
    if (name == "power_stability") return d.power_stability;
    if (name == "power_fluctuation_amplitude") return d.power_fluctuation_amplitude;
    if (name == "power_fluctuation_frequency") return d.power_fluctuation_frequency;
    if (name == "pulse_frequency") return d.pulse_frequency;
    if (name == "pulse_duration") return d.pulse_duration;
    if (name == "pulse_energy") return d.pulse_energy;
    if (name == "duty_cycle") return d.duty_cycle;
    if (name == "laser_temperature") return d.laser_temperature;
    if (name == "laser_operating_hours") return d.laser_operating_hours;
    return d.actual_power;  // default
}

float getISPMThermalScalar(const query::QueryResult::ISPMThermalData& d, const std::string& name) {
    if (name == "melt_pool_temperature") return d.melt_pool_temperature;
    if (name == "peak_temperature") return d.peak_temperature;
    if (name == "melt_pool_width") return d.melt_pool_width;
    if (name == "melt_pool_length") return d.melt_pool_length;
    if (name == "melt_pool_area") return d.melt_pool_area;
    if (name == "cooling_rate") return d.cooling_rate;
    if (name == "temperature_gradient") return d.temperature_gradient;
    if (name == "time_over_threshold_1200K") return d.time_over_threshold_1200K;
    if (name == "time_over_threshold_1680K") return d.time_over_threshold_1680K;
    if (name == "time_over_threshold_2400K") return d.time_over_threshold_2400K;
    return d.melt_pool_temperature;
}

float getISPMOpticalScalar(const query::QueryResult::ISPMOpticalData& d, const std::string& name) {
    if (name == "photodiode_intensity") return d.photodiode_intensity;
    if (name == "melt_pool_brightness") return d.melt_pool_brightness;
    if (name == "melt_pool_intensity_mean") return d.melt_pool_intensity_mean;
    if (name == "melt_pool_intensity_max") return d.melt_pool_intensity_max;
    if (name == "spatter_intensity") return d.spatter_intensity;
    if (name == "process_stability") return d.process_stability;
    if (name == "signal_to_noise_ratio") return d.signal_to_noise_ratio;
    if (name == "keyhole_intensity") return d.keyhole_intensity;
    return d.photodiode_intensity;
}

float getISPMAcousticScalar(const query::QueryResult::ISPMAcousticData& d, const std::string& name) {
    if (name == "acoustic_amplitude") return d.acoustic_amplitude;
    if (name == "acoustic_rms") return d.acoustic_rms;
    if (name == "acoustic_peak") return d.acoustic_peak;
    if (name == "spectral_energy") return d.spectral_energy;
    if (name == "process_stability") return d.process_stability;
    if (name == "acoustic_energy") return d.acoustic_energy;
    return d.acoustic_amplitude;
}

float getISPMStrainScalar(const query::QueryResult::ISPMStrainData& d, const std::string& name) {
    if (name == "von_mises_strain") return d.von_mises_strain;
    if (name == "principal_strain_max") return d.principal_strain_max;
    if (name == "total_displacement") return d.total_displacement;
    if (name == "von_mises_stress") return d.von_mises_stress;
    if (name == "process_stability") return d.process_stability;
    if (name == "strain_energy_density") return d.strain_energy_density;
    return d.von_mises_strain;
}

float getISPMPlumeScalar(const query::QueryResult::ISPMPlumeData& d, const std::string& name) {
    if (name == "plume_intensity") return d.plume_intensity;
    if (name == "plume_temperature") return d.plume_temperature;
    if (name == "plume_velocity") return d.plume_velocity;
    if (name == "plume_area") return d.plume_area;
    if (name == "process_stability") return d.process_stability;
    if (name == "plume_energy") return d.plume_energy;
    return d.plume_intensity;
}

void fillPointCloudColors(
    size_t n,
    const std::vector<float>& scalars,
    std::vector<float>& vertex_colors_rgb,
    float& scalar_bar_min,
    float& scalar_bar_max
) {
    float v_min = std::numeric_limits<float>::max();
    float v_max = std::numeric_limits<float>::lowest();
    for (size_t i = 0; i < n; ++i) {
        float s = scalars[i];
        if (s < v_min) v_min = s;
        if (s > v_max) v_max = s;
    }
    if (n == 0 || v_max <= v_min) {
        scalar_bar_min = std::numeric_limits<float>::quiet_NaN();
        scalar_bar_max = std::numeric_limits<float>::quiet_NaN();
        return;
    }
    scalar_bar_min = v_min;
    scalar_bar_max = v_max;
    vertex_colors_rgb.reserve(3 * n);
    for (size_t i = 0; i < n; ++i) {
        float t = (scalars[i] - v_min) / (v_max - v_min);
        float r = 0.12f + 0.36f * t;
        float g = 0.35f + 0.42f * t;
        float b = 0.58f + 0.42f * t;
        vertex_colors_rgb.push_back(r);
        vertex_colors_rgb.push_back(g);
        vertex_colors_rgb.push_back(b);
    }
}

}  // namespace

PointCloudVisualizationResult get_laser_monitoring_visualization_data(
    const std::string& model_id,
    int layer_start,
    int layer_end,
    const std::string& scalar_name,
    const std::string& uri,
    const std::string& db_name
) {
    PointCloudVisualizationResult out;
    out.active_scalar_name = scalar_name.empty() ? "actual_power" : scalar_name;

    query::MongoDBQueryClient client(uri, db_name);
    query::QueryResult result = client.queryLaserMonitoringData(
        model_id, layer_start, layer_end, kNoBboxMin, kNoBboxMax);

    size_t n = result.points.size();
    if (n == 0) return out;
    if (result.laser_temporal_data.size() != n) return out;

    out.positions.reserve(3 * n);
    out.scalars.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        const auto& p = result.points[i];
        out.positions.push_back(p[0]);
        out.positions.push_back(p[1]);
        out.positions.push_back(p[2]);
        out.scalars.push_back(getLaserScalar(result.laser_temporal_data[i], out.active_scalar_name));
    }

    fillPointCloudColors(n, out.scalars, out.vertex_colors_rgb, out.scalar_bar_min, out.scalar_bar_max);
    return out;
}

PointCloudVisualizationResult get_ispm_visualization_data(
    const std::string& model_id,
    int layer_start,
    int layer_end,
    const std::string& source_type,
    const std::string& scalar_name,
    const std::string& uri,
    const std::string& db_name
) {
    PointCloudVisualizationResult out;
    query::MongoDBQueryClient client(uri, db_name);
    query::QueryResult result;

    if (source_type == "ispm_thermal") {
        result = client.queryISPMThermal(model_id, 0.0f, 0.0f, layer_start, layer_end, kNoBboxMin, kNoBboxMax);
        out.active_scalar_name = scalar_name.empty() ? "melt_pool_temperature" : scalar_name;
    } else if (source_type == "ispm_optical") {
        result = client.queryISPMOptical(model_id, 0.0f, 0.0f, layer_start, layer_end, kNoBboxMin, kNoBboxMax);
        out.active_scalar_name = scalar_name.empty() ? "photodiode_intensity" : scalar_name;
    } else if (source_type == "ispm_acoustic") {
        result = client.queryISPMAcoustic(model_id, 0.0f, 0.0f, layer_start, layer_end, kNoBboxMin, kNoBboxMax);
        out.active_scalar_name = scalar_name.empty() ? "acoustic_amplitude" : scalar_name;
    } else if (source_type == "ispm_strain") {
        result = client.queryISPMStrain(model_id, 0.0f, 0.0f, layer_start, layer_end, kNoBboxMin, kNoBboxMax);
        out.active_scalar_name = scalar_name.empty() ? "von_mises_strain" : scalar_name;
    } else if (source_type == "ispm_plume") {
        result = client.queryISPMPlume(model_id, 0.0f, 0.0f, layer_start, layer_end, kNoBboxMin, kNoBboxMax);
        out.active_scalar_name = scalar_name.empty() ? "plume_intensity" : scalar_name;
    } else {
        return out;
    }

    size_t n = result.points.size();
    if (n == 0) return out;

    out.positions.reserve(3 * n);
    out.scalars.reserve(n);

    if (source_type == "ispm_thermal" && result.ispm_thermal_data.size() == n) {
        for (size_t i = 0; i < n; ++i) {
            const auto& p = result.points[i];
            out.positions.push_back(p[0]);
            out.positions.push_back(p[1]);
            out.positions.push_back(p[2]);
            out.scalars.push_back(getISPMThermalScalar(result.ispm_thermal_data[i], out.active_scalar_name));
        }
    } else if (source_type == "ispm_optical" && result.ispm_optical_data.size() == n) {
        for (size_t i = 0; i < n; ++i) {
            const auto& p = result.points[i];
            out.positions.push_back(p[0]);
            out.positions.push_back(p[1]);
            out.positions.push_back(p[2]);
            out.scalars.push_back(getISPMOpticalScalar(result.ispm_optical_data[i], out.active_scalar_name));
        }
    } else if (source_type == "ispm_acoustic" && result.ispm_acoustic_data.size() == n) {
        for (size_t i = 0; i < n; ++i) {
            const auto& p = result.points[i];
            out.positions.push_back(p[0]);
            out.positions.push_back(p[1]);
            out.positions.push_back(p[2]);
            out.scalars.push_back(getISPMAcousticScalar(result.ispm_acoustic_data[i], out.active_scalar_name));
        }
    } else if (source_type == "ispm_strain" && result.ispm_strain_data.size() == n) {
        for (size_t i = 0; i < n; ++i) {
            const auto& p = result.points[i];
            out.positions.push_back(p[0]);
            out.positions.push_back(p[1]);
            out.positions.push_back(p[2]);
            out.scalars.push_back(getISPMStrainScalar(result.ispm_strain_data[i], out.active_scalar_name));
        }
    } else if (source_type == "ispm_plume" && result.ispm_plume_data.size() == n) {
        for (size_t i = 0; i < n; ++i) {
            const auto& p = result.points[i];
            out.positions.push_back(p[0]);
            out.positions.push_back(p[1]);
            out.positions.push_back(p[2]);
            out.scalars.push_back(getISPMPlumeScalar(result.ispm_plume_data[i], out.active_scalar_name));
        }
    } else {
        return out;
    }

    fillPointCloudColors(n, out.scalars, out.vertex_colors_rgb, out.scalar_bar_min, out.scalar_bar_max);
    return out;
}

}  // namespace visualization
}  // namespace am_qadf_native
