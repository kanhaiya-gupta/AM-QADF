#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include "am_qadf_native/query/mongodb_query_client.hpp"
#include "am_qadf_native/query/ct_image_reader.hpp"
#include "am_qadf_native/query/query_result.hpp"
#include "am_qadf_native/query/point_converter.hpp"
#include <limits>
#include <array>

namespace py = pybind11;
using namespace am_qadf_native::query;

void bind_query(py::module& m) {
    // Contour (nested class)
    py::class_<QueryResult::Contour>(m, "Contour")
        .def_readwrite("points", &QueryResult::Contour::points)
        .def_readwrite("sub_type", &QueryResult::Contour::sub_type)
        .def_readwrite("color", &QueryResult::Contour::color)
        .def_readwrite("linewidth", &QueryResult::Contour::linewidth);
    
    // HatchMetadata (nested class)
    py::class_<QueryResult::HatchMetadata>(m, "HatchMetadata")
        .def_readwrite("hatch_type", &QueryResult::HatchMetadata::hatch_type)
        .def_readwrite("laser_beam_width", &QueryResult::HatchMetadata::laser_beam_width)
        .def_readwrite("hatch_spacing", &QueryResult::HatchMetadata::hatch_spacing)
        .def_readwrite("overlap_percentage", &QueryResult::HatchMetadata::overlap_percentage);
    
    // Vector (nested class) - NEW: vector-based format
    py::class_<QueryResult::Vector>(m, "Vector")
        .def_readwrite("x1", &QueryResult::Vector::x1)
        .def_readwrite("y1", &QueryResult::Vector::y1)
        .def_readwrite("x2", &QueryResult::Vector::x2)
        .def_readwrite("y2", &QueryResult::Vector::y2)
        .def_readwrite("z", &QueryResult::Vector::z)
        .def_readwrite("timestamp", &QueryResult::Vector::timestamp)
        .def_readwrite("dataindex", &QueryResult::Vector::dataindex);
    
    // VectorData (nested class) - NEW: process parameters for vectors
    py::class_<QueryResult::VectorData>(m, "VectorData")
        .def_readwrite("dataindex", &QueryResult::VectorData::dataindex)
        .def_readwrite("partid", &QueryResult::VectorData::partid)
        .def_readwrite("type", &QueryResult::VectorData::type)
        .def_readwrite("scanner", &QueryResult::VectorData::scanner)
        .def_readwrite("laserpower", &QueryResult::VectorData::laserpower)
        .def_readwrite("scannerspeed", &QueryResult::VectorData::scannerspeed)
        .def_readwrite("laser_beam_width", &QueryResult::VectorData::laser_beam_width)
        .def_readwrite("hatch_spacing", &QueryResult::VectorData::hatch_spacing)
        .def_readwrite("layer_index", &QueryResult::VectorData::layer_index);
    
    // LaserTemporalData (nested class) - Temporal sensor fields (Category 3)
    // Based on sensor_fields_research.md - Category 3: Temporal Sensors (For Anomaly Detection)
    py::class_<QueryResult::LaserTemporalData>(m, "LaserTemporalData")
        // Process Parameters (setpoints/commanded values)
        .def_readwrite("commanded_power", &QueryResult::LaserTemporalData::commanded_power)
        .def_readwrite("commanded_scan_speed", &QueryResult::LaserTemporalData::commanded_scan_speed)
        // Category 3.1 - Laser Power Sensors (Temporal)
        .def_readwrite("actual_power", &QueryResult::LaserTemporalData::actual_power)
        .def_readwrite("power_setpoint", &QueryResult::LaserTemporalData::power_setpoint)
        .def_readwrite("power_error", &QueryResult::LaserTemporalData::power_error)
        .def_readwrite("power_stability", &QueryResult::LaserTemporalData::power_stability)
        .def_readwrite("power_fluctuation_amplitude", &QueryResult::LaserTemporalData::power_fluctuation_amplitude)
        .def_readwrite("power_fluctuation_frequency", &QueryResult::LaserTemporalData::power_fluctuation_frequency)
        // Category 3.2 - Beam Temporal Characteristics
        .def_readwrite("pulse_frequency", &QueryResult::LaserTemporalData::pulse_frequency)
        .def_readwrite("pulse_duration", &QueryResult::LaserTemporalData::pulse_duration)
        .def_readwrite("pulse_energy", &QueryResult::LaserTemporalData::pulse_energy)
        .def_readwrite("duty_cycle", &QueryResult::LaserTemporalData::duty_cycle)
        .def_readwrite("beam_modulation_frequency", &QueryResult::LaserTemporalData::beam_modulation_frequency)
        // Category 3.3 - Laser System Health (Temporal)
        .def_readwrite("laser_temperature", &QueryResult::LaserTemporalData::laser_temperature)
        .def_readwrite("laser_cooling_water_temp", &QueryResult::LaserTemporalData::laser_cooling_water_temp)
        .def_readwrite("laser_cooling_flow_rate", &QueryResult::LaserTemporalData::laser_cooling_flow_rate)
        .def_readwrite("laser_power_supply_voltage", &QueryResult::LaserTemporalData::laser_power_supply_voltage)
        .def_readwrite("laser_power_supply_current", &QueryResult::LaserTemporalData::laser_power_supply_current)
        .def_readwrite("laser_diode_current", &QueryResult::LaserTemporalData::laser_diode_current)
        .def_readwrite("laser_diode_temperature", &QueryResult::LaserTemporalData::laser_diode_temperature)
        .def_readwrite("laser_operating_hours", &QueryResult::LaserTemporalData::laser_operating_hours)
        .def_readwrite("laser_pulse_count", &QueryResult::LaserTemporalData::laser_pulse_count);
    
    // ISPMThermalData (nested class) - ISPM_Thermal-specific fields (all extracted in C++)
    // Based on research: melt pool-dependent and time-dependent features from thermograms
    py::class_<QueryResult::ISPMThermalData>(m, "ISPMThermalData")
        // Basic temperature fields
        .def_readwrite("melt_pool_temperature", &QueryResult::ISPMThermalData::melt_pool_temperature)  // MPTmean
        .def_readwrite("peak_temperature", &QueryResult::ISPMThermalData::peak_temperature)  // MPTmax
        // Melt pool geometry
        .def_readwrite("melt_pool_width", &QueryResult::ISPMThermalData::melt_pool_width)  // MPW
        .def_readwrite("melt_pool_length", &QueryResult::ISPMThermalData::melt_pool_length)  // MPL
        .def_readwrite("melt_pool_depth", &QueryResult::ISPMThermalData::melt_pool_depth)
        // Additional geometric fields (from research)
        .def_readwrite("melt_pool_area", &QueryResult::ISPMThermalData::melt_pool_area)  // MPA
        .def_readwrite("melt_pool_eccentricity", &QueryResult::ISPMThermalData::melt_pool_eccentricity)  // MPE
        .def_readwrite("melt_pool_perimeter", &QueryResult::ISPMThermalData::melt_pool_perimeter)  // MPP
        // Thermal parameters
        .def_readwrite("cooling_rate", &QueryResult::ISPMThermalData::cooling_rate)
        .def_readwrite("temperature_gradient", &QueryResult::ISPMThermalData::temperature_gradient)
        // Time over threshold metrics (TOT)
        .def_readwrite("time_over_threshold_1200K", &QueryResult::ISPMThermalData::time_over_threshold_1200K)  // TOT1200K
        .def_readwrite("time_over_threshold_1680K", &QueryResult::ISPMThermalData::time_over_threshold_1680K)  // TOT1680K
        .def_readwrite("time_over_threshold_2400K", &QueryResult::ISPMThermalData::time_over_threshold_2400K)  // TOT2400K
        // Process events
        .def_readwrite("process_event", &QueryResult::ISPMThermalData::process_event);
    
    // ISPMOpticalData (nested class) - ISPM_Optical-specific fields (all extracted in C++)
    // Based on PBF-LB/M optical monitoring: photodiodes, cameras, melt pool imaging
    py::class_<QueryResult::ISPMOpticalData>(m, "ISPMOpticalData")
        // Photodiode signals
        .def_readwrite("photodiode_intensity", &QueryResult::ISPMOpticalData::photodiode_intensity)
        .def_readwrite("photodiode_frequency", &QueryResult::ISPMOpticalData::photodiode_frequency)
        .def_readwrite("photodiode_coaxial", &QueryResult::ISPMOpticalData::photodiode_coaxial)
        .def_readwrite("photodiode_off_axis", &QueryResult::ISPMOpticalData::photodiode_off_axis)
        // Melt pool brightness/intensity
        .def_readwrite("melt_pool_brightness", &QueryResult::ISPMOpticalData::melt_pool_brightness)
        .def_readwrite("melt_pool_intensity_mean", &QueryResult::ISPMOpticalData::melt_pool_intensity_mean)
        .def_readwrite("melt_pool_intensity_max", &QueryResult::ISPMOpticalData::melt_pool_intensity_max)
        .def_readwrite("melt_pool_intensity_std", &QueryResult::ISPMOpticalData::melt_pool_intensity_std)
        // Spatter detection
        .def_readwrite("spatter_detected", &QueryResult::ISPMOpticalData::spatter_detected)
        .def_readwrite("spatter_intensity", &QueryResult::ISPMOpticalData::spatter_intensity)
        .def_readwrite("spatter_count", &QueryResult::ISPMOpticalData::spatter_count)
        // Process stability metrics
        .def_readwrite("process_stability", &QueryResult::ISPMOpticalData::process_stability)
        .def_readwrite("intensity_variation", &QueryResult::ISPMOpticalData::intensity_variation)
        .def_readwrite("signal_to_noise_ratio", &QueryResult::ISPMOpticalData::signal_to_noise_ratio)
        // Melt pool imaging
        .def_readwrite("melt_pool_image_available", &QueryResult::ISPMOpticalData::melt_pool_image_available)
        .def_readwrite("melt_pool_area_pixels", &QueryResult::ISPMOpticalData::melt_pool_area_pixels)
        .def_readwrite("melt_pool_centroid_x", &QueryResult::ISPMOpticalData::melt_pool_centroid_x)
        .def_readwrite("melt_pool_centroid_y", &QueryResult::ISPMOpticalData::melt_pool_centroid_y)
        // Keyhole detection
        .def_readwrite("keyhole_detected", &QueryResult::ISPMOpticalData::keyhole_detected)
        .def_readwrite("keyhole_intensity", &QueryResult::ISPMOpticalData::keyhole_intensity)
        // Process events
        .def_readwrite("process_event", &QueryResult::ISPMOpticalData::process_event)
        // Frequency domain features
        .def_readwrite("dominant_frequency", &QueryResult::ISPMOpticalData::dominant_frequency)
        .def_readwrite("frequency_bandwidth", &QueryResult::ISPMOpticalData::frequency_bandwidth)
        .def_readwrite("spectral_energy", &QueryResult::ISPMOpticalData::spectral_energy);
    
    // ISPMAcousticData (nested class) - ISPM_Acoustic-specific fields (all extracted in C++)
    // Based on PBF-LB/M acoustic monitoring: acoustic emissions, sound sensors, event detection
    py::class_<QueryResult::ISPMAcousticData>(m, "ISPMAcousticData")
        // Acoustic emission signals
        .def_readwrite("acoustic_amplitude", &QueryResult::ISPMAcousticData::acoustic_amplitude)
        .def_readwrite("acoustic_frequency", &QueryResult::ISPMAcousticData::acoustic_frequency)
        .def_readwrite("acoustic_rms", &QueryResult::ISPMAcousticData::acoustic_rms)
        .def_readwrite("acoustic_peak", &QueryResult::ISPMAcousticData::acoustic_peak)
        // Frequency domain features
        .def_readwrite("dominant_frequency", &QueryResult::ISPMAcousticData::dominant_frequency)
        .def_readwrite("frequency_bandwidth", &QueryResult::ISPMAcousticData::frequency_bandwidth)
        .def_readwrite("spectral_centroid", &QueryResult::ISPMAcousticData::spectral_centroid)
        .def_readwrite("spectral_energy", &QueryResult::ISPMAcousticData::spectral_energy)
        .def_readwrite("spectral_rolloff", &QueryResult::ISPMAcousticData::spectral_rolloff)
        // Event detection
        .def_readwrite("spatter_event_detected", &QueryResult::ISPMAcousticData::spatter_event_detected)
        .def_readwrite("spatter_event_amplitude", &QueryResult::ISPMAcousticData::spatter_event_amplitude)
        .def_readwrite("defect_event_detected", &QueryResult::ISPMAcousticData::defect_event_detected)
        .def_readwrite("defect_event_amplitude", &QueryResult::ISPMAcousticData::defect_event_amplitude)
        .def_readwrite("anomaly_detected", &QueryResult::ISPMAcousticData::anomaly_detected)
        .def_readwrite("anomaly_type", &QueryResult::ISPMAcousticData::anomaly_type)
        // Process stability metrics
        .def_readwrite("process_stability", &QueryResult::ISPMAcousticData::process_stability)
        .def_readwrite("acoustic_variation", &QueryResult::ISPMAcousticData::acoustic_variation)
        .def_readwrite("signal_to_noise_ratio", &QueryResult::ISPMAcousticData::signal_to_noise_ratio)
        // Time-domain features
        .def_readwrite("zero_crossing_rate", &QueryResult::ISPMAcousticData::zero_crossing_rate)
        .def_readwrite("autocorrelation_peak", &QueryResult::ISPMAcousticData::autocorrelation_peak)
        // Frequency-domain features
        .def_readwrite("harmonic_ratio", &QueryResult::ISPMAcousticData::harmonic_ratio)
        .def_readwrite("spectral_flatness", &QueryResult::ISPMAcousticData::spectral_flatness)
        .def_readwrite("spectral_crest", &QueryResult::ISPMAcousticData::spectral_crest)
        // Process events
        .def_readwrite("process_event", &QueryResult::ISPMAcousticData::process_event)
        // Acoustic energy
        .def_readwrite("acoustic_energy", &QueryResult::ISPMAcousticData::acoustic_energy);
    
    // ISPM_Strain Data binding
    py::class_<QueryResult::ISPMStrainData>(m, "ISPMStrainData")
        // Strain components
        .def_readwrite("strain_xx", &QueryResult::ISPMStrainData::strain_xx)
        .def_readwrite("strain_yy", &QueryResult::ISPMStrainData::strain_yy)
        .def_readwrite("strain_zz", &QueryResult::ISPMStrainData::strain_zz)
        .def_readwrite("strain_xy", &QueryResult::ISPMStrainData::strain_xy)
        .def_readwrite("strain_xz", &QueryResult::ISPMStrainData::strain_xz)
        .def_readwrite("strain_yz", &QueryResult::ISPMStrainData::strain_yz)
        // Principal strains
        .def_readwrite("principal_strain_max", &QueryResult::ISPMStrainData::principal_strain_max)
        .def_readwrite("principal_strain_min", &QueryResult::ISPMStrainData::principal_strain_min)
        .def_readwrite("principal_strain_intermediate", &QueryResult::ISPMStrainData::principal_strain_intermediate)
        // Von Mises strain
        .def_readwrite("von_mises_strain", &QueryResult::ISPMStrainData::von_mises_strain)
        // Displacement
        .def_readwrite("displacement_x", &QueryResult::ISPMStrainData::displacement_x)
        .def_readwrite("displacement_y", &QueryResult::ISPMStrainData::displacement_y)
        .def_readwrite("displacement_z", &QueryResult::ISPMStrainData::displacement_z)
        .def_readwrite("total_displacement", &QueryResult::ISPMStrainData::total_displacement)
        // Strain rate
        .def_readwrite("strain_rate", &QueryResult::ISPMStrainData::strain_rate)
        // Residual stress
        .def_readwrite("residual_stress_xx", &QueryResult::ISPMStrainData::residual_stress_xx)
        .def_readwrite("residual_stress_yy", &QueryResult::ISPMStrainData::residual_stress_yy)
        .def_readwrite("residual_stress_zz", &QueryResult::ISPMStrainData::residual_stress_zz)
        .def_readwrite("von_mises_stress", &QueryResult::ISPMStrainData::von_mises_stress)
        // Temperature-compensated strain
        .def_readwrite("temperature_compensated_strain", &QueryResult::ISPMStrainData::temperature_compensated_strain)
        // Warping/distortion
        .def_readwrite("warping_detected", &QueryResult::ISPMStrainData::warping_detected)
        .def_readwrite("warping_magnitude", &QueryResult::ISPMStrainData::warping_magnitude)
        .def_readwrite("distortion_angle", &QueryResult::ISPMStrainData::distortion_angle)
        // Layer-wise strain accumulation
        .def_readwrite("cumulative_strain", &QueryResult::ISPMStrainData::cumulative_strain)
        .def_readwrite("layer_strain_increment", &QueryResult::ISPMStrainData::layer_strain_increment)
        // Event detection
        .def_readwrite("excessive_strain_event", &QueryResult::ISPMStrainData::excessive_strain_event)
        .def_readwrite("warping_event_detected", &QueryResult::ISPMStrainData::warping_event_detected)
        .def_readwrite("distortion_event_detected", &QueryResult::ISPMStrainData::distortion_event_detected)
        .def_readwrite("anomaly_detected", &QueryResult::ISPMStrainData::anomaly_detected)
        .def_readwrite("anomaly_type", &QueryResult::ISPMStrainData::anomaly_type)
        // Process stability metrics
        .def_readwrite("process_stability", &QueryResult::ISPMStrainData::process_stability)
        .def_readwrite("strain_variation", &QueryResult::ISPMStrainData::strain_variation)
        .def_readwrite("strain_uniformity", &QueryResult::ISPMStrainData::strain_uniformity)
        // Process events
        .def_readwrite("process_event", &QueryResult::ISPMStrainData::process_event)
        // Strain energy
        .def_readwrite("strain_energy_density", &QueryResult::ISPMStrainData::strain_energy_density);
    
    // ISPM_Plume Data binding
    py::class_<QueryResult::ISPMPlumeData>(m, "ISPMPlumeData")
        // Plume characteristics
        .def_readwrite("plume_intensity", &QueryResult::ISPMPlumeData::plume_intensity)
        .def_readwrite("plume_density", &QueryResult::ISPMPlumeData::plume_density)
        .def_readwrite("plume_temperature", &QueryResult::ISPMPlumeData::plume_temperature)
        .def_readwrite("plume_velocity", &QueryResult::ISPMPlumeData::plume_velocity)
        .def_readwrite("plume_velocity_x", &QueryResult::ISPMPlumeData::plume_velocity_x)
        .def_readwrite("plume_velocity_y", &QueryResult::ISPMPlumeData::plume_velocity_y)
        // Plume geometry
        .def_readwrite("plume_height", &QueryResult::ISPMPlumeData::plume_height)
        .def_readwrite("plume_width", &QueryResult::ISPMPlumeData::plume_width)
        .def_readwrite("plume_angle", &QueryResult::ISPMPlumeData::plume_angle)
        .def_readwrite("plume_spread", &QueryResult::ISPMPlumeData::plume_spread)
        .def_readwrite("plume_area", &QueryResult::ISPMPlumeData::plume_area)
        // Plume composition
        .def_readwrite("particle_concentration", &QueryResult::ISPMPlumeData::particle_concentration)
        .def_readwrite("metal_vapor_concentration", &QueryResult::ISPMPlumeData::metal_vapor_concentration)
        .def_readwrite("gas_composition_ratio", &QueryResult::ISPMPlumeData::gas_composition_ratio)
        // Plume dynamics
        .def_readwrite("plume_fluctuation_rate", &QueryResult::ISPMPlumeData::plume_fluctuation_rate)
        .def_readwrite("plume_instability_index", &QueryResult::ISPMPlumeData::plume_instability_index)
        .def_readwrite("plume_turbulence", &QueryResult::ISPMPlumeData::plume_turbulence)
        // Process quality indicators
        .def_readwrite("process_stability", &QueryResult::ISPMPlumeData::process_stability)
        .def_readwrite("plume_stability", &QueryResult::ISPMPlumeData::plume_stability)
        .def_readwrite("intensity_variation", &QueryResult::ISPMPlumeData::intensity_variation)
        // Event detection
        .def_readwrite("excessive_plume_event", &QueryResult::ISPMPlumeData::excessive_plume_event)
        .def_readwrite("unstable_plume_event", &QueryResult::ISPMPlumeData::unstable_plume_event)
        .def_readwrite("contamination_event", &QueryResult::ISPMPlumeData::contamination_event)
        .def_readwrite("anomaly_detected", &QueryResult::ISPMPlumeData::anomaly_detected)
        .def_readwrite("anomaly_type", &QueryResult::ISPMPlumeData::anomaly_type)
        // Plume energy metrics
        .def_readwrite("plume_energy", &QueryResult::ISPMPlumeData::plume_energy)
        .def_readwrite("energy_density", &QueryResult::ISPMPlumeData::energy_density)
        // Process events
        .def_readwrite("process_event", &QueryResult::ISPMPlumeData::process_event)
        // Signal quality
        .def_readwrite("signal_to_noise_ratio", &QueryResult::ISPMPlumeData::signal_to_noise_ratio)
        // Additional plume features
        .def_readwrite("plume_momentum", &QueryResult::ISPMPlumeData::plume_momentum)
        .def_readwrite("plume_pressure", &QueryResult::ISPMPlumeData::plume_pressure);
    
    // QueryResult
    py::class_<QueryResult>(m, "QueryResult")
        .def(py::init<>())
        .def_readwrite("points", &QueryResult::points)
        .def_readwrite("values", &QueryResult::values)
        .def_readwrite("timestamps", &QueryResult::timestamps)
        .def_readwrite("layers", &QueryResult::layers)
        .def_readwrite("hatch_metadata", &QueryResult::hatch_metadata)
        .def_readwrite("hatch_start_indices", &QueryResult::hatch_start_indices)
        .def_readwrite("contours", &QueryResult::contours)
        .def_readwrite("contour_layers", &QueryResult::contour_layers)
        .def_readwrite("vectors", &QueryResult::vectors)  // NEW: vector-based format
        .def_readwrite("vectordata", &QueryResult::vectordata)  // NEW: process parameters
        .def_readwrite("laser_temporal_data", &QueryResult::laser_temporal_data)  // Temporal sensor fields (Category 3)
        .def_readwrite("ispm_thermal_data", &QueryResult::ispm_thermal_data)  // ISPM_Thermal-specific fields (all extracted in C++)
        .def_readwrite("ispm_optical_data", &QueryResult::ispm_optical_data)  // ISPM_Optical-specific fields (all extracted in C++)
        .def_readwrite("ispm_acoustic_data", &QueryResult::ispm_acoustic_data)  // ISPM_Acoustic-specific fields (all extracted in C++)
        .def_readwrite("ispm_strain_data", &QueryResult::ispm_strain_data)  // ISPM_Strain-specific fields (all extracted in C++)
        .def_readwrite("ispm_plume_data", &QueryResult::ispm_plume_data)  // ISPM_Plume-specific fields (all extracted in C++)
        .def_readwrite("model_id", &QueryResult::model_id)
        .def_readwrite("signal_type", &QueryResult::signal_type)
        .def_readwrite("format", &QueryResult::format)  // NEW: "vector-based" (hatching) or "point-based" (other data)
        .def("num_points", &QueryResult::num_points)  // For point-based data (laser_monitoring_data, CT, ISPM)
        .def("num_vectors", &QueryResult::num_vectors)  // For vector-based hatching data
        .def("empty", &QueryResult::empty)  // Checks vectors for hatching, points for other data
        .def("has_multiple_signals", &QueryResult::has_multiple_signals)
        .def("has_contours", &QueryResult::has_contours);
    
    // MongoDBQueryClient
    py::class_<MongoDBQueryClient>(m, "MongoDBQueryClient")
        .def(py::init<const std::string&, const std::string&>())
        .def("query_laser_monitoring_data", &MongoDBQueryClient::queryLaserMonitoringData,
             py::arg("model_id"),
             py::arg("layer_start") = -1,
             py::arg("layer_end") = -1,
             py::arg("bbox_min") = std::array<float, 3>{std::numeric_limits<float>::lowest(), 
                                                          std::numeric_limits<float>::lowest(), 
                                                          std::numeric_limits<float>::lowest()},
             py::arg("bbox_max") = std::array<float, 3>{std::numeric_limits<float>::max(), 
                                                          std::numeric_limits<float>::max(), 
                                                          std::numeric_limits<float>::max()})
        .def("query_ispm_thermal", &MongoDBQueryClient::queryISPMThermal,
             py::arg("model_id"),
             py::arg("time_start") = 0.0f,
             py::arg("time_end") = 0.0f,
             py::arg("layer_start") = -1,
             py::arg("layer_end") = -1,
             py::arg("bbox_min") = std::array<float, 3>{std::numeric_limits<float>::lowest(),
                                                         std::numeric_limits<float>::lowest(),
                                                         std::numeric_limits<float>::lowest()},
             py::arg("bbox_max") = std::array<float, 3>{std::numeric_limits<float>::max(),
                                                         std::numeric_limits<float>::max(),
                                                         std::numeric_limits<float>::max()})
        .def("query_ispm_optical", &MongoDBQueryClient::queryISPMOptical,
             py::arg("model_id"),
             py::arg("time_start") = 0.0f,
             py::arg("time_end") = 0.0f,
             py::arg("layer_start") = -1,
             py::arg("layer_end") = -1,
             py::arg("bbox_min") = std::array<float, 3>{std::numeric_limits<float>::lowest(),
                                                         std::numeric_limits<float>::lowest(),
                                                         std::numeric_limits<float>::lowest()},
             py::arg("bbox_max") = std::array<float, 3>{std::numeric_limits<float>::max(),
                                                         std::numeric_limits<float>::max(),
                                                         std::numeric_limits<float>::max()})
        .def("query_ispm_acoustic", &MongoDBQueryClient::queryISPMAcoustic,
             py::arg("model_id"),
             py::arg("time_start") = 0.0f,
             py::arg("time_end") = 0.0f,
             py::arg("layer_start") = -1,
             py::arg("layer_end") = -1,
             py::arg("bbox_min") = std::array<float, 3>{std::numeric_limits<float>::lowest(),
                                                         std::numeric_limits<float>::lowest(),
                                                         std::numeric_limits<float>::lowest()},
             py::arg("bbox_max") = std::array<float, 3>{std::numeric_limits<float>::max(),
                                                         std::numeric_limits<float>::max(),
                                                         std::numeric_limits<float>::max()})
        .def("query_ispm_strain", &MongoDBQueryClient::queryISPMStrain,
             py::arg("model_id"),
             py::arg("time_start") = 0.0f,
             py::arg("time_end") = 0.0f,
             py::arg("layer_start") = -1,
             py::arg("layer_end") = -1,
             py::arg("bbox_min") = std::array<float, 3>{std::numeric_limits<float>::lowest(),
                                                         std::numeric_limits<float>::lowest(),
                                                         std::numeric_limits<float>::lowest()},
             py::arg("bbox_max") = std::array<float, 3>{std::numeric_limits<float>::max(),
                                                         std::numeric_limits<float>::max(),
                                                         std::numeric_limits<float>::max()})
        .def("query_ispm_plume", &MongoDBQueryClient::queryISPMPlume,
             py::arg("model_id"),
             py::arg("time_start") = 0.0f,
             py::arg("time_end") = 0.0f,
             py::arg("layer_start") = -1,
             py::arg("layer_end") = -1,
             py::arg("bbox_min") = std::array<float, 3>{std::numeric_limits<float>::lowest(),
                                                         std::numeric_limits<float>::lowest(),
                                                         std::numeric_limits<float>::lowest()},
             py::arg("bbox_max") = std::array<float, 3>{std::numeric_limits<float>::max(),
                                                         std::numeric_limits<float>::max(),
                                                         std::numeric_limits<float>::max()})
        .def("query_ct_scan", &MongoDBQueryClient::queryCTScan)
        .def("query_hatching_data", &MongoDBQueryClient::queryHatchingData,
             py::arg("model_id"),
             py::arg("layer_start") = -1,
             py::arg("layer_end") = -1,
             py::arg("bbox_min") = std::array<float, 3>{std::numeric_limits<float>::lowest(),
                                                         std::numeric_limits<float>::lowest(),
                                                         std::numeric_limits<float>::lowest()},
             py::arg("bbox_max") = std::array<float, 3>{std::numeric_limits<float>::max(),
                                                         std::numeric_limits<float>::max(),
                                                         std::numeric_limits<float>::max()});
    
    // CTImageReader
    py::class_<CTImageReader>(m, "CTImageReader")
        .def(py::init<>())
        .def("read_dicom_series", &CTImageReader::readDICOMSeries)
        .def("read_nifti", &CTImageReader::readNIfTI)
        .def("read_tiff_stack", &CTImageReader::readTIFFStack)
        .def("read_raw_binary", &CTImageReader::readRawBinary);
    
    // ============================================
    // PointConverter (Helper functions)
    // ============================================
    
    m.def("points_to_eigen_matrix", &am_qadf_native::query::pointsToEigenMatrix,
          py::arg("points"),
          "Convert std::vector<std::array<float, 3>> to Eigen::MatrixXd (n_points, 3)");
    
    m.def("eigen_matrix_to_points", &am_qadf_native::query::eigenMatrixToPoints,
          py::arg("matrix"),
          "Convert Eigen::MatrixXd to std::vector<std::array<float, 3>>");
    
    m.def("query_result_to_eigen_matrix", &am_qadf_native::query::queryResultToEigenMatrix,
          py::arg("result"),
          "Extract points from QueryResult and convert to Eigen::MatrixXd");
}
