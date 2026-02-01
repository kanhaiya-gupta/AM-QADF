#ifndef AM_QADF_NATIVE_QUERY_QUERY_RESULT_HPP
#define AM_QADF_NATIVE_QUERY_QUERY_RESULT_HPP

#include <vector>
#include <array>
#include <string>

namespace am_qadf_native {
namespace query {

// Query result structure
struct QueryResult {
    // Point-based data (for laser_monitoring_data, CT scan, ISPM - NOT for hatching)
    // NOTE: For hatching queries, use vectors/vectordata instead (see below)
    std::vector<std::array<float, 3>> points;  // [x, y, z] coordinates (legacy/other data types)
    std::vector<float> values;                  // Signal values (single signal)
    std::vector<float> timestamps;              // Timestamps
    std::vector<int> layers;                    // Layer indices
    
    // ISPM_Thermal (In-Situ Process Monitoring - Thermal) specific fields
    // Based on research: melt pool-dependent and time-dependent features from thermograms
    struct ISPMThermalData {
        // Basic temperature fields
        float melt_pool_temperature;     // MPTmean - Melt pool mean temperature (Celsius)
        float peak_temperature;          // MPTmax - Melt pool maximum temperature (Celsius)
        
        // Melt pool geometry (from melt_pool_size dict)
        float melt_pool_width;           // MPW - Minor axis of elliptical fit (mm)
        float melt_pool_length;          // MPL - Major axis of elliptical fit (mm)
        float melt_pool_depth;            // Melt pool depth (mm) - additional field
        
        // Additional geometric fields (from research)
        float melt_pool_area;            // MPA - Melt pool area (mm²)
        float melt_pool_eccentricity;    // MPE - Ratio of MPW to MPL (dimensionless)
        float melt_pool_perimeter;       // MPP - Melt pool perimeter (mm)
        
        // Thermal parameters
        float cooling_rate;               // Cooling rate (K/s)
        float temperature_gradient;      // Temperature gradient (K/mm)
        
        // Time over threshold metrics (TOT) - cooling behavior
        float time_over_threshold_1200K;  // TOT1200K - Time above camera sensitivity (ms)
        float time_over_threshold_1680K;  // TOT1680K - Time above solidification ~1660K (ms)
        float time_over_threshold_2400K;  // TOT2400K - Time above upper threshold (ms)
        
        // Process events
        std::string process_event;        // Process event (e.g., "layer_start", "hatch_complete")
    };
    std::vector<ISPMThermalData> ispm_thermal_data;     // ISPM_Thermal-specific data for each point
    
    // ISPM_Optical (In-Situ Process Monitoring - Optical) specific fields
    // Based on PBF-LB/M optical monitoring: photodiodes, cameras, melt pool imaging
    struct ISPMOpticalData {
        // Photodiode signals (primary optical monitoring in PBF-LB/M)
        float photodiode_intensity;      // Arbitrary units (0-10000 typical range)
        float photodiode_frequency;      // Hz (dominant frequency component)
        float photodiode_coaxial;        // Coaxial photodiode signal (if available, -1 if not)
        float photodiode_off_axis;       // Off-axis photodiode signal (if available, -1 if not)
        
        // Melt pool brightness/intensity
        float melt_pool_brightness;      // Arbitrary units (0-10000 typical range)
        float melt_pool_intensity_mean;  // Mean intensity value
        float melt_pool_intensity_max;   // Peak intensity value
        float melt_pool_intensity_std;   // Standard deviation (stability indicator)
        
        // Spatter detection (critical for PBF-LB/M quality)
        bool spatter_detected;           // Boolean flag
        float spatter_intensity;         // Intensity spike when spatter occurs (-1 if not detected)
        int spatter_count;               // Number of spatter events in time window
        
        // Process stability metrics
        float process_stability;         // 0-1 scale (1 = stable, 0 = unstable)
        float intensity_variation;        // Coefficient of variation (std/mean)
        float signal_to_noise_ratio;     // SNR in dB
        
        // Melt pool imaging (if camera-based system)
        bool melt_pool_image_available;  // Boolean flag
        int melt_pool_area_pixels;       // Melt pool area in pixels (-1 if not available)
        float melt_pool_centroid_x;       // Centroid X in pixels (-1 if not available)
        float melt_pool_centroid_y;      // Centroid Y in pixels (-1 if not available)
        
        // Keyhole detection (important for porosity prediction)
        bool keyhole_detected;           // Boolean flag
        float keyhole_intensity;         // Intensity spike indicating keyhole (-1 if not detected)
        
        // Process events
        std::string process_event;        // Process event (e.g., "layer_start", "spatter_event", "keyhole_event")
        
        // Frequency domain features (for advanced analysis)
        float dominant_frequency;         // Hz (-1 if not available)
        float frequency_bandwidth;        // Hz (spectral width, -1 if not available)
        float spectral_energy;            // Total spectral energy (-1 if not available)
    };
    std::vector<ISPMOpticalData> ispm_optical_data;     // ISPM_Optical-specific data for each point
    
    // ISPM_Acoustic (In-Situ Process Monitoring - Acoustic) specific fields
    // Based on PBF-LB/M acoustic monitoring: acoustic emissions, sound sensors, event detection
    struct ISPMAcousticData {
        // Acoustic emission signals (primary acoustic monitoring in PBF-LB/M)
        float acoustic_amplitude;      // Arbitrary units (0-10000 typical range)
        float acoustic_frequency;      // Hz (dominant frequency component)
        float acoustic_rms;            // Root mean square amplitude
        float acoustic_peak;           // Peak amplitude value
        
        // Frequency domain features
        float dominant_frequency;       // Hz (primary frequency component)
        float frequency_bandwidth;       // Hz (spectral width)
        float spectral_centroid;        // Hz (weighted average frequency)
        float spectral_energy;           // Total spectral energy
        float spectral_rolloff;         // Hz (frequency below which 85% of energy is contained, -1 if not available)
        
        // Event detection (critical for PBF-LB/M quality)
        bool spatter_event_detected;     // Boolean flag
        float spatter_event_amplitude;   // Amplitude spike when spatter occurs (-1 if not detected)
        bool defect_event_detected;      // Boolean flag
        float defect_event_amplitude;    // Amplitude spike when defect forms (-1 if not detected)
        bool anomaly_detected;           // Boolean flag
        std::string anomaly_type;        // Anomaly type (e.g., "lack_of_fusion", "keyhole_instability")
        
        // Process stability metrics
        float process_stability;         // 0-1 scale (1 = stable, 0 = unstable)
        float acoustic_variation;        // Coefficient of variation (std/mean)
        float signal_to_noise_ratio;     // SNR in dB
        
        // Time-domain features
        float zero_crossing_rate;        // Zero crossings per second (-1 if not available)
        float autocorrelation_peak;      // Peak autocorrelation value (-1 if not available)
        
        // Frequency-domain features (advanced analysis)
        float harmonic_ratio;            // Ratio of harmonic to fundamental energy (-1 if not available)
        float spectral_flatness;         // Measure of spectral uniformity 0-1 (-1 if not available)
        float spectral_crest;            // Peak-to-average ratio in frequency domain (-1 if not available)
        
        // Process events
        std::string process_event;       // Process event (e.g., "layer_start", "spatter_event", "defect_event")
        
        // Acoustic energy metrics
        float acoustic_energy;           // Total acoustic energy
    };
    std::vector<ISPMAcousticData> ispm_acoustic_data;     // ISPM_Acoustic-specific data for each point
    
    // ISPM_Strain (In-Situ Process Monitoring - Strain) Data
    struct ISPMStrainData {
        // Strain components (microstrain, typically -5000 to +5000 με)
        float strain_xx;                  // Normal strain in x-direction (με)
        float strain_yy;                  // Normal strain in y-direction (με)
        float strain_zz;                  // Normal strain in z-direction (με)
        float strain_xy;                  // Shear strain in xy-plane (με)
        float strain_xz;                  // Shear strain in xz-plane (με)
        float strain_yz;                  // Shear strain in yz-plane (με)
        
        // Principal strains (derived from strain tensor)
        float principal_strain_max;       // Maximum principal strain (με)
        float principal_strain_min;        // Minimum principal strain (με)
        float principal_strain_intermediate;  // Intermediate principal strain (με)
        
        // Equivalent/von Mises strain (important for yield/failure prediction)
        float von_mises_strain;           // Equivalent strain (με)
        
        // Deformation/displacement (mm)
        float displacement_x;              // Displacement in x-direction (mm)
        float displacement_y;              // Displacement in y-direction (mm)
        float displacement_z;              // Displacement in z-direction (mm)
        float total_displacement;          // Total displacement magnitude (mm)
        
        // Strain rate (με/s)
        float strain_rate;                // Rate of change of equivalent strain
        
        // Residual stress indicators (MPa, estimated from strain)
        float residual_stress_xx;          // Residual stress in x-direction (MPa, -1 if not available)
        float residual_stress_yy;          // Residual stress in y-direction (MPa, -1 if not available)
        float residual_stress_zz;          // Residual stress in z-direction (MPa, -1 if not available)
        float von_mises_stress;            // Equivalent von Mises stress (MPa, -1 if not available)
        
        // Temperature-compensated strain (με)
        float temperature_compensated_strain;  // Strain corrected for thermal expansion (-1 if not available)
        
        // Warping/distortion indicators
        bool warping_detected;            // Significant warping detected
        float warping_magnitude;          // Warping magnitude (mm, -1 if not detected)
        float distortion_angle;            // Distortion angle (degrees, -1 if not detected)
        
        // Layer-wise strain accumulation
        float cumulative_strain;           // Cumulative strain from build start (με)
        float layer_strain_increment;      // Strain increment for this layer (με)
        
        // Event detection (critical for PBF-LB/M quality)
        bool excessive_strain_event;       // Strain exceeds threshold
        bool warping_event_detected;       // Significant warping event
        bool distortion_event_detected;    // Distortion event detected
        bool anomaly_detected;             // Boolean flag
        std::string anomaly_type;          // Anomaly type (e.g., "excessive_warping", "residual_stress_build-up", "layer_delamination")
        
        // Process stability metrics
        float process_stability;           // 0-1 scale (1 = stable, 0 = unstable)
        float strain_variation;            // Coefficient of variation (std/mean)
        float strain_uniformity;           // 0-1 scale (1 = uniform, 0 = non-uniform)
        
        // Process events
        std::string process_event;         // Process event (e.g., "layer_start", "warping_event", "distortion_event")
        
        // Strain energy metrics
        float strain_energy_density;       // Strain energy per unit volume (J/m³, -1 if not available)
    };
    std::vector<ISPMStrainData> ispm_strain_data;     // ISPM_Strain-specific data for each point
    
    // ISPM_Plume (In-Situ Process Monitoring - Plume) Data
    struct ISPMPlumeData {
        // Plume characteristics (primary vapor plume monitoring in PBF-LB/M)
        float plume_intensity;          // Arbitrary units (0-10000 typical range)
        float plume_density;             // kg/m³ (vapor density in plume)
        float plume_temperature;         // Celsius (temperature of vapor plume)
        float plume_velocity;            // m/s (vertical velocity of plume)
        float plume_velocity_x;         // m/s (horizontal velocity in x-direction)
        float plume_velocity_y;         // m/s (horizontal velocity in y-direction)
        
        // Plume geometry
        float plume_height;             // mm (height of plume above melt pool)
        float plume_width;              // mm (width/diameter of plume at base)
        float plume_angle;               // degrees (angle of plume from vertical)
        float plume_spread;             // mm (spread/divergence of plume)
        float plume_area;               // mm² (cross-sectional area of plume)
        
        // Plume composition
        float particle_concentration;    // particles/m³ (particle concentration in plume)
        float metal_vapor_concentration; // kg/m³ (metal vapor concentration)
        float gas_composition_ratio;     // ratio (ratio of metal vapor to inert gas)
        
        // Plume dynamics
        float plume_fluctuation_rate;    // Hz (rate of plume fluctuations)
        float plume_instability_index;   // 0-1 scale (instability of plume, 1 = very unstable)
        float plume_turbulence;          // 0-1 scale (turbulence level in plume)
        
        // Process quality indicators
        float process_stability;        // 0-1 scale (1 = stable, 0 = unstable)
        float plume_stability;          // 0-1 scale (1 = stable plume, 0 = unstable)
        float intensity_variation;      // Coefficient of variation (std/mean)
        
        // Event detection (critical for PBF-LB/M quality)
        bool excessive_plume_event;     // Boolean flag
        bool unstable_plume_event;      // Boolean flag
        bool contamination_event;       // Boolean flag
        bool anomaly_detected;          // Boolean flag
        std::string anomaly_type;       // Anomaly type (e.g., "excessive_plume", "unstable_plume", "contamination")
        
        // Plume energy metrics
        float plume_energy;             // Total plume energy (arbitrary units)
        float energy_density;           // Energy per unit volume (J/m³)
        
        // Process events
        std::string process_event;      // Process event (e.g., "layer_start", "excessive_plume_event", "unstable_plume_event")
        
        // Signal-to-noise ratio
        float signal_to_noise_ratio;    // SNR in dB
        
        // Additional plume features
        float plume_momentum;           // kg·m/s (plume momentum, -1 if not available)
        float plume_pressure;           // Pa (pressure in plume region, -1 if not available)
    };
    std::vector<ISPMPlumeData> ispm_plume_data;     // ISPM_Plume-specific data for each point
    
    // Laser Temporal Sensor Data (Category 3: Temporal Sensors for Anomaly Detection)
    // Based on sensor_fields_research.md - Category 3: Temporal Sensors
    struct LaserTemporalData {
        // Process Parameters (setpoints/commanded values)
        float commanded_power;           // W (setpoint/commanded power)
        float commanded_scan_speed;     // mm/s (setpoint/commanded speed)
        
        // Category 3.1 - Laser Power Sensors (Temporal)
        float actual_power;              // W (measured actual power)
        float power_setpoint;            // W (commanded power - same as commanded_power)
        float power_error;               // W (actual_power - power_setpoint)
        float power_stability;           // % (coefficient of variation)
        float power_fluctuation_amplitude;  // W (peak-to-peak variation)
        float power_fluctuation_frequency;  // Hz (dominant frequency)
        
        // Category 3.2 - Beam Temporal Characteristics
        float pulse_frequency;           // Hz (for pulsed lasers)
        float pulse_duration;            // µs (pulse width FWHM)
        float pulse_energy;              // mJ (energy per pulse)
        float duty_cycle;                // % (on-time percentage)
        float beam_modulation_frequency; // Hz (power modulation frequency)
        
        // Category 3.3 - Laser System Health (Temporal)
        float laser_temperature;         // °C (laser head/cavity temperature)
        float laser_cooling_water_temp;  // °C
        float laser_cooling_flow_rate;   // L/min
        float laser_power_supply_voltage; // V
        float laser_power_supply_current; // A
        float laser_diode_current;       // A
        float laser_diode_temperature;  // °C
        float laser_operating_hours;     // hours (total laser operating time)
        int laser_pulse_count;           // count (total number of pulses for pulsed lasers)
    };
    std::vector<LaserTemporalData> laser_temporal_data;  // Temporal sensor data for each point
    
    // Hatch metadata (LEGACY: for old point-based format)
    // NOTE: For vector-based format, use vectordata instead
    struct HatchMetadata {
        std::string hatch_type;                  // "raster", "line", "infill", etc.
        float laser_beam_width;                  // Laser beam width (mm)
        float hatch_spacing;                     // Hatch spacing (mm)
        float overlap_percentage;                // Overlap percentage
    };
    std::vector<HatchMetadata> hatch_metadata;  // Metadata for each hatch (LEGACY: point-based format)
    std::vector<int> hatch_start_indices;       // Starting index in points array (LEGACY: point-based format)
    
    // Contour data (for hatching visualization)
    struct Contour {
        std::vector<std::array<float, 3>> points;  // Contour points
        std::string sub_type;                       // "inner" or "outer"
        std::string color;                          // Line color (e.g., "#f57900" for inner, "#204a87" for outer)
        float linewidth;                           // Line width
    };
    std::vector<Contour> contours;              // Contour paths per layer
    std::vector<int> contour_layers;           // Layer index for each contour
    
    // Vector-based format (NEW: correct format for accurate voxelization)
    struct Vector {
        float x1, y1, x2, y2, z;  // Line segment endpoints and z-height
        float timestamp;           // Timestamp
        int dataindex;            // Index into vectordata array
    };
    std::vector<Vector> vectors;  // Vector segments (not connected - each is independent)
    
    struct VectorData {
        int dataindex;            // Index to match with vectors
        int partid;               // Part ID
        std::string type;         // Type: "(INFILL)", "(CONTOUR)", "(PARTBOUNDARY)", etc.
        int scanner;              // Scanner ID
        float laserpower;         // Laser power
        float scannerspeed;        // Scanner speed
        float laser_beam_width;   // Laser beam width (mm)
        float hatch_spacing;      // Hatch spacing (mm)
        int layer_index;          // Layer index
        // Additional fields can be added as needed
    };
    std::vector<VectorData> vectordata;  // Process parameters for each vector
    
    // Metadata
    std::string model_id;
    std::string signal_type;
    std::string format;  // "vector-based" (correct for hatching) or "point-based" (legacy/other data types)
    
    // Helper methods
    int num_points() const { return static_cast<int>(points.size()); }  // For point-based data
    int num_vectors() const { return static_cast<int>(vectors.size()); }  // For vector-based hatching data
    bool empty() const { 
        // For hatching: check vectors; for other data: check points
        if (format == "vector-based") {
            return vectors.empty();
        }
        return points.empty(); 
    }
    
    // Check if multiple signals are available
    bool has_multiple_signals() const {
        // Check structured data sources
        return !laser_temporal_data.empty() || !ispm_thermal_data.empty() || !ispm_optical_data.empty() || !ispm_acoustic_data.empty() || !ispm_strain_data.empty() || !ispm_plume_data.empty() || !vectordata.empty();
    }
    
    // Check if contours are available
    bool has_contours() const {
        return !contours.empty();
    }
};

} // namespace query
} // namespace am_qadf_native

#endif // AM_QADF_NATIVE_QUERY_QUERY_RESULT_HPP
