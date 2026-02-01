#ifndef AM_QADF_NATIVE_CORRECTION_CALIBRATION_HPP
#define AM_QADF_NATIVE_CORRECTION_CALIBRATION_HPP

#include <string>
#include <vector>
#include <array>
#include <map>

namespace am_qadf_native {
namespace correction {

// Calibration data structure
struct CalibrationData {
    std::string sensor_id;
    std::string calibration_type;  // "intrinsic", "extrinsic", "distortion"
    std::map<std::string, float> parameters;
    std::vector<std::array<float, 3>> reference_points;
    std::vector<std::array<float, 3>> measured_points;
};

// Calibration utilities
class Calibration {
public:
    // Load calibration data from file
    CalibrationData loadFromFile(const std::string& filename);
    
    // Save calibration data to file
    void saveToFile(const CalibrationData& data, const std::string& filename);
    
    // Compute calibration parameters from reference and measured points
    CalibrationData computeCalibration(
        const std::vector<std::array<float, 3>>& reference_points,
        const std::vector<std::array<float, 3>>& measured_points,
        const std::string& calibration_type
    );
    
    // Validate calibration data
    bool validateCalibration(const CalibrationData& data) const;
};

} // namespace correction
} // namespace am_qadf_native

#endif // AM_QADF_NATIVE_CORRECTION_CALIBRATION_HPP
