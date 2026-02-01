#include "am_qadf_native/correction/calibration.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <vector>
#include <array>
#include <map>
#include <string>
#include <iomanip>
#include <cmath>

#ifdef EIGEN_AVAILABLE
#include <Eigen/Dense>
#endif

namespace am_qadf_native {
namespace correction {

CalibrationData Calibration::loadFromFile(const std::string& filename) {
    CalibrationData data;
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        return data;  // Return empty data if file cannot be opened
    }
    
    std::string line;
    std::string current_section;
    
    while (std::getline(file, line)) {
        // Remove whitespace
        line.erase(0, line.find_first_not_of(" \t"));
        line.erase(line.find_last_not_of(" \t") + 1);
        
        if (line.empty() || line[0] == '#') {
            continue;  // Skip empty lines and comments
        }
        
        // Parse sections
        if (line[0] == '[' && line.back() == ']') {
            current_section = line.substr(1, line.length() - 2);
            continue;
        }
        
        // Parse key-value pairs
        size_t eq_pos = line.find('=');
        if (eq_pos != std::string::npos) {
            std::string key = line.substr(0, eq_pos);
            std::string value = line.substr(eq_pos + 1);
            
            // Remove whitespace from key and value
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);
            
            if (current_section == "metadata") {
                if (key == "sensor_id") {
                    data.sensor_id = value;
                } else if (key == "calibration_type") {
                    data.calibration_type = value;
                }
            } else if (current_section == "parameters") {
                try {
                    float param_value = std::stof(value);
                    data.parameters[key] = param_value;
                } catch (...) {
                    // Skip invalid parameter values
                }
            } else if (current_section == "reference_points" || current_section == "measured_points") {
                // Parse point: "x,y,z"
                std::istringstream iss(value);
                std::string coord_str;
                std::array<float, 3> point = {0.0f, 0.0f, 0.0f};
                int coord_idx = 0;
                
                while (std::getline(iss, coord_str, ',') && coord_idx < 3) {
                    try {
                        point[coord_idx] = std::stof(coord_str);
                        coord_idx++;
                    } catch (...) {
                        break;
                    }
                }
                
                if (current_section == "reference_points") {
                    data.reference_points.push_back(point);
                } else {
                    data.measured_points.push_back(point);
                }
            }
        }
    }
    
    file.close();
    return data;
}

void Calibration::saveToFile(const CalibrationData& data, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return;  // Cannot open file for writing
    }
    
    // Write metadata section
    file << "[metadata]\n";
    file << "sensor_id=" << data.sensor_id << "\n";
    file << "calibration_type=" << data.calibration_type << "\n";
    file << "\n";
    
    // Write parameters section
    if (!data.parameters.empty()) {
        file << "[parameters]\n";
        for (const auto& param : data.parameters) {
            file << param.first << "=" << std::setprecision(10) << param.second << "\n";
        }
        file << "\n";
    }
    
    // Write reference points section
    if (!data.reference_points.empty()) {
        file << "[reference_points]\n";
        for (const auto& point : data.reference_points) {
            file << point[0] << "," << point[1] << "," << point[2] << "\n";
        }
        file << "\n";
    }
    
    // Write measured points section
    if (!data.measured_points.empty()) {
        file << "[measured_points]\n";
        for (const auto& point : data.measured_points) {
            file << point[0] << "," << point[1] << "," << point[2] << "\n";
        }
        file << "\n";
    }
    
    file.close();
}

CalibrationData Calibration::computeCalibration(
    const std::vector<std::array<float, 3>>& reference_points,
    const std::vector<std::array<float, 3>>& measured_points,
    const std::string& calibration_type
) {
    CalibrationData data;
    data.calibration_type = calibration_type;
    data.reference_points = reference_points;
    data.measured_points = measured_points;
    
    if (reference_points.size() != measured_points.size() || reference_points.empty()) {
        return data;  // Invalid input
    }
    
#ifdef EIGEN_AVAILABLE
    if (calibration_type == "translation" || calibration_type == "rigid") {
        // Compute translation: mean difference
        std::array<float, 3> translation = {0.0f, 0.0f, 0.0f};
        for (size_t i = 0; i < reference_points.size(); ++i) {
            translation[0] += measured_points[i][0] - reference_points[i][0];
            translation[1] += measured_points[i][1] - reference_points[i][1];
            translation[2] += measured_points[i][2] - reference_points[i][2];
        }
        
        float n = static_cast<float>(reference_points.size());
        translation[0] /= n;
        translation[1] /= n;
        translation[2] /= n;
        
        data.parameters["tx"] = translation[0];
        data.parameters["ty"] = translation[1];
        data.parameters["tz"] = translation[2];
        
        if (calibration_type == "rigid") {
            // Compute rotation using SVD (simplified - assumes small rotations)
            // For full rigid transformation, use Kabsch algorithm
            // For now, compute scale and approximate rotation
            float scale = 1.0f;
            
            // Compute scale from distances
            float ref_dist_sum = 0.0f;
            float meas_dist_sum = 0.0f;
            int pair_count = 0;
            
            for (size_t i = 0; i < reference_points.size(); ++i) {
                for (size_t j = i + 1; j < reference_points.size(); ++j) {
                    float ref_dx = reference_points[i][0] - reference_points[j][0];
                    float ref_dy = reference_points[i][1] - reference_points[j][1];
                    float ref_dz = reference_points[i][2] - reference_points[j][2];
                    float ref_dist = std::sqrt(ref_dx*ref_dx + ref_dy*ref_dy + ref_dz*ref_dz);
                    
                    float meas_dx = measured_points[i][0] - measured_points[j][0];
                    float meas_dy = measured_points[i][1] - measured_points[j][1];
                    float meas_dz = measured_points[i][2] - measured_points[j][2];
                    float meas_dist = std::sqrt(meas_dx*meas_dx + meas_dy*meas_dy + meas_dz*meas_dz);
                    
                    if (ref_dist > 1e-6f && meas_dist > 1e-6f) {
                        ref_dist_sum += ref_dist;
                        meas_dist_sum += meas_dist;
                        pair_count++;
                    }
                }
            }
            
            if (pair_count > 0 && ref_dist_sum > 1e-6f) {
                scale = meas_dist_sum / ref_dist_sum;
            }
            
            data.parameters["scale"] = scale;
            // Rotation angles (simplified - would need full SVD for accurate rotation)
            data.parameters["rx"] = 0.0f;  // Roll
            data.parameters["ry"] = 0.0f;  // Pitch
            data.parameters["rz"] = 0.0f;  // Yaw
        }
    } else if (calibration_type == "affine") {
        // Compute affine transformation using least squares
        // Solve: measured = A * reference + t
        // Using Eigen for linear system solving
        
        size_t n = reference_points.size();
        if (n < 4) {
            return data;  // Need at least 4 points for affine transformation
        }
        
        // Build system: [x' y' z'] = [A] * [x y z] + [t]
        // Rearranged: [x' y' z'] = [x y z 1] * [A^T; t^T]
        Eigen::MatrixXf A(n * 3, 12);  // 12 parameters: 3x3 matrix + 3x1 translation
        Eigen::VectorXf b(n * 3);
        
        for (size_t i = 0; i < n; ++i) {
            // For x coordinate
            A(i * 3 + 0, 0) = reference_points[i][0];
            A(i * 3 + 0, 1) = reference_points[i][1];
            A(i * 3 + 0, 2) = reference_points[i][2];
            A(i * 3 + 0, 3) = 1.0f;
            A(i * 3 + 0, 4) = 0.0f;
            A(i * 3 + 0, 5) = 0.0f;
            A(i * 3 + 0, 6) = 0.0f;
            A(i * 3 + 0, 7) = 0.0f;
            A(i * 3 + 0, 8) = 0.0f;
            A(i * 3 + 0, 9) = 0.0f;
            A(i * 3 + 0, 10) = 0.0f;
            A(i * 3 + 0, 11) = 0.0f;
            b(i * 3 + 0) = measured_points[i][0];
            
            // For y coordinate
            A(i * 3 + 1, 0) = 0.0f;
            A(i * 3 + 1, 1) = 0.0f;
            A(i * 3 + 1, 2) = 0.0f;
            A(i * 3 + 1, 3) = 0.0f;
            A(i * 3 + 1, 4) = reference_points[i][0];
            A(i * 3 + 1, 5) = reference_points[i][1];
            A(i * 3 + 1, 6) = reference_points[i][2];
            A(i * 3 + 1, 7) = 1.0f;
            A(i * 3 + 1, 8) = 0.0f;
            A(i * 3 + 1, 9) = 0.0f;
            A(i * 3 + 1, 10) = 0.0f;
            A(i * 3 + 1, 11) = 0.0f;
            b(i * 3 + 1) = measured_points[i][1];
            
            // For z coordinate
            A(i * 3 + 2, 0) = 0.0f;
            A(i * 3 + 2, 1) = 0.0f;
            A(i * 3 + 2, 2) = 0.0f;
            A(i * 3 + 2, 3) = 0.0f;
            A(i * 3 + 2, 4) = 0.0f;
            A(i * 3 + 2, 5) = 0.0f;
            A(i * 3 + 2, 6) = 0.0f;
            A(i * 3 + 2, 7) = 0.0f;
            A(i * 3 + 2, 8) = reference_points[i][0];
            A(i * 3 + 2, 9) = reference_points[i][1];
            A(i * 3 + 2, 10) = reference_points[i][2];
            A(i * 3 + 2, 11) = 1.0f;
            b(i * 3 + 2) = measured_points[i][2];
        }
        
        // Solve least squares: A * x = b
        Eigen::VectorXf x = A.colPivHouseholderQr().solve(b);
        
        // Store parameters (3x3 matrix + translation)
        data.parameters["m00"] = x(0);
        data.parameters["m01"] = x(1);
        data.parameters["m02"] = x(2);
        data.parameters["tx"] = x(3);
        data.parameters["m10"] = x(4);
        data.parameters["m11"] = x(5);
        data.parameters["m12"] = x(6);
        data.parameters["ty"] = x(7);
        data.parameters["m20"] = x(8);
        data.parameters["m21"] = x(9);
        data.parameters["m22"] = x(10);
        data.parameters["tz"] = x(11);
    } else if (calibration_type == "distortion") {
        // Compute distortion parameters (simplified)
        // For lens distortion, compute k1, k2, p1, p2 from point correspondences
        // This is a simplified version - full implementation would use non-linear optimization
        
        // Compute radial distortion coefficient k1 (simplified)
        float k1 = 0.0f;
        int valid_pairs = 0;
        
        for (size_t i = 0; i < reference_points.size(); ++i) {
            float ref_r_sq = reference_points[i][0]*reference_points[i][0] + 
                            reference_points[i][1]*reference_points[i][1];
            float meas_r_sq = measured_points[i][0]*measured_points[i][0] + 
                             measured_points[i][1]*measured_points[i][1];
            
            if (ref_r_sq > 1e-6f) {
                // Simplified: k1 = (r_measured^2 - r_ref^2) / (r_ref^4)
                float k1_estimate = (meas_r_sq - ref_r_sq) / (ref_r_sq * ref_r_sq);
                k1 += k1_estimate;
                valid_pairs++;
            }
        }
        
        if (valid_pairs > 0) {
            k1 /= valid_pairs;
        }
        
        data.parameters["k1"] = k1;
        data.parameters["k2"] = 0.0f;  // Would need more points for k2
        data.parameters["p1"] = 0.0f;  // Tangential distortion
        data.parameters["p2"] = 0.0f;
    }
#else
    // Fallback: Compute simple translation if Eigen not available
    std::array<float, 3> translation = {0.0f, 0.0f, 0.0f};
    for (size_t i = 0; i < reference_points.size(); ++i) {
        translation[0] += measured_points[i][0] - reference_points[i][0];
        translation[1] += measured_points[i][1] - reference_points[i][1];
        translation[2] += measured_points[i][2] - reference_points[i][2];
    }
    
    float n = static_cast<float>(reference_points.size());
    data.parameters["tx"] = translation[0] / n;
    data.parameters["ty"] = translation[1] / n;
    data.parameters["tz"] = translation[2] / n;
#endif
    
    return data;
}

bool Calibration::validateCalibration(const CalibrationData& data) const {
    // TODO: Validate calibration data
    // Check: non-empty, consistent sizes, valid parameters
    
    if (data.reference_points.size() != data.measured_points.size()) {
        return false;
    }
    
    if (data.reference_points.empty()) {
        return false;
    }
    
    return true;
}

} // namespace correction
} // namespace am_qadf_native
