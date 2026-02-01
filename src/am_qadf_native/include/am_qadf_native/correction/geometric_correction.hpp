#ifndef AM_QADF_NATIVE_CORRECTION_GEOMETRIC_CORRECTION_HPP
#define AM_QADF_NATIVE_CORRECTION_GEOMETRIC_CORRECTION_HPP

#include <openvdb/openvdb.h>
#include <string>
#include <memory>
#include <vector>
#include <array>

namespace am_qadf_native {
namespace correction {

using FloatGrid = openvdb::FloatGrid;
using FloatGridPtr = FloatGrid::Ptr;

// Distortion map structure
struct DistortionMap {
    std::vector<std::array<float, 3>> distortion_vectors;  // Correction vectors
    std::vector<std::array<float, 3>> reference_points;    // Reference coordinates
    std::string distortion_type;  // "lens", "sensor_misalignment", etc.
};

// Geometric distortion correction
class GeometricCorrection {
public:
    // Apply distortion correction based on calibration data
    FloatGridPtr correctDistortions(
        FloatGridPtr grid,
        const DistortionMap& distortion_map
    );
    
    // Correct lens distortion
    FloatGridPtr correctLensDistortion(
        FloatGridPtr grid,
        const std::vector<float>& distortion_coefficients  // k1, k2, p1, p2, k3
    );
    
    // Correct sensor misalignment
    FloatGridPtr correctSensorMisalignment(
        FloatGridPtr grid,
        const std::array<float, 3>& translation,
        const std::array<float, 3>& rotation  // Euler angles
    );
    
private:
    // Helper: Apply distortion correction to a point
    std::array<float, 3> applyDistortionCorrection(
        const std::array<float, 3>& point,
        const DistortionMap& distortion_map
    ) const;
};

} // namespace correction
} // namespace am_qadf_native

#endif // AM_QADF_NATIVE_CORRECTION_GEOMETRIC_CORRECTION_HPP
