#ifndef AM_QADF_NATIVE_CORRECTION_VALIDATION_HPP
#define AM_QADF_NATIVE_CORRECTION_VALIDATION_HPP

#include <openvdb/openvdb.h>
#include <string>
#include <memory>
#include <vector>
#include <map>

namespace am_qadf_native {
namespace correction {

using FloatGrid = openvdb::FloatGrid;
using FloatGridPtr = FloatGrid::Ptr;

// Validation result
struct ValidationResult {
    bool is_valid;
    std::vector<std::string> errors;
    std::vector<std::string> warnings;
    std::map<std::string, float> metrics;  // Quality metrics
};

// Data validation utilities
class Validation {
public:
    // Validate grid data
    ValidationResult validateGrid(FloatGridPtr grid);
    
    // Validate signal data
    ValidationResult validateSignalData(
        const std::vector<float>& values,
        float min_value = 0.0f,
        float max_value = 1.0f
    );
    
    // Validate coordinate data
    ValidationResult validateCoordinates(
        const std::vector<std::array<float, 3>>& points,
        const std::array<float, 3>& bbox_min,
        const std::array<float, 3>& bbox_max
    );
    
    // Check data consistency
    bool checkConsistency(
        FloatGridPtr grid1,
        FloatGridPtr grid2,
        float tolerance = 1e-6f
    );
    
private:
    // Helper: Check for NaN or Inf values
    bool hasInvalidValues(FloatGridPtr grid) const;
    
    // Helper: Check grid bounds
    bool checkBounds(FloatGridPtr grid) const;
    
    // Helper: Compute statistics
    void computeStatistics(FloatGridPtr grid, float& min_val, float& max_val,
                          float& mean_val, float& std_val) const;
};

} // namespace correction
} // namespace am_qadf_native

#endif // AM_QADF_NATIVE_CORRECTION_VALIDATION_HPP
