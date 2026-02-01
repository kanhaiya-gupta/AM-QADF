#ifndef AM_QADF_NATIVE_SYNCHRONIZATION_POINT_TRANSFORMATION_VALIDATE_HPP
#define AM_QADF_NATIVE_SYNCHRONIZATION_POINT_TRANSFORMATION_VALIDATE_HPP

#ifdef EIGEN_AVAILABLE
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include "am_qadf_native/synchronization/point_coordinate_transform.hpp"
#include "am_qadf_native/synchronization/point_bounds.hpp"  // For BoundingBox
#include <vector>
#include <string>

namespace am_qadf_native {
namespace synchronization {

// Validation result structure
struct ValidationResult {
    bool isValid;
    double max_error;
    double mean_error;
    double rms_error;
    std::vector<std::string> errors;
    std::vector<std::string> warnings;
    
    ValidationResult() : isValid(false), max_error(0.0), mean_error(0.0), rms_error(0.0) {}
};

// Result of bbox corners + centre correspondence check (9 known pairs).
struct BboxCorrespondenceValidation {
    double mean_distance;
    double max_distance;
    int num_pairs;  // 9 (8 corners + 1 centre)
    BboxCorrespondenceValidation() : mean_distance(0.0), max_distance(0.0), num_pairs(0) {}
};

// TransformationValidator: Validates transformations before bulk processing
// Performs validation on small sample (100-1000 points) to prevent data corruption
class TransformationValidator {
public:
    // Validate transformation with sample points
    // Validates transformation between coordinate systems using sample points
    ValidationResult validate(
        const CoordinateSystem& source_system,
        const CoordinateSystem& target_system,
        const std::vector<Eigen::Vector3d>& sample_points,
        double tolerance = 1e-9
    );
    
    // Validate transformation with pre-computed matrix
    // Validates transformation matrix directly using sample points
    ValidationResult validateWithMatrix(
        const Eigen::MatrixXd& source_points,
        const Eigen::Matrix4d& transformation,
        const Eigen::MatrixXd& target_points,
        double tolerance = 1e-9
    );
    
    // Validate with quality metrics (from TransformationComputer)
    // Uses quality metrics structure for comprehensive validation
    ValidationResult validateWithQualityMetrics(
        const Eigen::MatrixXd& source_points,
        const Eigen::Matrix4d& transformation,
        const Eigen::MatrixXd& target_points,
        double tolerance = 1e-9
    );
    
    // Round-trip validation
    // Transforms forward then inverse, checks if original point is recovered
    ValidationResult validateRoundTrip(
        const std::vector<Eigen::Vector3d>& points,
        const Eigen::Matrix4d& forward_transform,
        const Eigen::Matrix4d& inverse_transform,
        double tolerance = 1e-9
    );
    
    // Geometric consistency validation
    // Checks distance preservation, determinant sign, etc.
    ValidationResult validateGeometricConsistency(
        const std::vector<Eigen::Vector3d>& points,
        const Eigen::Matrix4d& transform
    );
    
    // Statistical bounds validation
    // Verifies transformed points are within expected bounds
    ValidationResult validateBounds(
        const std::vector<Eigen::Vector3d>& transformed_points,
        const BoundingBox& expected_bounds,
        double max_out_of_bounds_ratio = 0.01
    );

    // Point correspondence check on bbox corners (8) and centre (1) — known pairs, all in C++.
    // source_corners and reference_corners must be 8x3 (BoundingBox::corners() order). Centre computed from corners.
    BboxCorrespondenceValidation validateBboxCornersAndCentre(
        const Eigen::MatrixXd& source_corners,
        const Eigen::MatrixXd& reference_corners,
        const Eigen::Matrix4d& transformation
    );

private:
    // Helper: Apply transformation to point
    Eigen::Vector3d applyTransform(
        const Eigen::Vector3d& point,
        const Eigen::Matrix4d& transform
    );
    
    // Helper: Compute inverse transformation
    Eigen::Matrix4d computeInverse(const Eigen::Matrix4d& transform);
    
    // Helper: Check if transformation is valid (determinant, orthogonality, etc.)
    bool isValidTransformation(const Eigen::Matrix4d& transform);
};

} // namespace synchronization
} // namespace am_qadf_native

#endif // EIGEN_AVAILABLE
#endif // AM_QADF_NATIVE_SYNCHRONIZATION_POINT_TRANSFORMATION_VALIDATE_HPP
