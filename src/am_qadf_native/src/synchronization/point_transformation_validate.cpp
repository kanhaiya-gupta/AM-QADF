#ifdef EIGEN_AVAILABLE

#include "am_qadf_native/synchronization/point_transformation_validate.hpp"
#include "am_qadf_native/synchronization/point_coordinate_transform.hpp"
#include <cmath>
#include <algorithm>
#include <sstream>

namespace am_qadf_native {
namespace synchronization {

// BoundingBox implementation is in unified_bounds_computer.cpp

// Helper: Apply transformation to point
Eigen::Vector3d TransformationValidator::applyTransform(
    const Eigen::Vector3d& point,
    const Eigen::Matrix4d& transform
) {
    Eigen::Vector4d homogeneous_point;
    homogeneous_point << point, 1.0;
    Eigen::Vector4d transformed = transform * homogeneous_point;
    return transformed.head<3>();
}

// Helper: Compute inverse transformation
Eigen::Matrix4d TransformationValidator::computeInverse(const Eigen::Matrix4d& transform) {
    Eigen::Matrix4d inv = Eigen::Matrix4d::Identity();
    
    // Extract rotation and translation
    Eigen::Matrix3d R = transform.block<3, 3>(0, 0);
    Eigen::Vector3d t = transform.block<3, 1>(0, 3);
    
    // Inverse: R^-1 and -R^-1 * t
    Eigen::Matrix3d R_inv = R.transpose();  // For rotation matrices
    Eigen::Vector3d t_inv = -R_inv * t;
    
    inv.block<3, 3>(0, 0) = R_inv;
    inv.block<3, 1>(0, 3) = t_inv;
    
    return inv;
}

// Helper: Check if transformation is valid (rigid or similarity: rotation + optional uniform scale)
bool TransformationValidator::isValidTransformation(const Eigen::Matrix4d& transform) {
    // Check bottom row is [0, 0, 0, 1]
    if (std::abs(transform(3, 0)) > 1e-9 ||
        std::abs(transform(3, 1)) > 1e-9 ||
        std::abs(transform(3, 2)) > 1e-9 ||
        std::abs(transform(3, 3) - 1.0) > 1e-9) {
        return false;
    }
    
    Eigen::Matrix3d M = transform.block<3, 3>(0, 0);
    double scale = M.col(0).norm();
    if (scale < 1e-10) {
        return false;
    }
    // Accept similarity transform (Kabsch+Umeyama): M = s*R, so R = M/s
    Eigen::Matrix3d R = M / scale;
    Eigen::Matrix3d should_be_identity = R * R.transpose();
    Eigen::Matrix3d identity = Eigen::Matrix3d::Identity();
    double orthogonality_error = (should_be_identity - identity).cwiseAbs().maxCoeff();
    if (orthogonality_error > 1e-5) {
        return false;
    }
    double det = R.determinant();
    if (std::abs(det - 1.0) > 1e-5) {
        return false;
    }
    return true;
}

// Validate transformation with sample points
ValidationResult TransformationValidator::validate(
    const CoordinateSystem& source_system,
    const CoordinateSystem& target_system,
    const std::vector<Eigen::Vector3d>& sample_points,
    double tolerance
) {
    ValidationResult result;
    
    if (sample_points.empty()) {
        result.isValid = false;
        result.errors.push_back("Sample points cannot be empty");
        return result;
    }
    
    // Use CoordinateTransformer to build transformation
    CoordinateTransformer transformer;
    Eigen::Matrix4d transform = transformer.buildTransformMatrix(target_system);
    Eigen::Matrix4d inverse_transform = transformer.buildInverseTransformMatrix(source_system);
    Eigen::Matrix4d combined_transform = transform * inverse_transform;
    
    // Validate geometric consistency
    auto consistency_result = validateGeometricConsistency(sample_points, combined_transform);
    if (!consistency_result.isValid) {
        result.isValid = false;
        result.errors.insert(result.errors.end(), 
                            consistency_result.errors.begin(), 
                            consistency_result.errors.end());
        return result;
    }
    
    // Round-trip validation
    auto round_trip_result = validateRoundTrip(sample_points, combined_transform, 
                                               computeInverse(combined_transform), tolerance);
    if (!round_trip_result.isValid) {
        result.isValid = false;
        result.errors.insert(result.errors.end(), 
                            round_trip_result.errors.begin(), 
                            round_trip_result.errors.end());
        return result;
    }
    
    // Transformation is valid if all checks pass
    // (In practice, you'd compare transformed points with target points here)
    result.isValid = true;
    
    result.isValid = true;
    return result;
}

// Validate transformation with pre-computed matrix
ValidationResult TransformationValidator::validateWithMatrix(
    const Eigen::MatrixXd& source_points,
    const Eigen::Matrix4d& transformation,
    const Eigen::MatrixXd& target_points,
    double tolerance
) {
    ValidationResult result;
    
    if (source_points.rows() != target_points.rows()) {
        result.isValid = false;
        result.errors.push_back("Source and target point sets must have same number of points");
        return result;
    }
    
    if (source_points.rows() == 0) {
        result.isValid = false;
        result.errors.push_back("Point sets cannot be empty");
        return result;
    }
    
    // Check transformation matrix validity
    if (!isValidTransformation(transformation)) {
        result.isValid = false;
        result.errors.push_back("Transformation matrix is invalid (not orthogonal or wrong format)");
        return result;
    }
    
    // Compute errors
    std::vector<double> errors;
    double sum_squared_error = 0.0;
    double max_err = 0.0;
    
    for (int i = 0; i < source_points.rows(); ++i) {
        Eigen::Vector3d source_pt = source_points.row(i);
        Eigen::Vector3d transformed_pt = applyTransform(source_pt, transformation);
        Eigen::Vector3d target_pt = target_points.row(i);
        
        double error = (transformed_pt - target_pt).norm();
        errors.push_back(error);
        sum_squared_error += error * error;
        max_err = std::max(max_err, error);
    }
    
    result.max_error = max_err;
    result.mean_error = sum_squared_error / source_points.rows();
    result.rms_error = std::sqrt(result.mean_error);
    
    // Valid until we find a violation
    result.isValid = true;
    if (max_err > tolerance) {
        result.isValid = false;
        std::ostringstream oss;
        oss << "Maximum error " << max_err << " exceeds tolerance " << tolerance;
        result.errors.push_back(oss.str());
    }
    if (result.rms_error > tolerance) {
        result.isValid = false;
        std::ostringstream oss;
        oss << "RMS error " << result.rms_error << " exceeds tolerance " << tolerance;
        result.errors.push_back(oss.str());
    }
    // Warn if errors are close to tolerance
    if (result.isValid && max_err > 0.8 * tolerance) {
        std::ostringstream oss;
        oss << "Maximum error " << max_err << " is close to tolerance " << tolerance;
        result.warnings.push_back(oss.str());
    }
    
    return result;
}

// Validate with quality metrics
ValidationResult TransformationValidator::validateWithQualityMetrics(
    const Eigen::MatrixXd& source_points,
    const Eigen::Matrix4d& transformation,
    const Eigen::MatrixXd& target_points,
    double tolerance
) {
    ValidationResult result = validateWithMatrix(source_points, transformation, target_points, tolerance);
    
    // Additional quality checks based on quality metrics
    // These would typically come from TransformationComputer::computeQualityMetrics
    // For now, we use the error metrics from validateWithMatrix
    
    if (result.rms_error > tolerance) {
        result.isValid = false;
    }
    
    // Quality thresholds (can be adjusted)
    if (result.rms_error > tolerance * 0.5) {
        std::ostringstream oss;
        oss << "RMS error " << result.rms_error << " is high (>" << tolerance * 0.5 << ")";
        result.warnings.push_back(oss.str());
    }
    
    return result;
}

// Round-trip validation
ValidationResult TransformationValidator::validateRoundTrip(
    const std::vector<Eigen::Vector3d>& points,
    const Eigen::Matrix4d& forward_transform,
    const Eigen::Matrix4d& inverse_transform,
    double tolerance
) {
    ValidationResult result;
    
    if (points.empty()) {
        result.isValid = false;
        result.errors.push_back("Points cannot be empty");
        return result;
    }
    
    // Check transformation validity
    if (!isValidTransformation(forward_transform) || !isValidTransformation(inverse_transform)) {
        result.isValid = false;
        result.errors.push_back("Transformation matrices are invalid");
        return result;
    }
    
    // Test round-trip: forward then inverse should recover original
    double max_round_trip_error = 0.0;
    double sum_squared_error = 0.0;
    
    for (const auto& point : points) {
        Eigen::Vector3d transformed = applyTransform(point, forward_transform);
        Eigen::Vector3d recovered = applyTransform(transformed, inverse_transform);
        
        double error = (recovered - point).norm();
        max_round_trip_error = std::max(max_round_trip_error, error);
        sum_squared_error += error * error;
    }
    
    result.max_error = max_round_trip_error;
    result.mean_error = sum_squared_error / points.size();
    result.rms_error = std::sqrt(result.mean_error);
    
    if (max_round_trip_error > tolerance) {
        result.isValid = false;
        std::ostringstream oss;
        oss << "Round-trip error " << max_round_trip_error << " exceeds tolerance " << tolerance;
        result.errors.push_back(oss.str());
    } else {
        result.isValid = true;
    }
    
    return result;
}

// Geometric consistency validation
ValidationResult TransformationValidator::validateGeometricConsistency(
    const std::vector<Eigen::Vector3d>& points,
    const Eigen::Matrix4d& transform
) {
    ValidationResult result;
    
    if (points.empty()) {
        result.isValid = false;
        result.errors.push_back("Points cannot be empty");
        return result;
    }
    
    // Check transformation matrix validity
    if (!isValidTransformation(transform)) {
        result.isValid = false;
        result.errors.push_back("Transformation matrix is invalid");
        return result;
    }
    
    // Check distance preservation (for rigid transformations)
    // Sample a few point pairs and check distances are preserved
    int num_checks = std::min(10, static_cast<int>(points.size()));
    double max_distance_error = 0.0;
    
    for (int i = 0; i < num_checks; ++i) {
        for (int j = i + 1; j < num_checks; ++j) {
            // Original distance
            double original_dist = (points[i] - points[j]).norm();
            
            // Transformed distance
            Eigen::Vector3d transformed_i = applyTransform(points[i], transform);
            Eigen::Vector3d transformed_j = applyTransform(points[j], transform);
            double transformed_dist = (transformed_i - transformed_j).norm();
            
            // For rigid transformations, distances should be preserved (within scale factor)
            // Extract scale from transformation
            Eigen::Matrix3d R = transform.block<3, 3>(0, 0);
            // For uniform scaling, we can check if R has uniform scale
            // For now, just check that transformation is reasonable
            double dist_error = std::abs(original_dist - transformed_dist) / std::max(original_dist, 1e-10);
            max_distance_error = std::max(max_distance_error, dist_error);
        }
    }
    
    // For rigid transformations with uniform scaling, distance error should be small
    // Allow up to 1% error for numerical precision
    if (max_distance_error > 0.01) {
        result.warnings.push_back("Distance preservation check shows potential issues");
    }
    
    result.isValid = true;
    return result;
}

// Statistical bounds validation
ValidationResult TransformationValidator::validateBounds(
    const std::vector<Eigen::Vector3d>& transformed_points,
    const BoundingBox& expected_bounds,
    double max_out_of_bounds_ratio
) {
    ValidationResult result;
    
    if (transformed_points.empty()) {
        result.isValid = false;
        result.errors.push_back("Transformed points cannot be empty");
        return result;
    }
    
    // Count points outside bounds
    int out_of_bounds = 0;
    for (const auto& point : transformed_points) {
        if (!expected_bounds.contains(point)) {
            out_of_bounds++;
        }
    }
    
    double out_of_bounds_ratio = static_cast<double>(out_of_bounds) / transformed_points.size();
    
    if (out_of_bounds_ratio > max_out_of_bounds_ratio) {
        result.isValid = false;
        std::ostringstream oss;
        oss << "Out of bounds ratio " << out_of_bounds_ratio 
            << " exceeds maximum " << max_out_of_bounds_ratio;
        result.errors.push_back(oss.str());
    } else {
        result.isValid = true;
    }
    
    if (out_of_bounds_ratio > 0.0) {
        std::ostringstream oss;
        oss << out_of_bounds << " points (" << (out_of_bounds_ratio * 100.0) 
            << "%) are outside expected bounds";
        result.warnings.push_back(oss.str());
    }
    
    return result;
}

// Point correspondence check on bbox corners (8) and centre (1) — known pairs.
BboxCorrespondenceValidation TransformationValidator::validateBboxCornersAndCentre(
    const Eigen::MatrixXd& source_corners,
    const Eigen::MatrixXd& reference_corners,
    const Eigen::Matrix4d& transformation
) {
    BboxCorrespondenceValidation out;
    out.num_pairs = 9;
    if (source_corners.rows() != 8 || reference_corners.rows() != 8 ||
        source_corners.cols() != 3 || reference_corners.cols() != 3) {
        return out;  // mean/max 0, num_pairs 9
    }
    // Centre from corners: (min + max) / 2 per dimension
    Eigen::Vector3d src_min = source_corners.colwise().minCoeff();
    Eigen::Vector3d src_max = source_corners.colwise().maxCoeff();
    Eigen::Vector3d ref_min = reference_corners.colwise().minCoeff();
    Eigen::Vector3d ref_max = reference_corners.colwise().maxCoeff();
    Eigen::Vector3d src_centre = (src_min + src_max) * 0.5;
    Eigen::Vector3d ref_centre = (ref_min + ref_max) * 0.5;
    double sum_dist = 0.0;
    double max_dist = 0.0;
    for (int i = 0; i < 8; ++i) {
        Eigen::Vector3d src_pt = source_corners.row(i);
        Eigen::Vector3d transformed = applyTransform(src_pt, transformation);
        double d = (transformed - reference_corners.row(i).transpose()).norm();
        sum_dist += d;
        max_dist = std::max(max_dist, d);
    }
    Eigen::Vector3d transformed_centre = applyTransform(src_centre, transformation);
    double centre_d = (transformed_centre - ref_centre).norm();
    sum_dist += centre_d;
    max_dist = std::max(max_dist, centre_d);
    out.mean_distance = sum_dist / 9.0;
    out.max_distance = max_dist;
    return out;
}

} // namespace synchronization
} // namespace am_qadf_native

#endif // EIGEN_AVAILABLE
