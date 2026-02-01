#ifdef EIGEN_AVAILABLE

#include "am_qadf_native/synchronization/point_transform.hpp"
#include "am_qadf_native/synchronization/point_coordinate_transform.hpp"
#include <stdexcept>

namespace am_qadf_native {
namespace synchronization {

// Helper: Apply transformation to point
Eigen::Vector3d PointTransformer::applyTransform(
    const Eigen::Vector3d& point,
    const Eigen::Matrix4d& transform
) {
    Eigen::Vector4d homogeneous_point;
    homogeneous_point << point, 1.0;
    Eigen::Vector4d transformed = transform * homogeneous_point;
    return transformed.head<3>();
}

// Transform points from source to target coordinate system
Eigen::MatrixXd PointTransformer::transform(
    const Eigen::MatrixXd& points,
    const CoordinateSystem& source_system,
    const CoordinateSystem& target_system
) {
    if (points.rows() == 0) {
        return points;
    }
    
    if (points.cols() != 3) {
        throw std::invalid_argument("Points matrix must have 3 columns (x, y, z)");
    }
    
    // Use CoordinateTransformer for coordinate system transformation
    CoordinateTransformer transformer;
    return transformer.transformPoints(points, source_system, target_system);
}

// Transform with pre-validated transformation matrix (faster for repeated transforms)
Eigen::MatrixXd PointTransformer::transformWithMatrix(
    const Eigen::MatrixXd& points,
    const Eigen::Matrix4d& transform_matrix
) {
    if (points.rows() == 0) {
        return points;
    }
    
    if (points.cols() != 3) {
        throw std::invalid_argument("Points matrix must have 3 columns (x, y, z)");
    }
    
    // Use Eigen's vectorized operations for efficiency
    Eigen::MatrixXd transformed(points.rows(), 3);
    
    for (int i = 0; i < points.rows(); ++i) {
        transformed.row(i) = applyTransform(points.row(i), transform_matrix);
    }
    
    return transformed;
}

// Get transformation matrix (for reuse)
Eigen::Matrix4d PointTransformer::getTransformMatrix(
    const CoordinateSystem& source_system,
    const CoordinateSystem& target_system
) {
    // Use CoordinateTransformer to build transformation
    CoordinateTransformer transformer;
    
    // Build inverse transform from source system
    Eigen::Matrix4d source_inv = transformer.buildInverseTransformMatrix(source_system);
    
    // Build transform to target system
    Eigen::Matrix4d target_transform = transformer.buildTransformMatrix(target_system);
    
    // Combined: target_transform * source_inv
    return target_transform * source_inv;
}

} // namespace synchronization
} // namespace am_qadf_native

#endif // EIGEN_AVAILABLE
