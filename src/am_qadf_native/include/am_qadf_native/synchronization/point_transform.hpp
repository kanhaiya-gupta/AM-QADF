#ifndef AM_QADF_NATIVE_SYNCHRONIZATION_POINT_TRANSFORM_HPP
#define AM_QADF_NATIVE_SYNCHRONIZATION_POINT_TRANSFORM_HPP

#ifdef EIGEN_AVAILABLE
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include "am_qadf_native/synchronization/point_coordinate_transform.hpp"

namespace am_qadf_native {
namespace synchronization {

// PointTransformer: Efficient point transformation using Eigen
// Transforms point coordinates using pre-computed transformation matrices
class PointTransformer {
public:
    // Transform points from source to target coordinate system
    // Uses CoordinateTransformer internally
    Eigen::MatrixXd transform(
        const Eigen::MatrixXd& points,
        const CoordinateSystem& source_system,
        const CoordinateSystem& target_system
    );
    
    // Transform with pre-validated transformation matrix (faster for repeated transforms)
    // This is the recommended method for bulk transformation after validation
    Eigen::MatrixXd transformWithMatrix(
        const Eigen::MatrixXd& points,
        const Eigen::Matrix4d& transform_matrix
    );
    
    // Get transformation matrix (for reuse)
    // Builds transformation matrix from coordinate systems
    Eigen::Matrix4d getTransformMatrix(
        const CoordinateSystem& source_system,
        const CoordinateSystem& target_system
    );

private:
    // Helper: Apply transformation to point
    Eigen::Vector3d applyTransform(
        const Eigen::Vector3d& point,
        const Eigen::Matrix4d& transform
    );
};

} // namespace synchronization
} // namespace am_qadf_native

#endif // EIGEN_AVAILABLE
#endif // AM_QADF_NATIVE_SYNCHRONIZATION_POINT_TRANSFORM_HPP
