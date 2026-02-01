#ifndef AM_QADF_NATIVE_SYNCHRONIZATION_POINT_COORDINATE_TRANSFORM_HPP
#define AM_QADF_NATIVE_SYNCHRONIZATION_POINT_COORDINATE_TRANSFORM_HPP

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <string>

namespace am_qadf_native {
namespace synchronization {

// Coordinate system representation
struct CoordinateSystem {
    Eigen::Vector3d origin = Eigen::Vector3d::Zero();
    Eigen::Vector3d rotation_euler = Eigen::Vector3d::Zero();  // radians
    Eigen::Vector3d scale = Eigen::Vector3d::Ones();
    
    // Alternative: axis-angle rotation
    Eigen::Vector3d rotation_axis = Eigen::Vector3d::UnitZ();
    double rotation_angle = 0.0;  // radians
    bool use_axis_angle = false;
};

// Coordinate system transformer using Eigen
class CoordinateTransformer {
public:
    // Transform single point between coordinate systems
    Eigen::Vector3d transformPoint(
        const Eigen::Vector3d& point,
        const CoordinateSystem& from_system,
        const CoordinateSystem& to_system
    );
    
    // Transform batch of points (vectorized)
    // Input: points matrix (N, 3) where each row is a point
    Eigen::MatrixXd transformPoints(
        const Eigen::MatrixXd& points,
        const CoordinateSystem& from_system,
        const CoordinateSystem& to_system
    );
    
    // Build 4x4 homogeneous transformation matrix from coordinate system
    Eigen::Matrix4d buildTransformMatrix(const CoordinateSystem& system);
    
    // Build inverse transformation matrix
    Eigen::Matrix4d buildInverseTransformMatrix(const CoordinateSystem& system);
    
    // Compute 3x3 rotation matrix from Euler angles (ZYX order)
    Eigen::Matrix3d rotationMatrixFromEuler(
        double rx, double ry, double rz  // radians
    );
    
    // Compute 3x3 rotation matrix from axis-angle representation
    Eigen::Matrix3d rotationMatrixFromAxisAngle(
        const Eigen::Vector3d& axis,
        double angle  // radians
    );
    
    // Validate coordinate system
    bool validateCoordinateSystem(const CoordinateSystem& system);
    
private:
    // Helper: Apply transformation to point
    Eigen::Vector3d applyTransform(
        const Eigen::Vector3d& point,
        const CoordinateSystem& system
    );
    
    // Helper: Apply inverse transformation to point
    Eigen::Vector3d applyInverseTransform(
        const Eigen::Vector3d& point,
        const CoordinateSystem& system
    );
};

} // namespace synchronization
} // namespace am_qadf_native

#endif // AM_QADF_NATIVE_SYNCHRONIZATION_POINT_COORDINATE_TRANSFORM_HPP
