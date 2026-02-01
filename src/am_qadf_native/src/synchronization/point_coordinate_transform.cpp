#include "am_qadf_native/synchronization/point_coordinate_transform.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace am_qadf_native {
namespace synchronization {

Eigen::Vector3d CoordinateTransformer::transformPoint(
    const Eigen::Vector3d& point,
    const CoordinateSystem& from_system,
    const CoordinateSystem& to_system
) {
    // Step 1: Convert from from_system's local coordinates to "normalized" coordinates
    // Apply A's inverse transform: T⁻¹ → R⁻¹ → S⁻¹
    Eigen::Vector3d p = applyInverseTransform(point, from_system);
    
    // Step 2: Convert from "normalized" to to_system's local coordinates
    // Apply B's transform: S → R → T
    p = applyTransform(p, to_system);
    
    return p;
}

Eigen::MatrixXd CoordinateTransformer::transformPoints(
    const Eigen::MatrixXd& points,
    const CoordinateSystem& from_system,
    const CoordinateSystem& to_system
) {
    if (points.rows() == 0) {
        return points;
    }
    
    if (points.cols() != 3) {
        throw std::invalid_argument("Points matrix must have 3 columns (x, y, z)");
    }
    
    Eigen::MatrixXd transformed = points;
    
    // Step 1: Apply inverse transform from from_system
    for (int i = 0; i < transformed.rows(); ++i) {
        transformed.row(i) = applyInverseTransform(transformed.row(i), from_system);
    }
    
    // Step 2: Apply transform to to_system
    for (int i = 0; i < transformed.rows(); ++i) {
        transformed.row(i) = applyTransform(transformed.row(i), to_system);
    }
    
    return transformed;
}

Eigen::Matrix4d CoordinateTransformer::buildTransformMatrix(const CoordinateSystem& system) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    
    // Build rotation matrix
    Eigen::Matrix3d R;
    if (system.use_axis_angle) {
        R = rotationMatrixFromAxisAngle(system.rotation_axis, system.rotation_angle);
    } else {
        R = rotationMatrixFromEuler(
            system.rotation_euler.x(),
            system.rotation_euler.y(),
            system.rotation_euler.z()
        );
    }
    
    // Build scale matrix
    Eigen::Matrix3d S = Eigen::Matrix3d::Identity();
    S(0, 0) = system.scale.x();
    S(1, 1) = system.scale.y();
    S(2, 2) = system.scale.z();
    
    // Build translation vector
    Eigen::Vector3d t = system.origin;
    
    // Combined transformation: T = [R*S | t; 0 0 0 | 1]
    T.block<3, 3>(0, 0) = R * S;
    T.block<3, 1>(0, 3) = t;
    
    return T;
}

Eigen::Matrix4d CoordinateTransformer::buildInverseTransformMatrix(const CoordinateSystem& system) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    
    // Build inverse rotation matrix
    Eigen::Matrix3d R;
    if (system.use_axis_angle) {
        R = rotationMatrixFromAxisAngle(system.rotation_axis, system.rotation_angle);
    } else {
        R = rotationMatrixFromEuler(
            system.rotation_euler.x(),
            system.rotation_euler.y(),
            system.rotation_euler.z()
        );
    }
    Eigen::Matrix3d R_inv = R.transpose();  // For rotation matrices, inverse = transpose
    
    // Build inverse scale matrix
    Eigen::Matrix3d S_inv = Eigen::Matrix3d::Identity();
    // Avoid division by zero
    double sx = (std::abs(system.scale.x()) > 1e-10) ? system.scale.x() : 1.0;
    double sy = (std::abs(system.scale.y()) > 1e-10) ? system.scale.y() : 1.0;
    double sz = (std::abs(system.scale.z()) > 1e-10) ? system.scale.z() : 1.0;
    S_inv(0, 0) = 1.0 / sx;
    S_inv(1, 1) = 1.0 / sy;
    S_inv(2, 2) = 1.0 / sz;
    
    // Build inverse translation
    Eigen::Vector3d t_inv = -R_inv * S_inv * system.origin;
    
    // Combined inverse transformation: T⁻¹ = [S⁻¹*R⁻¹ | t_inv; 0 0 0 | 1]
    T.block<3, 3>(0, 0) = S_inv * R_inv;
    T.block<3, 1>(0, 3) = t_inv;
    
    return T;
}

Eigen::Matrix3d CoordinateTransformer::rotationMatrixFromEuler(
    double rx, double ry, double rz
) {
    // Rotation matrices around each axis
    Eigen::Matrix3d Rx;
    Rx << 1, 0, 0,
          0, std::cos(rx), -std::sin(rx),
          0, std::sin(rx), std::cos(rx);
    
    Eigen::Matrix3d Ry;
    Ry << std::cos(ry), 0, std::sin(ry),
          0, 1, 0,
          -std::sin(ry), 0, std::cos(ry);
    
    Eigen::Matrix3d Rz;
    Rz << std::cos(rz), -std::sin(rz), 0,
          std::sin(rz), std::cos(rz), 0,
          0, 0, 1;
    
    // Combined rotation: Rz * Ry * Rx (ZYX Euler angles)
    return Rz * Ry * Rx;
}

Eigen::Matrix3d CoordinateTransformer::rotationMatrixFromAxisAngle(
    const Eigen::Vector3d& axis,
    double angle
) {
    // Normalize axis
    Eigen::Vector3d n = axis.normalized();
    
    // Rodrigues' rotation formula
    double c = std::cos(angle);
    double s = std::sin(angle);
    double t = 1.0 - c;
    
    Eigen::Matrix3d R;
    R << t * n.x() * n.x() + c,           t * n.x() * n.y() - s * n.z(), t * n.x() * n.z() + s * n.y(),
         t * n.x() * n.y() + s * n.z(),   t * n.y() * n.y() + c,         t * n.y() * n.z() - s * n.x(),
         t * n.x() * n.z() - s * n.y(),   t * n.y() * n.z() + s * n.x(), t * n.z() * n.z() + c;
    
    return R;
}

bool CoordinateTransformer::validateCoordinateSystem(const CoordinateSystem& system) {
    // Check that origin is valid (finite)
    if (!system.origin.allFinite()) {
        return false;
    }
    
    // Check that scale is valid and non-zero
    if (!system.scale.allFinite() || system.scale.minCoeff() <= 0.0) {
        return false;
    }
    
    // Check rotation
    if (system.use_axis_angle) {
        if (!system.rotation_axis.allFinite() || !std::isfinite(system.rotation_angle)) {
            return false;
        }
    } else {
        if (!system.rotation_euler.allFinite()) {
            return false;
        }
    }
    
    return true;
}

Eigen::Vector3d CoordinateTransformer::applyTransform(
    const Eigen::Vector3d& point,
    const CoordinateSystem& system
) {
    Eigen::Vector3d p = point;
    
    // Apply scaling (multiply by scale) - FIRST
    p = p.cwiseProduct(system.scale);
    
    // Apply rotation - SECOND
    if (system.use_axis_angle) {
        Eigen::Matrix3d R = rotationMatrixFromAxisAngle(system.rotation_axis, system.rotation_angle);
        p = R * p;
    } else {
        Eigen::Matrix3d R = rotationMatrixFromEuler(
            system.rotation_euler.x(),
            system.rotation_euler.y(),
            system.rotation_euler.z()
        );
        p = R * p;
    }
    
    // Apply translation (add origin) - LAST
    p = p + system.origin;
    
    return p;
}

Eigen::Vector3d CoordinateTransformer::applyInverseTransform(
    const Eigen::Vector3d& point,
    const CoordinateSystem& system
) {
    Eigen::Vector3d p = point;
    
    // Check if system is identity
    bool is_identity = system.origin.isZero() && 
                       system.rotation_euler.isZero() && 
                       system.scale.isOnes() &&
                       !system.use_axis_angle;
    
    if (is_identity) {
        return p;
    }
    
    // Apply inverse translation (subtract origin) - FIRST
    p = p - system.origin;
    
    // Apply inverse rotation - SECOND
    if (system.use_axis_angle) {
        Eigen::Matrix3d R = rotationMatrixFromAxisAngle(system.rotation_axis, system.rotation_angle);
        Eigen::Matrix3d R_inv = R.transpose();
        p = R_inv * p;
    } else {
        Eigen::Matrix3d R = rotationMatrixFromEuler(
            system.rotation_euler.x(),
            system.rotation_euler.y(),
            system.rotation_euler.z()
        );
        Eigen::Matrix3d R_inv = R.transpose();
        p = R_inv * p;
    }
    
    // Apply inverse scaling (divide by scale) - LAST
    // Avoid division by zero
    Eigen::Vector3d scale_safe = system.scale.cwiseMax(Eigen::Vector3d::Constant(1e-10));
    p = p.cwiseQuotient(scale_safe);
    
    return p;
}

} // namespace synchronization
} // namespace am_qadf_native
