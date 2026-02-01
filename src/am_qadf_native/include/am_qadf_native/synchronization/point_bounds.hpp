#ifndef AM_QADF_NATIVE_SYNCHRONIZATION_POINT_BOUNDS_HPP
#define AM_QADF_NATIVE_SYNCHRONIZATION_POINT_BOUNDS_HPP

#ifdef EIGEN_AVAILABLE
#include <Eigen/Dense>
#include "am_qadf_native/synchronization/point_coordinate_transform.hpp"
#include <vector>

namespace am_qadf_native {
namespace synchronization {

// Bounding box structure
struct BoundingBox {
    double min_x, min_y, min_z;
    double max_x, max_y, max_z;
    
    // Default: empty (min > max) so isValid() is false; distinguishes from single point at origin (0,0,0,0,0,0)
    BoundingBox() : min_x(1.0), min_y(1.0), min_z(1.0),
                    max_x(0.0), max_y(0.0), max_z(0.0) {}
    
    BoundingBox(double min_x, double min_y, double min_z,
                double max_x, double max_y, double max_z)
        : min_x(min_x), min_y(min_y), min_z(min_z),
          max_x(max_x), max_y(max_y), max_z(max_z) {}
    
    void expand(const Eigen::Vector3d& point);
    void expand(const BoundingBox& other);
    bool contains(const Eigen::Vector3d& point) const;
    
    double width() const { return max_x - min_x; }
    double height() const { return max_y - min_y; }
    double depth() const { return max_z - min_z; }
    
    Eigen::Vector3d center() const {
        return Eigen::Vector3d(
            (min_x + max_x) / 2.0,
            (min_y + max_y) / 2.0,
            (min_z + max_z) / 2.0
        );
    }
    
    Eigen::Vector3d size() const {
        return Eigen::Vector3d(width(), height(), depth());
    }
    
    bool isValid() const {
        return min_x <= max_x && min_y <= max_y && min_z <= max_z;
    }

    // Return 8 corner points in fixed order for correspondence (source corner i = same physical corner as target corner i).
    // Order: (min,min,min), (min,min,max), (min,max,min), (min,max,max), (max,min,min), (max,min,max), (max,max,min), (max,max,max).
    Eigen::MatrixXd corners() const;
};

// UnifiedBoundsComputer: Computes union bounding box from transformed points
class UnifiedBoundsComputer {
public:
    // Compute union bounds from transformed points
    // All points should already be in the unified coordinate system
    BoundingBox computeUnionBounds(
        const std::vector<Eigen::MatrixXd>& transformed_point_sets
    );

    // Compute bounding box from a single point set (rows = points, cols = x,y,z).
    BoundingBox computeBoundsFromPoints(const Eigen::MatrixXd& points);
    
    // Incremental bounds expansion
    // Transforms points on-the-fly and expands bounds incrementally
    BoundingBox computeIncremental(
        const std::vector<Eigen::MatrixXd>& point_sets,
        const std::vector<CoordinateSystem>& source_systems,
        const CoordinateSystem& target_system
    );
    
    // Add padding to bounds
    // Adds padding to all sides of the bounding box
    BoundingBox addPadding(const BoundingBox& bounds, double padding);
    
    // Add percentage-based padding
    // Adds padding as percentage of each dimension
    BoundingBox addPercentagePadding(const BoundingBox& bounds, double padding_percent);

private:
    // Helper: Initialize bounding box from first point
    void initializeFromPoint(BoundingBox& bounds, const Eigen::Vector3d& point);
};

} // namespace synchronization
} // namespace am_qadf_native

#endif // EIGEN_AVAILABLE
#endif // AM_QADF_NATIVE_SYNCHRONIZATION_POINT_BOUNDS_HPP
