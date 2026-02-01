#ifdef EIGEN_AVAILABLE

#include "am_qadf_native/synchronization/point_bounds.hpp"
#include "am_qadf_native/synchronization/point_transform.hpp"
#include "am_qadf_native/synchronization/point_coordinate_transform.hpp"  // Includes full CoordinateSystem definition
#include <limits>
#include <stdexcept>

namespace am_qadf_native {
namespace synchronization {

// BoundingBox implementation
void BoundingBox::expand(const Eigen::Vector3d& point) {
    if (!isValid()) {
        // Initialize from first point
        min_x = max_x = point.x();
        min_y = max_y = point.y();
        min_z = max_z = point.z();
    } else {
        min_x = std::min(min_x, point.x());
        min_y = std::min(min_y, point.y());
        min_z = std::min(min_z, point.z());
        max_x = std::max(max_x, point.x());
        max_y = std::max(max_y, point.y());
        max_z = std::max(max_z, point.z());
    }
}

void BoundingBox::expand(const BoundingBox& other) {
    if (!other.isValid()) {
        return;  // Nothing to expand with
    }
    
    if (!isValid()) {
        // Initialize from other
        *this = other;
    } else {
        min_x = std::min(min_x, other.min_x);
        min_y = std::min(min_y, other.min_y);
        min_z = std::min(min_z, other.min_z);
        max_x = std::max(max_x, other.max_x);
        max_y = std::max(max_y, other.max_y);
        max_z = std::max(max_z, other.max_z);
    }
}

bool BoundingBox::contains(const Eigen::Vector3d& point) const {
    if (!isValid()) {
        return false;
    }
    return point.x() >= min_x && point.x() <= max_x &&
           point.y() >= min_y && point.y() <= max_y &&
           point.z() >= min_z && point.z() <= max_z;
}

// 8 corners in fixed order for correspondence: (min,min,min) .. (max,max,max)
Eigen::MatrixXd BoundingBox::corners() const {
    Eigen::MatrixXd out(8, 3);
    if (!isValid()) {
        out.setZero();
        return out;
    }
    out(0, 0) = min_x; out(0, 1) = min_y; out(0, 2) = min_z;
    out(1, 0) = min_x; out(1, 1) = min_y; out(1, 2) = max_z;
    out(2, 0) = min_x; out(2, 1) = max_y; out(2, 2) = min_z;
    out(3, 0) = min_x; out(3, 1) = max_y; out(3, 2) = max_z;
    out(4, 0) = max_x; out(4, 1) = min_y; out(4, 2) = min_z;
    out(5, 0) = max_x; out(5, 1) = min_y; out(5, 2) = max_z;
    out(6, 0) = max_x; out(6, 1) = max_y; out(6, 2) = min_z;
    out(7, 0) = max_x; out(7, 1) = max_y; out(7, 2) = max_z;
    return out;
}

// Helper: Initialize bounding box from first point
void UnifiedBoundsComputer::initializeFromPoint(BoundingBox& bounds, const Eigen::Vector3d& point) {
    bounds.min_x = bounds.max_x = point.x();
    bounds.min_y = bounds.max_y = point.y();
    bounds.min_z = bounds.max_z = point.z();
}

// Helper: Compute bounds from single point set
BoundingBox UnifiedBoundsComputer::computeBoundsFromPoints(const Eigen::MatrixXd& points) {
    if (points.rows() == 0) {
        return BoundingBox();
    }
    
    if (points.cols() != 3) {
        throw std::invalid_argument("Points matrix must have 3 columns (x, y, z)");
    }
    
    // Compute min/max in each dimension
    Eigen::Vector3d min_pt = points.colwise().minCoeff();
    Eigen::Vector3d max_pt = points.colwise().maxCoeff();
    
    return BoundingBox(
        min_pt.x(), min_pt.y(), min_pt.z(),
        max_pt.x(), max_pt.y(), max_pt.z()
    );
}

// Compute union bounds from transformed points
BoundingBox UnifiedBoundsComputer::computeUnionBounds(
    const std::vector<Eigen::MatrixXd>& transformed_point_sets
) {
    if (transformed_point_sets.empty()) {
        return BoundingBox();
    }
    
    BoundingBox union_bounds;
    bool initialized = false;
    
    for (const auto& points : transformed_point_sets) {
        if (points.rows() == 0) {
            continue;  // Skip empty point sets
        }
        
        BoundingBox bounds = computeBoundsFromPoints(points);
        
        if (!initialized) {
            union_bounds = bounds;
            initialized = true;
        } else {
            union_bounds.expand(bounds);
        }
    }
    
    return union_bounds;
}

// Incremental bounds expansion
BoundingBox UnifiedBoundsComputer::computeIncremental(
    const std::vector<Eigen::MatrixXd>& point_sets,
    const std::vector<CoordinateSystem>& source_systems,
    const CoordinateSystem& target_system
) {
    if (point_sets.size() != source_systems.size()) {
        throw std::invalid_argument(
            "Number of point sets must match number of source coordinate systems"
        );
    }
    
    if (point_sets.empty()) {
        return BoundingBox();
    }
    
    PointTransformer transformer;
    BoundingBox union_bounds;
    bool initialized = false;
    
    for (size_t i = 0; i < point_sets.size(); ++i) {
        if (point_sets[i].rows() == 0) {
            continue;  // Skip empty point sets
        }
        
        // Transform points to target coordinate system
        Eigen::MatrixXd transformed = transformer.transform(
            point_sets[i], source_systems[i], target_system
        );
        
        // Compute bounds from transformed points
        BoundingBox bounds = computeBoundsFromPoints(transformed);
        
        if (!initialized) {
            union_bounds = bounds;
            initialized = true;
        } else {
            union_bounds.expand(bounds);
        }
    }
    
    return union_bounds;
}

// Add padding to bounds
BoundingBox UnifiedBoundsComputer::addPadding(const BoundingBox& bounds, double padding) {
    if (!bounds.isValid()) {
        return bounds;
    }
    
    if (padding < 0) {
        throw std::invalid_argument("Padding must be non-negative");
    }
    
    BoundingBox padded = bounds;
    padded.min_x -= padding;
    padded.min_y -= padding;
    padded.min_z -= padding;
    padded.max_x += padding;
    padded.max_y += padding;
    padded.max_z += padding;
    
    return padded;
}

// Add percentage-based padding
BoundingBox UnifiedBoundsComputer::addPercentagePadding(
    const BoundingBox& bounds, 
    double padding_percent
) {
    if (!bounds.isValid()) {
        return bounds;
    }
    
    if (padding_percent < 0) {
        throw std::invalid_argument("Padding percentage must be non-negative");
    }
    
    Eigen::Vector3d size = bounds.size();
    double padding_x = size.x() * padding_percent / 100.0;
    double padding_y = size.y() * padding_percent / 100.0;
    double padding_z = size.z() * padding_percent / 100.0;
    
    BoundingBox padded = bounds;
    padded.min_x -= padding_x;
    padded.min_y -= padding_y;
    padded.min_z -= padding_z;
    padded.max_x += padding_x;
    padded.max_y += padding_y;
    padded.max_z += padding_z;
    
    return padded;
}

} // namespace synchronization
} // namespace am_qadf_native

#endif // EIGEN_AVAILABLE
