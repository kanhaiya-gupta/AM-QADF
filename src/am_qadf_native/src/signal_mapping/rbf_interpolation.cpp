#include "am_qadf_native/signal_mapping/rbf_interpolation.hpp"
#include "am_qadf_native/signal_mapping/interpolation_base.hpp"
#include <openvdb/openvdb.h>
#include <openvdb/math/Coord.h>
#include <openvdb/math/Vec3.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>
#include <stdexcept>

#ifdef EIGEN_AVAILABLE
#include <Eigen/Dense>
#endif

namespace am_qadf_native {
namespace signal_mapping {

RBFMapper::RBFMapper(const std::string& kernel_type, float epsilon)
    : kernel_type_(kernel_type), epsilon_(epsilon) {
    if (epsilon_ <= 0.0f) {
        throw std::invalid_argument("RBF epsilon must be positive");
    }
}

void RBFMapper::setKernelType(const std::string& kernel_type) {
    kernel_type_ = kernel_type;
}

void RBFMapper::setEpsilon(float epsilon) {
    if (epsilon <= 0.0f) {
        throw std::invalid_argument("RBF epsilon must be positive");
    }
    epsilon_ = epsilon;
}

float RBFMapper::rbfKernel(float distance) const {
    if (kernel_type_ == "gaussian") {
        return std::exp(-epsilon_ * distance * distance);
    } else if (kernel_type_ == "multiquadric") {
        return std::sqrt(1.0f + epsilon_ * epsilon_ * distance * distance);
    } else if (kernel_type_ == "thin_plate") {
        if (distance > 0.0f) {
            return distance * distance * std::log(distance);
        }
        return 0.0f;
    }
    // Default: Gaussian
    return std::exp(-epsilon_ * distance * distance);
}

void RBFMapper::solveRBFSystem(
    const std::vector<Point>& points,
    const std::vector<float>& values,
    std::vector<float>& weights
) {
#ifdef EIGEN_AVAILABLE
    const size_t n = points.size();
    if (n == 0) {
        weights.clear();
        return;
    }
    
    // Build RBF kernel matrix A[i,j] = phi(||x_i - x_j||)
    Eigen::MatrixXf A(n, n);
    
    // Helper function to compute distance
    auto dist = [](const Point& a, const Point& b) -> float {
        float dx = a.x - b.x;
        float dy = a.y - b.y;
        float dz = a.z - b.z;
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    };
    
    // Fill matrix
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float d = dist(points[i], points[j]);
            A(i, j) = rbfKernel(d);
        }
    }
    
    // Build right-hand side vector
    Eigen::VectorXf f(n);
    for (size_t i = 0; i < n; ++i) {
        f(i) = values[i];
    }
    
    // Solve linear system: A * w = f
    Eigen::VectorXf w = A.colPivHouseholderQr().solve(f);
    
    // Copy weights to output vector
    weights.resize(n);
    for (size_t i = 0; i < n; ++i) {
        weights[i] = w(i);
    }
#else
    // Fallback: If Eigen not available, use identity weights (nearest neighbor)
    weights.resize(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        weights[i] = values[i];
    }
    throw std::runtime_error("RBF interpolation requires Eigen library. Please ensure EIGEN_AVAILABLE is defined.");
#endif
}

void RBFMapper::map(
    FloatGridPtr grid,
    const std::vector<Point>& points,
    const std::vector<float>& values
) {
    if (points.size() != values.size()) {
        throw std::invalid_argument("Points and values size mismatch");
    }
    if (points.empty()) {
        return;  // No points to interpolate
    }
    
#ifdef EIGEN_AVAILABLE
    // Step 1: Solve RBF system to get weights
    std::vector<float> weights;
    solveRBFSystem(points, values, weights);
    
    // Step 2: Get grid bounding box
    openvdb::CoordBBox bbox = grid->evalActiveVoxelBoundingBox();
    if (bbox.empty()) {
        // Compute bounding box from points
        float min_x = points[0].x, max_x = points[0].x;
        float min_y = points[0].y, max_y = points[0].y;
        float min_z = points[0].z, max_z = points[0].z;
        
        for (const auto& pt : points) {
            min_x = std::min(min_x, pt.x);
            max_x = std::max(max_x, pt.x);
            min_y = std::min(min_y, pt.y);
            max_y = std::max(max_y, pt.y);
            min_z = std::min(min_z, pt.z);
            max_z = std::max(max_z, pt.z);
        }
        
        // Expand bbox slightly
        // voxelSize() returns Vec3<double>, extract x component (typically uniform)
        float margin = static_cast<float>(grid->voxelSize().x()) * 2.0f;
        min_x -= margin; max_x += margin;
        min_y -= margin; max_y += margin;
        min_z -= margin; max_z += margin;
        
        // Convert to index coordinates
        auto min_coord = grid->transform().worldToIndexCellCentered(
            openvdb::Vec3R(min_x, min_y, min_z)
        );
        auto max_coord = grid->transform().worldToIndexCellCentered(
            openvdb::Vec3R(max_x, max_y, max_z)
        );
        
        bbox = openvdb::CoordBBox(
            openvdb::Coord(
                static_cast<int>(std::floor(min_coord.x())),
                static_cast<int>(std::floor(min_coord.y())),
                static_cast<int>(std::floor(min_coord.z()))
            ),
            openvdb::Coord(
                static_cast<int>(std::ceil(max_coord.x())),
                static_cast<int>(std::ceil(max_coord.y())),
                static_cast<int>(std::ceil(max_coord.z()))
            )
        );
    }
    
    // Step 3: Evaluate RBF at each voxel: f(x) = sum(w_i * phi(||x - x_i||))
    auto& tree = grid->tree();
    const auto& transform = grid->transform();
    
    // Helper function to compute distance
    auto dist = [](const openvdb::Vec3R& a, const Point& b) -> float {
        float dx = static_cast<float>(a.x() - b.x);
        float dy = static_cast<float>(a.y() - b.y);
        float dz = static_cast<float>(a.z() - b.z);
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    };
    
    for (auto iter = bbox.begin(); iter; ++iter) {
        openvdb::Coord voxel_coord = *iter;
        
        // Convert voxel coordinate to world coordinate
        openvdb::Vec3R world_pos = transform.indexToWorld(voxel_coord);
        
        // Evaluate RBF: f(x) = sum(w_i * phi(||x - x_i||))
        float interpolated_value = 0.0f;
        for (size_t i = 0; i < points.size(); ++i) {
            float d = dist(world_pos, points[i]);
            interpolated_value += weights[i] * rbfKernel(d);
        }
        
        tree.setValue(voxel_coord, interpolated_value);
    }
#else
    throw std::runtime_error("RBF interpolation requires Eigen library. Please ensure EIGEN_AVAILABLE is defined.");
#endif
}

} // namespace signal_mapping
} // namespace am_qadf_native
