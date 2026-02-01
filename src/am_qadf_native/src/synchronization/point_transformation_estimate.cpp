#ifdef EIGEN_AVAILABLE

#include "am_qadf_native/synchronization/point_transformation_estimate.hpp"
#include "am_qadf_native/synchronization/point_transformation_sampling.hpp"
#include <Eigen/Geometry>
#include <cmath>
#include <algorithm>
#include <random>
#include <array>
#include <limits>
#include <set>
#include <stdexcept>
#include <vector>

namespace am_qadf_native {
namespace synchronization {

namespace {

// 24 rotational permutations of cube vertex indices 0..7 (same order as BoundingBox::corners()).
// Vertex i has coords ( (i&4)?1:-1, (i&2)?1:-1, (i&1)?1:-1 ). Each permutation P gives ref_reordered[i] = ref[P(i)].
std::vector<std::array<int, 8>> get24CubeRotations() {
    std::set<std::array<int, 8>> seen;
    std::vector<std::array<int, 8>> result;
    for (int ix = 0; ix < 4; ++ix) {
        for (int iy = 0; iy < 4; ++iy) {
            for (int iz = 0; iz < 4; ++iz) {
                Eigen::Matrix3d R = (
                    Eigen::AngleAxisd(iz * M_PI / 2.0, Eigen::Vector3d::UnitZ()) *
                    Eigen::AngleAxisd(iy * M_PI / 2.0, Eigen::Vector3d::UnitY()) *
                    Eigen::AngleAxisd(ix * M_PI / 2.0, Eigen::Vector3d::UnitX())
                ).toRotationMatrix();
                std::array<int, 8> perm;
                for (int i = 0; i < 8; ++i) {
                    double x = (i & 4) ? 1.0 : -1.0;
                    double y = (i & 2) ? 1.0 : -1.0;
                    double z = (i & 1) ? 1.0 : -1.0;
                    Eigen::Vector3d v(x, y, z);
                    Eigen::Vector3d r = R * v;
                    int rx = (r(0) > 0) ? 1 : 0;
                    int ry = (r(1) > 0) ? 1 : 0;
                    int rz = (r(2) > 0) ? 1 : 0;
                    perm[i] = (rx << 2) | (ry << 1) | rz;
                }
                if (seen.insert(perm).second) {
                    result.push_back(perm);
                }
            }
        }
    }
    return result;
}

}  // namespace

// Helper: Apply transformation to point
Eigen::Vector3d TransformationComputer::applyTransform(
    const Eigen::Vector3d& point,
    const Eigen::Matrix4d& transform
) {
    Eigen::Vector4d homogeneous_point;
    homogeneous_point << point, 1.0;
    Eigen::Vector4d transformed = transform * homogeneous_point;
    return transformed.head<3>();
}

// Helper: Compute centroid of point set
Eigen::Vector3d TransformationComputer::computeCentroid(const Eigen::MatrixXd& points) {
    if (points.rows() == 0) {
        return Eigen::Vector3d::Zero();
    }
    return points.colwise().mean();
}

// Helper: Center points (subtract centroid)
Eigen::MatrixXd TransformationComputer::centerPoints(
    const Eigen::MatrixXd& points,
    const Eigen::Vector3d& centroid
) {
    Eigen::MatrixXd centered = points;
    for (int i = 0; i < centered.rows(); ++i) {
        centered.row(i) -= centroid.transpose();
    }
    return centered;
}

// Helper: Check if three points are non-collinear
bool TransformationComputer::areNonCollinear(
    const Eigen::Vector3d& p1,
    const Eigen::Vector3d& p2,
    const Eigen::Vector3d& p3
) {
    Eigen::Vector3d v1 = p2 - p1;
    Eigen::Vector3d v2 = p3 - p1;
    Eigen::Vector3d cross = v1.cross(v2);
    double magnitude = cross.norm();
    return magnitude > 1e-6;
}

// Helper: Compute adaptive threshold (0.5% of coordinate scale, with minimum floor)
double TransformationComputer::computeAdaptiveThreshold(const Eigen::MatrixXd& points) {
    const double min_threshold = 1e-3;  // floor so we never require impossibly tight fit
    if (points.rows() == 0) {
        return min_threshold;
    }
    
    // Compute bounding box
    Eigen::Vector3d min_pt = points.colwise().minCoeff();
    Eigen::Vector3d max_pt = points.colwise().maxCoeff();
    Eigen::Vector3d scale = max_pt - min_pt;
    
    // Use maximum dimension for scale
    double max_scale = scale.maxCoeff();
    
    // Return 0.5% of scale, but at least min_threshold
    double adaptive = 0.005 * max_scale;
    return (adaptive >= min_threshold) ? adaptive : min_threshold;
}

// Helper: Select spatial region for multi-set sampling
TransformationComputer::SpatialRegion TransformationComputer::selectSpatialRegion(
    const Eigen::MatrixXd& source_points,
    const Eigen::MatrixXd& target_points,
    int region_index,
    int total_regions
) {
    if (source_points.rows() != target_points.rows()) {
        throw std::invalid_argument("Source and target point sets must have same number of points");
    }
    
    if (source_points.rows() == 0) {
        throw std::invalid_argument("Point sets cannot be empty");
    }
    
    // Compute bounding box
    Eigen::Vector3d min_pt = source_points.colwise().minCoeff();
    Eigen::Vector3d max_pt = source_points.colwise().maxCoeff();
    Eigen::Vector3d scale = max_pt - min_pt;
    
    // Divide space into grid cells (simple approach: divide each dimension)
    // For 10 regions, use roughly 2x2x3 or 3x3x2 grid
    int grid_x = static_cast<int>(std::ceil(std::cbrt(total_regions)));
    int grid_y = grid_x;
    int grid_z = static_cast<int>(std::ceil(static_cast<double>(total_regions) / (grid_x * grid_y)));
    
    // Compute which cell this region corresponds to
    int cell_x = region_index % grid_x;
    int cell_y = (region_index / grid_x) % grid_y;
    int cell_z = region_index / (grid_x * grid_y);
    
    // Compute cell bounds
    Eigen::Vector3d cell_size = scale.array() / Eigen::Vector3d(grid_x, grid_y, grid_z).array();
    Eigen::Vector3d cell_min = min_pt + Eigen::Vector3d(
        cell_x * cell_size.x(),
        cell_y * cell_size.y(),
        cell_z * cell_size.z()
    );
    Eigen::Vector3d cell_max = cell_min + cell_size;
    
    // Select points within this cell
    std::vector<int> region_indices;
    for (int i = 0; i < source_points.rows(); ++i) {
        Eigen::Vector3d pt = source_points.row(i);
        if (pt.x() >= cell_min.x() && pt.x() < cell_max.x() &&
            pt.y() >= cell_min.y() && pt.y() < cell_max.y() &&
            pt.z() >= cell_min.z() && pt.z() < cell_max.z()) {
            region_indices.push_back(i);
        }
    }
    
    // If cell is empty, fall back to random sampling from entire set
    if (region_indices.empty()) {
        region_indices.resize(std::min(100, static_cast<int>(source_points.rows())));
        std::iota(region_indices.begin(), region_indices.end(), 0);
    }
    
    // Extract region points
    Eigen::MatrixXd region_source(region_indices.size(), 3);
    Eigen::MatrixXd region_target(region_indices.size(), 3);
    for (size_t i = 0; i < region_indices.size(); ++i) {
        region_source.row(i) = source_points.row(region_indices[i]);
        region_target.row(i) = target_points.row(region_indices[i]);
    }
    
    SpatialRegion region;
    region.source_points = region_source;
    region.target_points = region_target;
    return region;
}

// Helper: Sample 3 non-collinear points from region
TransformationComputer::PointSample TransformationComputer::selectNonCollinearSample(
    const Eigen::MatrixXd& source_points,
    const Eigen::MatrixXd& target_points
) {
    if (source_points.rows() < 3) {
        throw std::invalid_argument("Need at least 3 points for sampling");
    }
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, static_cast<int>(source_points.rows()) - 1);
    
    // Try up to 1000 times to find non-collinear points
    for (int attempt = 0; attempt < 1000; ++attempt) {
        int idx1 = dis(gen);
        int idx2 = dis(gen);
        int idx3 = dis(gen);
        
        // Ensure unique indices
        while (idx2 == idx1) idx2 = dis(gen);
        while (idx3 == idx1 || idx3 == idx2) idx3 = dis(gen);
        
        Eigen::Vector3d p1 = source_points.row(idx1);
        Eigen::Vector3d p2 = source_points.row(idx2);
        Eigen::Vector3d p3 = source_points.row(idx3);
        
        if (areNonCollinear(p1, p2, p3)) {
            PointSample sample;
            sample.source.resize(3, 3);
            sample.target.resize(3, 3);
            sample.source.row(0) = source_points.row(idx1);
            sample.source.row(1) = source_points.row(idx2);
            sample.source.row(2) = source_points.row(idx3);
            sample.target.row(0) = target_points.row(idx1);
            sample.target.row(1) = target_points.row(idx2);
            sample.target.row(2) = target_points.row(idx3);
            return sample;
        }
    }
    
    throw std::runtime_error("Failed to find non-collinear points after 1000 attempts");
}

// Helper: Count inliers for a transformation
int TransformationComputer::countInliers(
    const Eigen::MatrixXd& source_points,
    const Eigen::MatrixXd& target_points,
    const Eigen::Matrix4d& transformation,
    double threshold
) {
    if (source_points.rows() != target_points.rows()) {
        throw std::invalid_argument("Source and target point sets must have same number of points");
    }
    
    int inliers = 0;
    for (int i = 0; i < source_points.rows(); ++i) {
        Eigen::Vector3d source_pt = source_points.row(i);
        Eigen::Vector3d transformed_pt = applyTransform(source_pt, transformation);
        Eigen::Vector3d target_pt = target_points.row(i);
        
        double error = (transformed_pt - target_pt).norm();
        if (error <= threshold) {
            inliers++;
        }
    }
    
    return inliers;
}

// Helper: Extract inlier points
TransformationComputer::InlierPoints TransformationComputer::extractInliers(
    const Eigen::MatrixXd& source_points,
    const Eigen::MatrixXd& target_points,
    const Eigen::Matrix4d& transformation,
    double threshold
) {
    if (source_points.rows() != target_points.rows()) {
        throw std::invalid_argument("Source and target point sets must have same number of points");
    }
    
    std::vector<int> inlier_indices;
    for (int i = 0; i < source_points.rows(); ++i) {
        Eigen::Vector3d source_pt = source_points.row(i);
        Eigen::Vector3d transformed_pt = applyTransform(source_pt, transformation);
        Eigen::Vector3d target_pt = target_points.row(i);
        
        double error = (transformed_pt - target_pt).norm();
        if (error <= threshold) {
            inlier_indices.push_back(i);
        }
    }
    
    InlierPoints inliers;
    inliers.source.resize(inlier_indices.size(), 3);
    inliers.target.resize(inlier_indices.size(), 3);
    
    for (size_t i = 0; i < inlier_indices.size(); ++i) {
        inliers.source.row(i) = source_points.row(inlier_indices[i]);
        inliers.target.row(i) = target_points.row(inlier_indices[i]);
    }
    
    return inliers;
}

// Kabsch algorithm for optimal rotation
Eigen::Matrix3d TransformationComputer::computeKabschRotation(
    const Eigen::MatrixXd& source_points,
    const Eigen::MatrixXd& target_points,
    const Eigen::Vector3d& source_centroid,
    const Eigen::Vector3d& target_centroid
) {
    if (source_points.rows() != target_points.rows() || source_points.rows() < 3) {
        throw std::invalid_argument("Need at least 3 corresponding points");
    }
    
    // Center points
    Eigen::MatrixXd source_centered = centerPoints(source_points, source_centroid);
    Eigen::MatrixXd target_centered = centerPoints(target_points, target_centroid);
    
    // Compute cross-covariance matrix
    Eigen::Matrix3d H = (source_centered.transpose() * target_centered) / source_points.rows();
    
    // SVD decomposition
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    
    // Kabsch: R = V * U^T
    Eigen::Matrix3d R = V * U.transpose();
    
    // Ensure proper rotation (det = 1)
    if (R.determinant() < 0) {
        V.col(2) *= -1.0;
        R = V * U.transpose();
    }
    
    return R;
}

// Umeyama's method for uniform scaling
double TransformationComputer::computeUmeyamaScaling(
    const Eigen::MatrixXd& source_points,
    const Eigen::MatrixXd& target_points,
    const Eigen::Matrix3d& rotation,
    const Eigen::Vector3d& source_centroid,
    const Eigen::Vector3d& target_centroid
) {
    if (source_points.rows() != target_points.rows() || source_points.rows() < 3) {
        throw std::invalid_argument("Need at least 3 corresponding points");
    }
    
    // Center points
    Eigen::MatrixXd source_centered = centerPoints(source_points, source_centroid);
    Eigen::MatrixXd target_centered = centerPoints(target_points, target_centroid);
    
    // Compute variance of source points
    double source_variance = source_centered.array().square().sum() / source_points.rows();
    
    if (source_variance < 1e-10) {
        return 1.0;  // No scaling if source has no variance
    }
    
    // Compute cross-covariance matrix
    Eigen::Matrix3d H = (source_centered.transpose() * target_centered) / source_points.rows();
    
    // SVD decomposition
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    
    // Compute trace of S from SVD
    double trace_S = svd.singularValues().sum();
    
    // Uniform scaling
    double scale = trace_S / source_variance;
    
    return scale;
}

// Stage 2: Optimal transformation using Kabsch + Umeyama
Eigen::Matrix4d TransformationComputer::computeOptimalTransformation(
    const Eigen::MatrixXd& source_points,
    const Eigen::MatrixXd& target_points,
    const std::string& method
) {
    if (source_points.rows() != target_points.rows() || source_points.rows() < 3) {
        throw std::invalid_argument("Need at least 3 corresponding points");
    }
    
    if (method != "kabsch_umeyama") {
        throw std::invalid_argument("Only 'kabsch_umeyama' method is currently supported");
    }
    
    // Compute centroids
    Eigen::Vector3d source_centroid = computeCentroid(source_points);
    Eigen::Vector3d target_centroid = computeCentroid(target_points);
    
    // Compute rotation using Kabsch algorithm
    Eigen::Matrix3d rotation = computeKabschRotation(
        source_points, target_points, source_centroid, target_centroid
    );
    
    // Compute uniform scaling using Umeyama's method
    double scale = computeUmeyamaScaling(
        source_points, target_points, rotation, source_centroid, target_centroid
    );
    
    // Compute translation: t = target_centroid - scale * R * source_centroid
    Eigen::Vector3d translation = target_centroid - scale * rotation * source_centroid;
    
    // Build 4x4 transformation matrix
    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
    transform.block<3, 3>(0, 0) = scale * rotation;
    transform.block<3, 1>(0, 3) = translation;
    
    return transform;
}

// Stage 1: RANSAC-based robust estimation (multi-set approach)
RANSACResult TransformationComputer::computeWithRANSAC(
    const Eigen::MatrixXd& source_points,
    const Eigen::MatrixXd& target_points,
    double threshold,
    int min_sets,
    int consensus_threshold,
    double consensus_tolerance,
    int max_iterations,
    double confidence,
    int min_inliers
) {
    if (source_points.rows() != target_points.rows()) {
        throw std::invalid_argument("Source and target point sets must have same number of points");
    }
    
    if (source_points.rows() < 3) {
        throw std::invalid_argument("Need at least 3 points for RANSAC");
    }
    
    // Compute adaptive threshold if not provided (with floor so we get inliers)
    if (threshold <= 0) {
        threshold = computeAdaptiveThreshold(source_points);
    }
    const double min_inlier_threshold = 1e-3;
    if (threshold < min_inlier_threshold) {
        threshold = min_inlier_threshold;
    }
    
    // Consensus tolerance: matrix elements (especially translation) are in data units (e.g. mm).
    // Default 1e-6 is too strict when coordinates are 0..100; use scale-adaptive value so
    // transforms that agree within a fraction of the data scale are grouped.
    const double min_consensus_tolerance = 1e-3;  // floor so we never use impossibly strict 1e-6
    double adaptive = computeAdaptiveThreshold(source_points);
    double effective_consensus_tolerance = consensus_tolerance;
    if (consensus_tolerance <= 1e-3) {
        // Override default or very small tolerance with scale-adaptive (or floor)
        effective_consensus_tolerance = (adaptive >= min_consensus_tolerance) ? adaptive : min_consensus_tolerance;
    } else if (adaptive > consensus_tolerance) {
        effective_consensus_tolerance = adaptive;
    }
    if (effective_consensus_tolerance < min_consensus_tolerance) {
        effective_consensus_tolerance = min_consensus_tolerance;
    }

    std::vector<TransformationCandidate> candidates;
    
    // Sample multiple sets from different spatial regions
    for (int set_idx = 0; set_idx < min_sets; ++set_idx) {
        // Select region (divide point cloud into spatial regions)
        auto region = selectSpatialRegion(source_points, target_points, set_idx, min_sets);
        
        // Sample 3 non-collinear points from this region
        auto sample = selectNonCollinearSample(region.source_points, region.target_points);
        
        // Compute transformation from this set
        Eigen::Matrix4d transform = computeOptimalTransformation(
            sample.source, sample.target, "kabsch_umeyama"
        );
        
        // Count inliers for this transformation
        int inliers = countInliers(
            source_points, target_points, transform, threshold
        );
        
        TransformationCandidate candidate;
        candidate.transformation = transform;
        candidate.num_inliers = inliers;
        candidate.set_index = set_idx;
        candidates.push_back(candidate);
    }
    
    // Find consensus: transformations that agree (within scale-adaptive tolerance)
    auto consensus_groups = findConsensusGroups(candidates, effective_consensus_tolerance);
    
    // Select best consensus group (most sets agreeing)
    auto best_group = selectBestConsensusGroup(consensus_groups);
    
    Eigen::Matrix4d consensus_transform;
    if (static_cast<int>(best_group.size()) >= consensus_threshold) {
        // Compute consensus transformation from agreeing sets
        consensus_transform = computeConsensusTransform(best_group, "median");
    } else {
        // Fallback: use the candidate with the most inliers when consensus is insufficient
        int best_idx = 0;
        int best_inliers = candidates[0].num_inliers;
        for (size_t i = 1; i < candidates.size(); ++i) {
            if (candidates[i].num_inliers > best_inliers) {
                best_inliers = candidates[i].num_inliers;
                best_idx = static_cast<int>(i);
            }
        }
        consensus_transform = candidates[best_idx].transformation;
    }
    
    // Recompute optimal transformation from all inliers of consensus
    auto inlier_points = extractInliers(
        source_points, target_points, consensus_transform, threshold
    );
    
    Eigen::Matrix4d final_transform;
    if (inlier_points.source.rows() >= min_inliers) {
        // Final optimal transformation from all inliers
        final_transform = computeOptimalTransformation(
            inlier_points.source, inlier_points.target, "kabsch_umeyama"
        );
    } else {
        // Fallback: use consensus transform directly and treat all points as inliers
        final_transform = consensus_transform;
        inlier_points.source = source_points;
        inlier_points.target = target_points;
    }
    
    // Build result
    RANSACResult result;
    result.transformation = final_transform;
    result.inlier_source_points = inlier_points.source;
    result.inlier_target_points = inlier_points.target;
    result.num_inliers = inlier_points.source.rows();
    result.confidence = static_cast<double>(best_group.size()) / min_sets;
    
    // Build inlier indices
    result.inlier_indices.clear();
    for (int i = 0; i < source_points.rows(); ++i) {
        Eigen::Vector3d source_pt = source_points.row(i);
        Eigen::Vector3d transformed_pt = applyTransform(source_pt, final_transform);
        Eigen::Vector3d target_pt = target_points.row(i);
        double error = (transformed_pt - target_pt).norm();
        if (error <= threshold) {
            result.inlier_indices.push_back(i);
        }
    }
    
    return result;
}

// Helper: Find consensus groups (transformations that agree)
std::vector<std::vector<TransformationCandidate>> TransformationComputer::findConsensusGroups(
    const std::vector<TransformationCandidate>& candidates,
    double tolerance
) {
    std::vector<std::vector<TransformationCandidate>> groups;
    std::vector<bool> assigned(candidates.size(), false);
    
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (assigned[i]) continue;
        
        std::vector<TransformationCandidate> group;
        group.push_back(candidates[i]);
        assigned[i] = true;
        
        for (size_t j = i + 1; j < candidates.size(); ++j) {
            if (assigned[j]) continue;
            
            // Check if transformations agree (compare transformation matrices)
            Eigen::Matrix4d diff = candidates[i].transformation - candidates[j].transformation;
            double max_diff = diff.cwiseAbs().maxCoeff();
            
            if (max_diff <= tolerance) {
                group.push_back(candidates[j]);
                assigned[j] = true;
            }
        }
        
        groups.push_back(group);
    }
    
    return groups;
}

// Helper: Select best consensus group (most sets agreeing)
std::vector<TransformationCandidate> TransformationComputer::selectBestConsensusGroup(
    const std::vector<std::vector<TransformationCandidate>>& groups
) {
    if (groups.empty()) {
        return std::vector<TransformationCandidate>();
    }
    
    // Find group with most members
    size_t best_idx = 0;
    size_t best_size = groups[0].size();
    
    for (size_t i = 1; i < groups.size(); ++i) {
        if (groups[i].size() > best_size) {
            best_size = groups[i].size();
            best_idx = i;
        }
    }
    
    return groups[best_idx];
}

// Helper: Compute consensus transformation from agreeing sets
Eigen::Matrix4d TransformationComputer::computeConsensusTransform(
    const std::vector<TransformationCandidate>& group,
    const std::string& method
) {
    if (group.empty()) {
        throw std::invalid_argument("Cannot compute consensus from empty group");
    }
    
    if (method == "median") {
        // Simple approach: use median transformation (just use first for now)
        // TODO: Implement proper median computation
        return group[0].transformation;
    } else if (method == "weighted_average") {
        // Weighted average by number of inliers
        Eigen::Matrix4d sum = Eigen::Matrix4d::Zero();
        double total_weight = 0.0;
        
        for (const auto& candidate : group) {
            double weight = static_cast<double>(candidate.num_inliers);
            sum += weight * candidate.transformation;
            total_weight += weight;
        }
        
        if (total_weight > 0) {
            return sum / total_weight;
        } else {
            return group[0].transformation;
        }
    } else {
        throw std::invalid_argument("Unknown consensus method: " + method);
    }
}

// Combined: RANSAC + Optimal (recommended)
Eigen::Matrix4d TransformationComputer::computeRobustTransformation(
    const Eigen::MatrixXd& source_points,
    const Eigen::MatrixXd& target_points,
    TransformationQuality& quality_metrics
) {
    // Stage 1: RANSAC for robust estimation
    RANSACResult ransac_result = computeWithRANSAC(
        source_points, target_points
    );
    
    // Stage 2: Compute quality metrics
    quality_metrics = computeQualityMetrics(
        source_points, target_points, ransac_result.transformation
    );
    
    return ransac_result.transformation;
}

// Refine existing transformation using data
Eigen::Matrix4d TransformationComputer::refineTransformation(
    const Eigen::Matrix4d& initial_transform,
    const Eigen::MatrixXd& source_points,
    const Eigen::MatrixXd& target_points
) {
    // Apply initial transformation to source points
    Eigen::MatrixXd transformed_source(source_points.rows(), 3);
    for (int i = 0; i < source_points.rows(); ++i) {
        transformed_source.row(i) = applyTransform(source_points.row(i), initial_transform);
    }
    
    // Compute refinement transformation (transformed_source -> target_points)
    // This is a small correction, so we can use direct Kabsch+Umeyama
    return computeOptimalTransformation(transformed_source, target_points, "kabsch_umeyama");
}

// Compute quality metrics for transformation
TransformationQuality TransformationComputer::computeQualityMetrics(
    const Eigen::MatrixXd& source_points,
    const Eigen::MatrixXd& target_points,
    const Eigen::Matrix4d& transformation
) {
    if (source_points.rows() != target_points.rows() || source_points.rows() == 0) {
        throw std::invalid_argument("Point sets must have same number of points and be non-empty");
    }
    
    TransformationQuality quality;
    
    // Compute errors for all points
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
    
    // RMS error
    quality.rms_error = std::sqrt(sum_squared_error / source_points.rows());
    
    // Mean error
    quality.mean_error = sum_squared_error / source_points.rows();
    
    // Max error
    quality.max_error = max_err;
    
    // Alignment quality: Score from 0-100% based on RMS error magnitude
    // Quality = max(0.0, 100.0 - RMS_error × 10.0)
    // This assumes RMS error in same units as coordinates
    // For typical manufacturing scales (mm), RMS < 0.1mm gives quality > 99%
    quality.alignment_quality = std::max(0.0, 100.0 - quality.rms_error * 10.0);
    
    // Confidence: Composite score combining alignment quality and error metrics
    // Confidence = max(0.0, min(100.0, alignment_quality - RMS_error × 10.0))
    quality.confidence = std::max(0.0, std::min(100.0, quality.alignment_quality - quality.rms_error * 10.0));
    
    return quality;
}

// Bbox corners: try all 24 rotational mappings (which ref corner = which source corner), then for each
// try all C(8,3)=56 triplets — fit on 3, validate on all 8; return transform with least error.
Eigen::Matrix4d TransformationComputer::computeTransformationFromBboxCorners(
    const Eigen::MatrixXd& source_corners,
    const Eigen::MatrixXd& reference_corners,
    TransformationQuality* quality_out,
    std::vector<BboxFitCandidate>* all_fits_out,
    Eigen::MatrixXd* best_ref_corners_out
) {
    if (source_corners.rows() != 8 || reference_corners.rows() != 8 ||
        source_corners.cols() != 3 || reference_corners.cols() != 3) {
        throw std::invalid_argument(
            "computeTransformationFromBboxCorners: source and reference must be 8x3 (bbox corners)"
        );
    }
    if (all_fits_out) {
        all_fits_out->clear();
    }
    static const std::vector<std::array<int, 8>> rotations = get24CubeRotations();
    Eigen::Matrix4d best_transform = Eigen::Matrix4d::Identity();
    Eigen::MatrixXd best_ref_reordered(8, 3);
    double best_max_error = std::numeric_limits<double>::max();
    TransformationQuality best_quality{};
    int perm_index = 0;
    for (const auto& P : rotations) {
        Eigen::MatrixXd ref_reordered(8, 3);
        for (int i = 0; i < 8; ++i) {
            ref_reordered.row(i) = reference_corners.row(P[i]);
        }
        std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> triplets =
            TransformationSampling::enumerateTripletSamplesFrom8(source_corners, ref_reordered);
        int trip_index = 0;
        for (const auto& pair_3 : triplets) {
            Eigen::Matrix4d T = computeOptimalTransformation(
                pair_3.first, pair_3.second, "kabsch_umeyama"
            );
            TransformationQuality q = computeQualityMetrics(
                source_corners, ref_reordered, T
            );
            if (all_fits_out) {
                all_fits_out->push_back(BboxFitCandidate{
                    perm_index, trip_index,
                    q.max_error, q.mean_error, q.rms_error
                });
            }
            if (q.max_error < best_max_error) {
                best_max_error = q.max_error;
                best_transform = T;
                best_ref_reordered = ref_reordered;
                best_quality = q;
            }
            ++trip_index;
        }
        ++perm_index;
    }
    if (quality_out) {
        *quality_out = best_quality;
    }
    if (best_ref_corners_out) {
        *best_ref_corners_out = best_ref_reordered;
    }
    return best_transform;
}

ScaleTranslationRotation decomposeSimilarityTransform(const Eigen::Matrix4d& M) {
    ScaleTranslationRotation out{};
    out.scale = 1.0;
    out.tx = out.ty = out.tz = 0.0;
    out.rot_x_deg = out.rot_y_deg = out.rot_z_deg = 0.0;
    if (M.rows() != 4 || M.cols() != 4) {
        return out;
    }
    out.tx = M(0, 3);
    out.ty = M(1, 3);
    out.tz = M(2, 3);
    Eigen::Matrix3d S = M.block<3, 3>(0, 0);
    double det = S.determinant();
    if (det <= 0.0) {
        return out;
    }
    double scale = std::cbrt(det);
    if (std::abs(scale) < 1e-12) {
        return out;
    }
    out.scale = scale;
    Eigen::Matrix3d R = S / scale;
    // Euler ZYX: R = Rz * Ry * Rx; extract ry from R(2,0) = -sin(ry)
    double sy = -R(2, 0);
    sy = std::max(-1.0, std::min(1.0, sy));
    double ry_rad = std::asin(sy);
    double cy = std::cos(ry_rad);
    const double eps = 1e-8;
    if (std::abs(cy) < eps) {
        out.rot_x_deg = 0.0;
        out.rot_z_deg = std::atan2(-R(1, 2), R(1, 1)) * (180.0 / M_PI);
    } else {
        out.rot_x_deg = std::atan2(R(2, 1) / cy, R(2, 2) / cy) * (180.0 / M_PI);
        out.rot_z_deg = std::atan2(R(1, 0) / cy, R(0, 0) / cy) * (180.0 / M_PI);
    }
    out.rot_y_deg = ry_rad * (180.0 / M_PI);
    return out;
}

} // namespace synchronization
} // namespace am_qadf_native

#endif // EIGEN_AVAILABLE
