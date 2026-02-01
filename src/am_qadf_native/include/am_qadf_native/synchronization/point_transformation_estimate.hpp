#ifndef AM_QADF_NATIVE_SYNCHRONIZATION_POINT_TRANSFORMATION_ESTIMATE_HPP
#define AM_QADF_NATIVE_SYNCHRONIZATION_POINT_TRANSFORMATION_ESTIMATE_HPP

#ifdef EIGEN_AVAILABLE
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <vector>
#include <string>

namespace am_qadf_native {
namespace synchronization {

// Result structure for RANSAC transformation computation
struct RANSACResult {
    Eigen::Matrix4d transformation;
    std::vector<int> inlier_indices;
    Eigen::MatrixXd inlier_source_points;
    Eigen::MatrixXd inlier_target_points;
    int num_inliers;
    double confidence;
};

// Quality metrics for transformation validation
struct TransformationQuality {
    double rms_error;
    double alignment_quality;  // 0-100%
    double confidence;         // 0-100%
    double max_error;
    double mean_error;
};

// Per-fit error for one (permutation, triplet) in bbox corner fitting (24 × 56 fits)
struct BboxFitCandidate {
    int permutation_index;  // 0..23
    int triplet_index;      // 0..55
    double max_error;
    double mean_error;
    double rms_error;
};

// Transformation candidate for consensus finding
struct TransformationCandidate {
    Eigen::Matrix4d transformation;
    int num_inliers;
    int set_index;
};

// Decomposition of 4x4 similarity matrix: scale, translation (tx,ty,tz), rotation (Euler ZYX in degrees)
struct ScaleTranslationRotation {
    double scale;
    double tx;
    double ty;
    double tz;
    double rot_x_deg;
    double rot_y_deg;
    double rot_z_deg;
};

// Decompose 4x4 similarity matrix into scale, translation, and rotation (Euler ZYX, degrees).
// M = [s*R | t; 0 0 0 1]. Returns scale, (tx,ty,tz), (rot_x, rot_y, rot_z) in degrees.
ScaleTranslationRotation decomposeSimilarityTransform(const Eigen::Matrix4d& M);

// TransformationComputer: Computes transformation matrices from data points
// Uses RANSAC for robust estimation, Kabsch for optimal rotation, Umeyama for scaling
class TransformationComputer {
public:
    // Stage 1: RANSAC-based robust estimation (multi-set approach)
    // Samples multiple sets from different spatial regions for robust transformation
    RANSACResult computeWithRANSAC(
        const Eigen::MatrixXd& source_points,
        const Eigen::MatrixXd& target_points,
        double threshold = 0.005,  // 0.5% of coordinate scale (adaptive)
        int min_sets = 10,          // At least 10 sets from different regions
        int consensus_threshold = 7, // At least 7 sets must agree
        double consensus_tolerance = 1e-6,  // Tolerance for consensus agreement
        int max_iterations = 1000,
        double confidence = 0.99,
        int min_inliers = 3
    );
    
    // Stage 2: Optimal transformation using Kabsch + Umeyama
    // Computes optimal rigid transformation (rotation + translation + uniform scaling)
    Eigen::Matrix4d computeOptimalTransformation(
        const Eigen::MatrixXd& source_points,
        const Eigen::MatrixXd& target_points,
        const std::string& method = "kabsch_umeyama"
    );
    
    // Combined: RANSAC + Optimal (recommended)
    // Computes robust transformation with quality metrics
    Eigen::Matrix4d computeRobustTransformation(
        const Eigen::MatrixXd& source_points,
        const Eigen::MatrixXd& target_points,
        TransformationQuality& quality_metrics
    );
    
    // Kabsch algorithm for optimal rotation
    // Computes rotation matrix that minimizes RMS error between point sets
    Eigen::Matrix3d computeKabschRotation(
        const Eigen::MatrixXd& source_points,
        const Eigen::MatrixXd& target_points,
        const Eigen::Vector3d& source_centroid,
        const Eigen::Vector3d& target_centroid
    );
    
    // Umeyama's method for uniform scaling
    // Computes optimal uniform scale factor
    double computeUmeyamaScaling(
        const Eigen::MatrixXd& source_points,
        const Eigen::MatrixXd& target_points,
        const Eigen::Matrix3d& rotation,
        const Eigen::Vector3d& source_centroid,
        const Eigen::Vector3d& target_centroid
    );
    
    // Refine existing transformation using data
    // Uses RANSAC + Kabsch to refine an initial transformation
    Eigen::Matrix4d refineTransformation(
        const Eigen::Matrix4d& initial_transform,
        const Eigen::MatrixXd& source_points,
        const Eigen::MatrixXd& target_points
    );
    
    // Compute quality metrics for transformation
    // Returns RMS error, alignment quality, confidence, max/mean error
    TransformationQuality computeQualityMetrics(
        const Eigen::MatrixXd& source_points,
        const Eigen::MatrixXd& target_points,
        const Eigen::Matrix4d& transformation
    );

    // Bbox corners (8x3 each): 24 permutations × 56 triplets — fit on 3, validate on 8; return transform with least error.
    // If all_fits_out is non-null, append one BboxFitCandidate per fit.
    // If best_ref_corners_out is non-null, set to the reference corners reordered by the best permutation
    //   (so validation must use this order: T*source_corners vs best_ref_corners_out).
    Eigen::Matrix4d computeTransformationFromBboxCorners(
        const Eigen::MatrixXd& source_corners,
        const Eigen::MatrixXd& reference_corners,
        TransformationQuality* quality_out = nullptr,
        std::vector<BboxFitCandidate>* all_fits_out = nullptr,
        Eigen::MatrixXd* best_ref_corners_out = nullptr
    );

private:
    // Helper: Check if three points are non-collinear
    // Returns true if cross-product magnitude > 1e-6
    bool areNonCollinear(
        const Eigen::Vector3d& p1,
        const Eigen::Vector3d& p2,
        const Eigen::Vector3d& p3
    );
    
    // Helper: Compute adaptive threshold (0.5% of coordinate scale)
    double computeAdaptiveThreshold(const Eigen::MatrixXd& points);
    
    // Helper: Select spatial region for multi-set sampling
    struct SpatialRegion {
        Eigen::MatrixXd source_points;
        Eigen::MatrixXd target_points;
    };
    SpatialRegion selectSpatialRegion(
        const Eigen::MatrixXd& source_points,
        const Eigen::MatrixXd& target_points,
        int region_index,
        int total_regions
    );
    
    // Helper: Sample 3 non-collinear points from region
    struct PointSample {
        Eigen::MatrixXd source;  // 3x3 matrix (3 points, 3 coords each)
        Eigen::MatrixXd target;  // 3x3 matrix
    };
    PointSample selectNonCollinearSample(
        const Eigen::MatrixXd& source_points,
        const Eigen::MatrixXd& target_points
    );
    
    // Helper: Count inliers for a transformation
    int countInliers(
        const Eigen::MatrixXd& source_points,
        const Eigen::MatrixXd& target_points,
        const Eigen::Matrix4d& transformation,
        double threshold
    );
    
    // Helper: Extract inlier points
    struct InlierPoints {
        Eigen::MatrixXd source;
        Eigen::MatrixXd target;
    };
    InlierPoints extractInliers(
        const Eigen::MatrixXd& source_points,
        const Eigen::MatrixXd& target_points,
        const Eigen::Matrix4d& transformation,
        double threshold
    );
    
    // Helper: Find consensus groups (transformations that agree)
    std::vector<std::vector<TransformationCandidate>> findConsensusGroups(
        const std::vector<TransformationCandidate>& candidates,
        double tolerance
    );
    
    // Helper: Select best consensus group (most sets agreeing)
    std::vector<TransformationCandidate> selectBestConsensusGroup(
        const std::vector<std::vector<TransformationCandidate>>& groups
    );
    
    // Helper: Compute consensus transformation from agreeing sets
    Eigen::Matrix4d computeConsensusTransform(
        const std::vector<TransformationCandidate>& group,
        const std::string& method = "median"  // "median" or "weighted_average"
    );
    
    // Helper: Apply transformation to point
    Eigen::Vector3d applyTransform(
        const Eigen::Vector3d& point,
        const Eigen::Matrix4d& transform
    );
    
    // Helper: Compute centroid of point set
    Eigen::Vector3d computeCentroid(const Eigen::MatrixXd& points);
    
    // Helper: Center points (subtract centroid)
    Eigen::MatrixXd centerPoints(
        const Eigen::MatrixXd& points,
        const Eigen::Vector3d& centroid
    );
};

} // namespace synchronization
} // namespace am_qadf_native

#endif // EIGEN_AVAILABLE
#endif // AM_QADF_NATIVE_SYNCHRONIZATION_POINT_TRANSFORMATION_ESTIMATE_HPP
