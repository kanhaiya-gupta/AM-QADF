#ifndef AM_QADF_NATIVE_SYNCHRONIZATION_HATCHING_SAMPLING_HPP
#define AM_QADF_NATIVE_SYNCHRONIZATION_HATCHING_SAMPLING_HPP

#ifdef EIGEN_AVAILABLE
#include <Eigen/Dense>
#include <vector>

namespace am_qadf_native {
namespace synchronization {

/**
 * Build a point set from hatching vectors using only (x1, y1, z) — one point
 * per segment — and stratify sampling by z so that points come from different
 * layers. This avoids collinearity when the same z has many collinear segment
 * endpoints.
 *
 * @param x1  x-coordinate of segment start (one per vector)
 * @param y1  y-coordinate of segment start (one per vector)
 * @param z   z-height / layer (one per vector)
 * @param max_per_layer  Maximum points to keep per distinct z (default 500)
 * @param max_total      Maximum total points to return (default 10000), 0 = no limit
 * @return  Eigen::MatrixXd of shape (n, 3) with rows (x1, y1, z), stratified across z
 */
Eigen::MatrixXd sample_hatching_points_by_z(
    const std::vector<double>& x1,
    const std::vector<double>& y1,
    const std::vector<double>& z,
    int max_per_layer = 500,
    int max_total = 10000
);

/**
 * Overload for float vectors (converts to double internally).
 */
Eigen::MatrixXd sample_hatching_points_by_z(
    const std::vector<float>& x1,
    const std::vector<float>& y1,
    const std::vector<float>& z,
    int max_per_layer = 500,
    int max_total = 10000
);

}  // namespace synchronization
}  // namespace am_qadf_native

#endif  // EIGEN_AVAILABLE
#endif  // AM_QADF_NATIVE_SYNCHRONIZATION_HATCHING_SAMPLING_HPP
