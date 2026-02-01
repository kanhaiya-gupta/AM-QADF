#ifndef AM_QADF_NATIVE_QUERY_POINT_CONVERTER_HPP
#define AM_QADF_NATIVE_QUERY_POINT_CONVERTER_HPP

#ifdef EIGEN_AVAILABLE
#include <Eigen/Dense>
#include "am_qadf_native/query/query_result.hpp"
#include <vector>
#include <array>

namespace am_qadf_native {
namespace query {

// Helper functions to convert QueryResult points to Eigen matrices

// Convert QueryResult.points (std::vector<std::array<float, 3>>) to Eigen::MatrixXd
// Returns matrix with shape (n_points, 3) where each row is a point [x, y, z]
Eigen::MatrixXd pointsToEigenMatrix(const std::vector<std::array<float, 3>>& points);

// Convert Eigen::MatrixXd back to std::vector<std::array<float, 3>>
// Useful for saving transformed points back to MongoDB
std::vector<std::array<float, 3>> eigenMatrixToPoints(const Eigen::MatrixXd& matrix);

// Convert QueryResult to Eigen::MatrixXd (extracts points)
// Convenience function that extracts points from QueryResult
Eigen::MatrixXd queryResultToEigenMatrix(const QueryResult& result);

} // namespace query
} // namespace am_qadf_native

#endif // EIGEN_AVAILABLE
#endif // AM_QADF_NATIVE_QUERY_POINT_CONVERTER_HPP
