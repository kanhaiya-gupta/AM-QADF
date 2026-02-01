#ifdef EIGEN_AVAILABLE

#include "am_qadf_native/query/point_converter.hpp"
#include <stdexcept>

namespace am_qadf_native {
namespace query {

// Convert QueryResult.points to Eigen::MatrixXd
Eigen::MatrixXd pointsToEigenMatrix(const std::vector<std::array<float, 3>>& points) {
    const Eigen::Index n = static_cast<Eigen::Index>(points.size());
    if (n == 0) {
        return Eigen::MatrixXd(0, 3);
    }
    Eigen::MatrixXd result(n, 3);
    result.setZero();
    for (Eigen::Index i = 0; i < n; ++i) {
        const auto& p = points[static_cast<size_t>(i)];
        result(i, 0) = static_cast<double>(p[0]);
        result(i, 1) = static_cast<double>(p[1]);
        result(i, 2) = static_cast<double>(p[2]);
    }
    return result;
}

// Convert Eigen::MatrixXd back to std::vector<std::array<float, 3>>
std::vector<std::array<float, 3>> eigenMatrixToPoints(const Eigen::MatrixXd& matrix) {
    if (matrix.cols() != 3) {
        throw std::invalid_argument("Matrix must have 3 columns (x, y, z)");
    }
    
    std::vector<std::array<float, 3>> points;
    points.reserve(matrix.rows());
    
    for (int i = 0; i < matrix.rows(); ++i) {
        std::array<float, 3> point;
        point[0] = static_cast<float>(matrix(i, 0));
        point[1] = static_cast<float>(matrix(i, 1));
        point[2] = static_cast<float>(matrix(i, 2));
        points.push_back(point);
    }
    
    return points;
}

// Convert QueryResult to Eigen::MatrixXd (extracts points)
Eigen::MatrixXd queryResultToEigenMatrix(const QueryResult& result) {
    return pointsToEigenMatrix(result.points);
}

} // namespace query
} // namespace am_qadf_native

#endif // EIGEN_AVAILABLE
