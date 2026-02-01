/**
 * C++ unit tests for point_converter (point_converter.cpp).
 * Built when EIGEN_AVAILABLE. Tests pointsToEigenMatrix, eigenMatrixToPoints,
 * queryResultToEigenMatrix.
 *
 * Aligned with docs/Tests/Test_New_Plans.md (tests/unit/cpp/query/).
 */

#ifdef EIGEN_AVAILABLE

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "am_qadf_native/query/point_converter.hpp"
#include "am_qadf_native/query/query_result.hpp"
#include <Eigen/Dense>
#include <array>
#include <vector>

using namespace am_qadf_native::query;

namespace { const double tol = 1e-6; }

TEST_CASE("PointConverter: pointsToEigenMatrix empty", "[query][point_converter]") {
    std::vector<std::array<float, 3>> points;
    Eigen::MatrixXd m = pointsToEigenMatrix(points);
    REQUIRE(m.rows() == 0);
    REQUIRE(m.cols() == 3);
}

TEST_CASE("PointConverter: pointsToEigenMatrix single point", "[query][point_converter]") {
    std::vector<std::array<float, 3>> points = { {1.0f, 2.0f, 3.0f} };
    Eigen::MatrixXd m = pointsToEigenMatrix(points);
    REQUIRE(m.rows() == 1);
    REQUIRE(m.cols() == 3);
    double c0 = m.coeff(0, 0);
    double c1 = m.coeff(0, 1);
    double c2 = m.coeff(0, 2);
    REQUIRE_THAT(c0, Catch::Matchers::WithinAbs(1.0, tol));
    REQUIRE_THAT(c1, Catch::Matchers::WithinAbs(2.0, tol));
    REQUIRE_THAT(c2, Catch::Matchers::WithinAbs(3.0, tol));
}

TEST_CASE("PointConverter: pointsToEigenMatrix multiple points", "[query][point_converter]") {
    std::vector<std::array<float, 3>> points = {
        {0.0f, 0.0f, 0.0f},
        {1.0f, 1.0f, 1.0f}
    };
    Eigen::MatrixXd m = pointsToEigenMatrix(points);
    REQUIRE(m.rows() == 2);
    REQUIRE(m.cols() == 3);
    REQUIRE_THAT(m(1, 0), Catch::Matchers::WithinAbs(1.0, tol));
    REQUIRE_THAT(m(1, 1), Catch::Matchers::WithinAbs(1.0, tol));
    REQUIRE_THAT(m(1, 2), Catch::Matchers::WithinAbs(1.0, tol));
}

TEST_CASE("PointConverter: eigenMatrixToPoints wrong columns throws", "[query][point_converter]") {
    Eigen::MatrixXd m(1, 2);
    m << 1, 2;
    REQUIRE_THROWS_AS(eigenMatrixToPoints(m), std::invalid_argument);
}

TEST_CASE("PointConverter: eigenMatrixToPoints empty", "[query][point_converter]") {
    Eigen::MatrixXd m(0, 3);
    auto points = eigenMatrixToPoints(m);
    REQUIRE(points.empty());
}

TEST_CASE("PointConverter: eigenMatrixToPoints round-trip", "[query][point_converter]") {
    std::vector<std::array<float, 3>> in = { {1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f} };
    Eigen::MatrixXd m = pointsToEigenMatrix(in);
    auto out = eigenMatrixToPoints(m);
    REQUIRE(out.size() == 2);
    REQUIRE(out[0][0] == 1.0f);
    REQUIRE(out[0][1] == 2.0f);
    REQUIRE(out[0][2] == 3.0f);
    REQUIRE(out[1][0] == 4.0f);
    REQUIRE(out[1][1] == 5.0f);
    REQUIRE(out[1][2] == 6.0f);
}

TEST_CASE("PointConverter: queryResultToEigenMatrix", "[query][point_converter]") {
    QueryResult r;
    r.points.push_back({1.0f, 2.0f, 3.0f});
    r.points.push_back({4.0f, 5.0f, 6.0f});
    Eigen::MatrixXd m = queryResultToEigenMatrix(r);
    REQUIRE(m.rows() == 2);
    REQUIRE(m.cols() == 3);
    REQUIRE_THAT(m(0, 0), Catch::Matchers::WithinAbs(1.0, tol));
    REQUIRE_THAT(m(1, 2), Catch::Matchers::WithinAbs(6.0, tol));
}

TEST_CASE("PointConverter: queryResultToEigenMatrix empty result", "[query][point_converter]") {
    QueryResult r;
    Eigen::MatrixXd m = queryResultToEigenMatrix(r);
    REQUIRE(m.rows() == 0);
    REQUIRE(m.cols() == 3);
}

#endif // EIGEN_AVAILABLE
