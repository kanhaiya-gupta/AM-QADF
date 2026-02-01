/**
 * C++ unit tests for PointTransformer (point_transform.cpp).
 * Built only when EIGEN_AVAILABLE. Tests transform, transformWithMatrix, getTransformMatrix.
 *
 * Aligned with src/am_qadf_native/synchronization/point_transform.hpp
 */

#ifdef EIGEN_AVAILABLE

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "am_qadf_native/synchronization/point_transform.hpp"
#include "am_qadf_native/synchronization/point_coordinate_transform.hpp"
#include <Eigen/Dense>
#include <cmath>

using namespace am_qadf_native::synchronization;

namespace {

const double tol = 1e-10;

} // namespace

// ---------------------------------------------------------------------------
// getTransformMatrix
// ---------------------------------------------------------------------------

TEST_CASE("PointTransformer: getTransformMatrix identity systems", "[synchronization][point_transform]") {
    PointTransformer trans;
    CoordinateSystem src;
    CoordinateSystem tgt;
    Eigen::Matrix4d M = trans.getTransformMatrix(src, tgt);
    REQUIRE(M.rows() == 4);
    REQUIRE(M.cols() == 4);
    REQUIRE_THAT(M(0, 0), Catch::Matchers::WithinAbs(1.0, tol));
    REQUIRE_THAT(M(1, 1), Catch::Matchers::WithinAbs(1.0, tol));
    REQUIRE_THAT(M(2, 2), Catch::Matchers::WithinAbs(1.0, tol));
    REQUIRE_THAT(M(3, 3), Catch::Matchers::WithinAbs(1.0, tol));
}

TEST_CASE("PointTransformer: getTransformMatrix translation", "[synchronization][point_transform]") {
    PointTransformer trans;
    CoordinateSystem src;
    CoordinateSystem tgt;
    tgt.origin = Eigen::Vector3d(1, 2, 3);
    Eigen::Matrix4d M = trans.getTransformMatrix(src, tgt);
    REQUIRE_THAT(M(0, 3), Catch::Matchers::WithinAbs(1.0, tol));
    REQUIRE_THAT(M(1, 3), Catch::Matchers::WithinAbs(2.0, tol));
    REQUIRE_THAT(M(2, 3), Catch::Matchers::WithinAbs(3.0, tol));
}

// ---------------------------------------------------------------------------
// transform (source -> target coordinate systems)
// ---------------------------------------------------------------------------

TEST_CASE("PointTransformer: transform empty points returns empty", "[synchronization][point_transform]") {
    PointTransformer trans;
    Eigen::MatrixXd points(0, 3);
    CoordinateSystem src, tgt;
    Eigen::MatrixXd result = trans.transform(points, src, tgt);
    REQUIRE(result.rows() == 0);
    REQUIRE(result.cols() == 3);
}

TEST_CASE("PointTransformer: transform identity systems preserves points", "[synchronization][point_transform]") {
    PointTransformer trans;
    Eigen::MatrixXd points(2, 3);
    points << 1, 2, 3, 4, 5, 6;
    CoordinateSystem src, tgt;
    Eigen::MatrixXd result = trans.transform(points, src, tgt);
    REQUIRE(result.rows() == 2);
    REQUIRE(result.cols() == 3);
    REQUIRE_THAT(result(0, 0), Catch::Matchers::WithinAbs(1.0, tol));
    REQUIRE_THAT(result(0, 1), Catch::Matchers::WithinAbs(2.0, tol));
    REQUIRE_THAT(result(0, 2), Catch::Matchers::WithinAbs(3.0, tol));
    REQUIRE_THAT(result(1, 0), Catch::Matchers::WithinAbs(4.0, tol));
    REQUIRE_THAT(result(1, 1), Catch::Matchers::WithinAbs(5.0, tol));
    REQUIRE_THAT(result(1, 2), Catch::Matchers::WithinAbs(6.0, tol));
}

TEST_CASE("PointTransformer: transform wrong columns throws", "[synchronization][point_transform]") {
    PointTransformer trans;
    Eigen::MatrixXd points(1, 2);
    points << 1, 2;
    CoordinateSystem src, tgt;
    REQUIRE_THROWS_AS(trans.transform(points, src, tgt), std::invalid_argument);
}

TEST_CASE("PointTransformer: transform with translation", "[synchronization][point_transform]") {
    PointTransformer trans;
    Eigen::MatrixXd points(1, 3);
    points << 0, 0, 0;
    CoordinateSystem src;
    CoordinateSystem tgt;
    tgt.origin = Eigen::Vector3d(10, 20, 30);
    Eigen::MatrixXd result = trans.transform(points, src, tgt);
    REQUIRE_THAT(result(0, 0), Catch::Matchers::WithinAbs(10.0, tol));
    REQUIRE_THAT(result(0, 1), Catch::Matchers::WithinAbs(20.0, tol));
    REQUIRE_THAT(result(0, 2), Catch::Matchers::WithinAbs(30.0, tol));
}

// ---------------------------------------------------------------------------
// transformWithMatrix
// ---------------------------------------------------------------------------

TEST_CASE("PointTransformer: transformWithMatrix identity", "[synchronization][point_transform]") {
    PointTransformer trans;
    Eigen::MatrixXd points(1, 3);
    points << 1, 2, 3;
    Eigen::Matrix4d M = Eigen::Matrix4d::Identity();
    Eigen::MatrixXd result = trans.transformWithMatrix(points, M);
    REQUIRE(result.rows() == 1);
    REQUIRE(result.cols() == 3);
    REQUIRE_THAT(result(0, 0), Catch::Matchers::WithinAbs(1.0, tol));
    REQUIRE_THAT(result(0, 1), Catch::Matchers::WithinAbs(2.0, tol));
    REQUIRE_THAT(result(0, 2), Catch::Matchers::WithinAbs(3.0, tol));
}

TEST_CASE("PointTransformer: transformWithMatrix translation", "[synchronization][point_transform]") {
    PointTransformer trans;
    Eigen::MatrixXd points(1, 3);
    points << 0, 0, 0;
    Eigen::Matrix4d M = Eigen::Matrix4d::Identity();
    M(0, 3) = 5;
    M(1, 3) = 10;
    M(2, 3) = 15;
    Eigen::MatrixXd result = trans.transformWithMatrix(points, M);
    REQUIRE_THAT(result(0, 0), Catch::Matchers::WithinAbs(5.0, tol));
    REQUIRE_THAT(result(0, 1), Catch::Matchers::WithinAbs(10.0, tol));
    REQUIRE_THAT(result(0, 2), Catch::Matchers::WithinAbs(15.0, tol));
}

TEST_CASE("PointTransformer: transformWithMatrix empty returns empty", "[synchronization][point_transform]") {
    PointTransformer trans;
    Eigen::MatrixXd points(0, 3);
    Eigen::Matrix4d M = Eigen::Matrix4d::Identity();
    Eigen::MatrixXd result = trans.transformWithMatrix(points, M);
    REQUIRE(result.rows() == 0);
    REQUIRE(result.cols() == 3);
}

TEST_CASE("PointTransformer: transformWithMatrix wrong columns throws", "[synchronization][point_transform]") {
    PointTransformer trans;
    Eigen::MatrixXd points(1, 2);
    points << 1, 2;
    Eigen::Matrix4d M = Eigen::Matrix4d::Identity();
    REQUIRE_THROWS_AS(trans.transformWithMatrix(points, M), std::invalid_argument);
}

#endif // EIGEN_AVAILABLE
