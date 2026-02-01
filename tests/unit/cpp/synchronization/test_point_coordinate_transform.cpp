/**
 * C++ unit tests for CoordinateTransformer (point_coordinate_transform.cpp).
 * Built when EIGEN_AVAILABLE. Tests transformPoint, transformPoints,
 * buildTransformMatrix, buildInverseTransformMatrix, rotationMatrixFromEuler,
 * rotationMatrixFromAxisAngle, validateCoordinateSystem.
 */

#ifdef EIGEN_AVAILABLE

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "am_qadf_native/synchronization/point_coordinate_transform.hpp"
#include <Eigen/Dense>
#include <cmath>

using namespace am_qadf_native::synchronization;

namespace { const double tol = 1e-10; }

TEST_CASE("CoordinateTransformer: transformPoint identity", "[synchronization][point_coordinate_transform]") {
    CoordinateTransformer trans;
    CoordinateSystem src, tgt;
    Eigen::Vector3d p(1, 2, 3);
    Eigen::Vector3d out = trans.transformPoint(p, src, tgt);
    REQUIRE_THAT(out.x(), Catch::Matchers::WithinAbs(1.0, tol));
    REQUIRE_THAT(out.y(), Catch::Matchers::WithinAbs(2.0, tol));
    REQUIRE_THAT(out.z(), Catch::Matchers::WithinAbs(3.0, tol));
}

TEST_CASE("CoordinateTransformer: transformPoints empty returns empty", "[synchronization][point_coordinate_transform]") {
    CoordinateTransformer trans;
    Eigen::MatrixXd points(0, 3);
    CoordinateSystem src, tgt;
    Eigen::MatrixXd out = trans.transformPoints(points, src, tgt);
    REQUIRE(out.rows() == 0);
    REQUIRE(out.cols() == 3);
}

TEST_CASE("CoordinateTransformer: transformPoints wrong columns throws", "[synchronization][point_coordinate_transform]") {
    CoordinateTransformer trans;
    Eigen::MatrixXd points(1, 2);
    points << 1, 2;
    CoordinateSystem src, tgt;
    REQUIRE_THROWS_AS(trans.transformPoints(points, src, tgt), std::invalid_argument);
}

TEST_CASE("CoordinateTransformer: transformPoints identity", "[synchronization][point_coordinate_transform]") {
    CoordinateTransformer trans;
    Eigen::MatrixXd points(2, 3);
    points << 1, 2, 3, 4, 5, 6;
    CoordinateSystem src, tgt;
    Eigen::MatrixXd out = trans.transformPoints(points, src, tgt);
    REQUIRE(out.rows() == 2);
    REQUIRE(out.cols() == 3);
    REQUIRE_THAT(out(0, 0), Catch::Matchers::WithinAbs(1.0, tol));
    REQUIRE_THAT(out(1, 2), Catch::Matchers::WithinAbs(6.0, tol));
}

TEST_CASE("CoordinateTransformer: buildTransformMatrix identity", "[synchronization][point_coordinate_transform]") {
    CoordinateTransformer trans;
    CoordinateSystem sys;
    Eigen::Matrix4d M = trans.buildTransformMatrix(sys);
    REQUIRE_THAT(M(0, 0), Catch::Matchers::WithinAbs(1.0, tol));
    REQUIRE_THAT(M(3, 3), Catch::Matchers::WithinAbs(1.0, tol));
}

TEST_CASE("CoordinateTransformer: buildTransformMatrix translation", "[synchronization][point_coordinate_transform]") {
    CoordinateTransformer trans;
    CoordinateSystem sys;
    sys.origin = Eigen::Vector3d(10, 20, 30);
    Eigen::Matrix4d M = trans.buildTransformMatrix(sys);
    REQUIRE_THAT(M(0, 3), Catch::Matchers::WithinAbs(10.0, tol));
    REQUIRE_THAT(M(1, 3), Catch::Matchers::WithinAbs(20.0, tol));
    REQUIRE_THAT(M(2, 3), Catch::Matchers::WithinAbs(30.0, tol));
}

TEST_CASE("CoordinateTransformer: buildInverseTransformMatrix identity", "[synchronization][point_coordinate_transform]") {
    CoordinateTransformer trans;
    CoordinateSystem sys;
    Eigen::Matrix4d M = trans.buildInverseTransformMatrix(sys);
    REQUIRE_THAT(M(0, 0), Catch::Matchers::WithinAbs(1.0, tol));
    REQUIRE_THAT(M(3, 3), Catch::Matchers::WithinAbs(1.0, tol));
}

TEST_CASE("CoordinateTransformer: rotationMatrixFromEuler zeros", "[synchronization][point_coordinate_transform]") {
    CoordinateTransformer trans;
    Eigen::Matrix3d R = trans.rotationMatrixFromEuler(0, 0, 0);
    REQUIRE_THAT(R(0, 0), Catch::Matchers::WithinAbs(1.0, tol));
    REQUIRE_THAT(R(1, 1), Catch::Matchers::WithinAbs(1.0, tol));
    REQUIRE_THAT(R(2, 2), Catch::Matchers::WithinAbs(1.0, tol));
}

TEST_CASE("CoordinateTransformer: rotationMatrixFromAxisAngle", "[synchronization][point_coordinate_transform]") {
    CoordinateTransformer trans;
    Eigen::Matrix3d R = trans.rotationMatrixFromAxisAngle(Eigen::Vector3d::UnitZ(), 0);
    REQUIRE_THAT(R(0, 0), Catch::Matchers::WithinAbs(1.0, tol));
    REQUIRE_THAT(R(2, 2), Catch::Matchers::WithinAbs(1.0, tol));
}

TEST_CASE("CoordinateTransformer: validateCoordinateSystem default", "[synchronization][point_coordinate_transform]") {
    CoordinateTransformer trans;
    CoordinateSystem sys;
    REQUIRE(trans.validateCoordinateSystem(sys));
}

#endif // EIGEN_AVAILABLE
