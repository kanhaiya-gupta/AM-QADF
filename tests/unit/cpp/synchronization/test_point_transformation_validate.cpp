/**
 * C++ unit tests for TransformationValidator (point_transformation_validate.cpp).
 * Built when EIGEN_AVAILABLE. Tests validate, validateWithMatrix,
 * validateBboxCornersAndCentre.
 */

#ifdef EIGEN_AVAILABLE

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "am_qadf_native/synchronization/point_transformation_validate.hpp"
#include "am_qadf_native/synchronization/point_bounds.hpp"
#include <Eigen/Dense>
#include <vector>

using namespace am_qadf_native::synchronization;

namespace { const double tol = 1e-9; }

TEST_CASE("TransformationValidator: validate empty sample points invalid", "[synchronization][point_transformation_validate]") {
    TransformationValidator val;
    CoordinateSystem src, tgt;
    std::vector<Eigen::Vector3d> points;
    ValidationResult r = val.validate(src, tgt, points, 1e-9);
    REQUIRE_FALSE(r.isValid);
    REQUIRE_FALSE(r.errors.empty());
}

TEST_CASE("TransformationValidator: validateWithMatrix identity", "[synchronization][point_transformation_validate]") {
    TransformationValidator val;
    Eigen::MatrixXd src(2, 3);
    src << 0, 0, 0, 1, 1, 1;
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    Eigen::MatrixXd tgt = src;
    ValidationResult r = val.validateWithMatrix(src, T, tgt, 1e-9);
    REQUIRE(r.isValid);
    REQUIRE_THAT(r.rms_error, Catch::Matchers::WithinAbs(0.0, 1e-6));
}

TEST_CASE("TransformationValidator: validateBboxCornersAndCentre identity", "[synchronization][point_transformation_validate]") {
    TransformationValidator val;
    BoundingBox box(0, 0, 0, 1, 1, 1);
    Eigen::MatrixXd corners = box.corners();
    REQUIRE(corners.rows() == 8);
    REQUIRE(corners.cols() == 3);
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    BboxCorrespondenceValidation r = val.validateBboxCornersAndCentre(corners, corners, T);
    REQUIRE(r.num_pairs == 9);
    REQUIRE_THAT(r.mean_distance, Catch::Matchers::WithinAbs(0.0, 1e-6));
    REQUIRE_THAT(r.max_distance, Catch::Matchers::WithinAbs(0.0, 1e-6));
}

#endif // EIGEN_AVAILABLE
