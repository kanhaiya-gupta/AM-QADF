/**
 * C++ unit tests for TransformationComputer and decomposeSimilarityTransform
 * (point_transformation_estimate.cpp). Built when EIGEN_AVAILABLE.
 */

#ifdef EIGEN_AVAILABLE

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "am_qadf_native/synchronization/point_transformation_estimate.hpp"
#include <Eigen/Dense>

using namespace am_qadf_native::synchronization;

namespace { const double tol = 1e-8; }

TEST_CASE("decomposeSimilarityTransform: identity", "[synchronization][point_transformation_estimate]") {
    Eigen::Matrix4d M = Eigen::Matrix4d::Identity();
    ScaleTranslationRotation r = decomposeSimilarityTransform(M);
    REQUIRE_THAT(r.scale, Catch::Matchers::WithinAbs(1.0, tol));
    REQUIRE_THAT(r.tx, Catch::Matchers::WithinAbs(0.0, tol));
    REQUIRE_THAT(r.ty, Catch::Matchers::WithinAbs(0.0, tol));
    REQUIRE_THAT(r.tz, Catch::Matchers::WithinAbs(0.0, tol));
}

TEST_CASE("TransformationComputer: computeOptimalTransformation identical points", "[synchronization][point_transformation_estimate]") {
    TransformationComputer comp;
    Eigen::MatrixXd src(3, 3);
    src << 0, 0, 0, 1, 0, 0, 0, 1, 0;
    Eigen::MatrixXd tgt = src;
    Eigen::Matrix4d T = comp.computeOptimalTransformation(src, tgt, "kabsch_umeyama");
    REQUIRE(T.rows() == 4);
    REQUIRE(T.cols() == 4);
    REQUIRE_THAT(T(0, 0), Catch::Matchers::WithinAbs(1.0, 0.1));
    REQUIRE_THAT(T(3, 3), Catch::Matchers::WithinAbs(1.0, tol));
}

TEST_CASE("TransformationComputer: computeQualityMetrics identity", "[synchronization][point_transformation_estimate]") {
    TransformationComputer comp;
    Eigen::MatrixXd src(2, 3);
    src << 0, 0, 0, 1, 1, 1;
    Eigen::MatrixXd tgt = src;
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    TransformationQuality q = comp.computeQualityMetrics(src, tgt, T);
    REQUIRE_THAT(q.rms_error, Catch::Matchers::WithinAbs(0.0, tol));
    REQUIRE_THAT(q.max_error, Catch::Matchers::WithinAbs(0.0, tol));
}

TEST_CASE("TransformationComputer: computeKabschRotation same points", "[synchronization][point_transformation_estimate]") {
    TransformationComputer comp;
    Eigen::MatrixXd src(3, 3);
    src << 0, 0, 0, 1, 0, 0, 0, 1, 0;
    Eigen::MatrixXd tgt = src;
    Eigen::Vector3d sc = src.colwise().mean();
    Eigen::Vector3d tc = tgt.colwise().mean();
    Eigen::Matrix3d R = comp.computeKabschRotation(src, tgt, sc, tc);
    REQUIRE(R.rows() == 3);
    REQUIRE(R.cols() == 3);
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    REQUIRE((R - I).cwiseAbs().maxCoeff() < 0.1);
}

#endif // EIGEN_AVAILABLE
