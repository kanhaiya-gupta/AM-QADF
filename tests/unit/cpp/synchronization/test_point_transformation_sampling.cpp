/**
 * C++ unit tests for TransformationSampling (point_transformation_sampling.cpp).
 * Built when EIGEN_AVAILABLE. Tests enumerateTripletSamplesFrom8.
 */

#ifdef EIGEN_AVAILABLE

#include <catch2/catch_test_macros.hpp>
#include "am_qadf_native/synchronization/point_transformation_sampling.hpp"
#include <Eigen/Dense>

using namespace am_qadf_native::synchronization;

TEST_CASE("TransformationSampling: enumerateTripletSamplesFrom8 wrong rows throws", "[synchronization][point_transformation_sampling]") {
    Eigen::MatrixXd src(7, 3);
    Eigen::MatrixXd tgt(8, 3);
    src.setZero();
    tgt.setZero();
    REQUIRE_THROWS_AS(
        TransformationSampling::enumerateTripletSamplesFrom8(src, tgt),
        std::invalid_argument
    );
}

TEST_CASE("TransformationSampling: enumerateTripletSamplesFrom8 wrong cols throws", "[synchronization][point_transformation_sampling]") {
    Eigen::MatrixXd src(8, 2);
    Eigen::MatrixXd tgt(8, 3);
    src.setZero();
    tgt.setZero();
    REQUIRE_THROWS_AS(
        TransformationSampling::enumerateTripletSamplesFrom8(src, tgt),
        std::invalid_argument
    );
}

TEST_CASE("TransformationSampling: enumerateTripletSamplesFrom8 returns 56 pairs", "[synchronization][point_transformation_sampling]") {
    Eigen::MatrixXd src(8, 3);
    Eigen::MatrixXd tgt(8, 3);
    src.setZero();
    tgt.setZero();
    auto pairs = TransformationSampling::enumerateTripletSamplesFrom8(src, tgt);
    REQUIRE(pairs.size() == 56);  // C(8,3) = 56
    for (const auto& p : pairs) {
        REQUIRE(p.first.rows() == 3);
        REQUIRE(p.first.cols() == 3);
        REQUIRE(p.second.rows() == 3);
        REQUIRE(p.second.cols() == 3);
    }
}

#endif // EIGEN_AVAILABLE
