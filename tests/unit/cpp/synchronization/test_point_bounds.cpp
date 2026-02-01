/**
 * C++ unit tests for BoundingBox and UnifiedBoundsComputer (point_bounds.cpp).
 * Built only when EIGEN_AVAILABLE. Tests expand, contains, width/height/depth,
 * center, size, isValid, corners, computeUnionBounds, computeBoundsFromPoints,
 * addPadding, addPercentagePadding; optional computeIncremental.
 *
 * Aligned with src/am_qadf_native/synchronization/point_bounds.hpp
 */

#ifdef EIGEN_AVAILABLE

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "am_qadf_native/synchronization/point_bounds.hpp"
#include "am_qadf_native/synchronization/point_coordinate_transform.hpp"
#include <Eigen/Dense>
#include <vector>

using namespace am_qadf_native::synchronization;

// ---------------------------------------------------------------------------
// BoundingBox
// ---------------------------------------------------------------------------

TEST_CASE("BoundingBox: default constructor", "[synchronization][point_bounds]") {
    BoundingBox b;
    REQUIRE_FALSE(b.isValid());
    REQUIRE(b.min_x > b.max_x);  // empty sentinel
}

TEST_CASE("BoundingBox: parameterized constructor", "[synchronization][point_bounds]") {
    BoundingBox b(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
    REQUIRE(b.min_x == 1.0);
    REQUIRE(b.min_y == 2.0);
    REQUIRE(b.min_z == 3.0);
    REQUIRE(b.max_x == 4.0);
    REQUIRE(b.max_y == 5.0);
    REQUIRE(b.max_z == 6.0);
    REQUIRE(b.isValid());
    REQUIRE(b.width() == 3.0);
    REQUIRE(b.height() == 3.0);
    REQUIRE(b.depth() == 3.0);
}

TEST_CASE("BoundingBox: expand with point", "[synchronization][point_bounds]") {
    BoundingBox b(0, 0, 0, 1, 1, 1);
    b.expand(Eigen::Vector3d(2, 0.5, -1));
    REQUIRE(b.min_x == 0.0);
    REQUIRE(b.min_y == 0.0);
    REQUIRE(b.min_z == -1.0);
    REQUIRE(b.max_x == 2.0);
    REQUIRE(b.max_y == 1.0);
    REQUIRE(b.max_z == 1.0);
}

TEST_CASE("BoundingBox: expand invalid with point initializes", "[synchronization][point_bounds]") {
    BoundingBox b;
    REQUIRE_FALSE(b.isValid());
    b.expand(Eigen::Vector3d(1, 2, 3));
    REQUIRE(b.isValid());
    REQUIRE(b.min_x == 1.0);
    REQUIRE(b.max_x == 1.0);
    REQUIRE(b.min_y == 2.0);
    REQUIRE(b.max_y == 2.0);
    REQUIRE(b.min_z == 3.0);
    REQUIRE(b.max_z == 3.0);
}

TEST_CASE("BoundingBox: expand with other box", "[synchronization][point_bounds]") {
    BoundingBox b(0, 0, 0, 1, 1, 1);
    BoundingBox other(2, 1, -1, 3, 2, 0);
    b.expand(other);
    REQUIRE(b.min_x == 0.0);
    REQUIRE(b.min_y == 0.0);
    REQUIRE(b.min_z == -1.0);
    REQUIRE(b.max_x == 3.0);
    REQUIRE(b.max_y == 2.0);
    REQUIRE(b.max_z == 1.0);
}

TEST_CASE("BoundingBox: contains", "[synchronization][point_bounds]") {
    BoundingBox b(0, 0, 0, 2, 2, 2);
    REQUIRE(b.contains(Eigen::Vector3d(1, 1, 1)));
    REQUIRE(b.contains(Eigen::Vector3d(0, 0, 0)));
    REQUIRE(b.contains(Eigen::Vector3d(2, 2, 2)));
    REQUIRE_FALSE(b.contains(Eigen::Vector3d(3, 1, 1)));
    REQUIRE_FALSE(b.contains(Eigen::Vector3d(-0.1, 1, 1)));
}

TEST_CASE("BoundingBox: contains invalid returns false", "[synchronization][point_bounds]") {
    BoundingBox b;
    REQUIRE_FALSE(b.contains(Eigen::Vector3d(0, 0, 0)));
}

TEST_CASE("BoundingBox: center and size", "[synchronization][point_bounds]") {
    BoundingBox b(0, 0, 0, 2, 4, 6);
    auto c = b.center();
    REQUIRE_THAT(c.x(), Catch::Matchers::WithinAbs(1.0, 1e-10));
    REQUIRE_THAT(c.y(), Catch::Matchers::WithinAbs(2.0, 1e-10));
    REQUIRE_THAT(c.z(), Catch::Matchers::WithinAbs(3.0, 1e-10));
    auto s = b.size();
    REQUIRE(s.x() == 2.0);
    REQUIRE(s.y() == 4.0);
    REQUIRE(s.z() == 6.0);
}

TEST_CASE("BoundingBox: corners returns 8x3", "[synchronization][point_bounds]") {
    BoundingBox b(0, 0, 0, 1, 1, 1);
    Eigen::MatrixXd corners = b.corners();
    REQUIRE(corners.rows() == 8);
    REQUIRE(corners.cols() == 3);
    // Use coeff and locals to avoid Eigen expression / ABI issues when crossing shared lib boundary
    double c00 = corners.coeff(0, 0), c01 = corners.coeff(0, 1), c02 = corners.coeff(0, 2);
    double c70 = corners.coeff(7, 0), c71 = corners.coeff(7, 1), c72 = corners.coeff(7, 2);
    REQUIRE_THAT(c00, Catch::Matchers::WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(c01, Catch::Matchers::WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(c02, Catch::Matchers::WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(c70, Catch::Matchers::WithinAbs(1.0, 1e-10));
    REQUIRE_THAT(c71, Catch::Matchers::WithinAbs(1.0, 1e-10));
    REQUIRE_THAT(c72, Catch::Matchers::WithinAbs(1.0, 1e-10));
}

// ---------------------------------------------------------------------------
// UnifiedBoundsComputer
// ---------------------------------------------------------------------------

TEST_CASE("UnifiedBoundsComputer: computeUnionBounds empty returns invalid", "[synchronization][point_bounds]") {
    UnifiedBoundsComputer comp;
    std::vector<Eigen::MatrixXd> sets;
    BoundingBox b = comp.computeUnionBounds(sets);
    REQUIRE_FALSE(b.isValid());
}

TEST_CASE("UnifiedBoundsComputer: computeBoundsFromPoints empty returns invalid", "[synchronization][point_bounds]") {
    UnifiedBoundsComputer comp;
    Eigen::MatrixXd points(0, 3);
    BoundingBox b = comp.computeBoundsFromPoints(points);
    REQUIRE_FALSE(b.isValid());
}

TEST_CASE("UnifiedBoundsComputer: computeBoundsFromPoints single point", "[synchronization][point_bounds]") {
    UnifiedBoundsComputer comp;
    Eigen::MatrixXd points(1, 3);
    points << 1.0, 2.0, 3.0;
    BoundingBox b = comp.computeBoundsFromPoints(points);
    REQUIRE(b.isValid());
    REQUIRE(b.min_x == 1.0);
    REQUIRE(b.max_x == 1.0);
    REQUIRE(b.min_y == 2.0);
    REQUIRE(b.max_y == 2.0);
    REQUIRE(b.min_z == 3.0);
    REQUIRE(b.max_z == 3.0);
}

TEST_CASE("UnifiedBoundsComputer: computeBoundsFromPoints multiple points", "[synchronization][point_bounds]") {
    UnifiedBoundsComputer comp;
    Eigen::MatrixXd points(3, 3);
    points << 0, 0, 0,
              2, 1, 0,
              1, 2, 3;
    BoundingBox b = comp.computeBoundsFromPoints(points);
    REQUIRE(b.isValid());
    REQUIRE(b.min_x == 0.0);
    REQUIRE(b.max_x == 2.0);
    REQUIRE(b.min_y == 0.0);
    REQUIRE(b.max_y == 2.0);
    REQUIRE(b.min_z == 0.0);
    REQUIRE(b.max_z == 3.0);
}

TEST_CASE("UnifiedBoundsComputer: computeUnionBounds two sets", "[synchronization][point_bounds]") {
    UnifiedBoundsComputer comp;
    Eigen::MatrixXd p1(1, 3);
    p1 << 0, 0, 0;
    Eigen::MatrixXd p2(1, 3);
    p2 << 2, 2, 2;
    std::vector<Eigen::MatrixXd> sets = { p1, p2 };
    BoundingBox b = comp.computeUnionBounds(sets);
    REQUIRE(b.isValid());
    REQUIRE(b.min_x == 0.0);
    REQUIRE(b.max_x == 2.0);
    REQUIRE(b.min_y == 0.0);
    REQUIRE(b.max_y == 2.0);
    REQUIRE(b.min_z == 0.0);
    REQUIRE(b.max_z == 2.0);
}

TEST_CASE("UnifiedBoundsComputer: addPadding", "[synchronization][point_bounds]") {
    UnifiedBoundsComputer comp;
    BoundingBox b(1, 1, 1, 2, 2, 2);
    BoundingBox padded = comp.addPadding(b, 0.5);
    REQUIRE(padded.min_x == 0.5);
    REQUIRE(padded.min_y == 0.5);
    REQUIRE(padded.min_z == 0.5);
    REQUIRE(padded.max_x == 2.5);
    REQUIRE(padded.max_y == 2.5);
    REQUIRE(padded.max_z == 2.5);
}

TEST_CASE("UnifiedBoundsComputer: addPadding invalid returns same", "[synchronization][point_bounds]") {
    UnifiedBoundsComputer comp;
    BoundingBox b;
    BoundingBox padded = comp.addPadding(b, 1.0);
    REQUIRE_FALSE(padded.isValid());
}

TEST_CASE("UnifiedBoundsComputer: addPadding negative throws", "[synchronization][point_bounds]") {
    UnifiedBoundsComputer comp;
    BoundingBox b(0, 0, 0, 1, 1, 1);
    REQUIRE_THROWS_AS(comp.addPadding(b, -0.1), std::invalid_argument);
}

TEST_CASE("UnifiedBoundsComputer: addPercentagePadding", "[synchronization][point_bounds]") {
    UnifiedBoundsComputer comp;
    BoundingBox b(0, 0, 0, 10, 10, 10);
    BoundingBox padded = comp.addPercentagePadding(b, 10.0);
    REQUIRE(padded.min_x == -1.0);
    REQUIRE(padded.max_x == 11.0);
    REQUIRE(padded.min_y == -1.0);
    REQUIRE(padded.max_y == 11.0);
    REQUIRE(padded.min_z == -1.0);
    REQUIRE(padded.max_z == 11.0);
}

TEST_CASE("UnifiedBoundsComputer: addPercentagePadding negative throws", "[synchronization][point_bounds]") {
    UnifiedBoundsComputer comp;
    BoundingBox b(0, 0, 0, 1, 1, 1);
    REQUIRE_THROWS_AS(comp.addPercentagePadding(b, -5.0), std::invalid_argument);
}

TEST_CASE("UnifiedBoundsComputer: computeIncremental empty returns invalid", "[synchronization][point_bounds]") {
    UnifiedBoundsComputer comp;
    std::vector<Eigen::MatrixXd> point_sets;
    std::vector<CoordinateSystem> source_systems;
    CoordinateSystem target;
    BoundingBox b = comp.computeIncremental(point_sets, source_systems, target);
    REQUIRE_FALSE(b.isValid());
}

TEST_CASE("UnifiedBoundsComputer: computeIncremental size mismatch throws", "[synchronization][point_bounds]") {
    UnifiedBoundsComputer comp;
    Eigen::MatrixXd p(1, 3);
    p << 0, 0, 0;
    std::vector<Eigen::MatrixXd> point_sets = { p };
    std::vector<CoordinateSystem> source_systems;  // empty
    CoordinateSystem target;
    REQUIRE_THROWS_AS(comp.computeIncremental(point_sets, source_systems, target), std::invalid_argument);
}

TEST_CASE("UnifiedBoundsComputer: computeIncremental one set", "[synchronization][point_bounds]") {
    UnifiedBoundsComputer comp;
    Eigen::MatrixXd p(2, 3);
    p << 1, 2, 3, 4, 5, 6;
    std::vector<Eigen::MatrixXd> point_sets = { p };
    CoordinateSystem identity;
    std::vector<CoordinateSystem> source_systems = { identity };
    CoordinateSystem target;
    BoundingBox b = comp.computeIncremental(point_sets, source_systems, target);
    REQUIRE(b.isValid());
    REQUIRE(b.min_x == 1.0);
    REQUIRE(b.max_x == 4.0);
    REQUIRE(b.min_y == 2.0);
    REQUIRE(b.max_y == 5.0);
    REQUIRE(b.min_z == 3.0);
    REQUIRE(b.max_z == 6.0);
}

#endif // EIGEN_AVAILABLE
