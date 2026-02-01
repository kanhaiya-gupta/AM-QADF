/**
 * C++ unit tests for QueryResult (query_result.hpp).
 * Header-only struct: points, vectors, format, empty(), num_points(), num_vectors(),
 * has_multiple_signals(), has_contours().
 *
 * Aligned with docs/Tests/Test_New_Plans.md (tests/unit/cpp/query/).
 */

#include <catch2/catch_test_macros.hpp>
#include "am_qadf_native/query/query_result.hpp"
#include <array>
#include <vector>

using namespace am_qadf_native::query;

TEST_CASE("QueryResult: default empty", "[query][query_result]") {
    QueryResult r;
    REQUIRE(r.points.empty());
    REQUIRE(r.vectors.empty());
    REQUIRE(r.empty());
    REQUIRE(r.num_points() == 0);
    REQUIRE(r.num_vectors() == 0);
    REQUIRE_FALSE(r.has_multiple_signals());
    REQUIRE_FALSE(r.has_contours());
}

TEST_CASE("QueryResult: point-based format empty", "[query][query_result]") {
    QueryResult r;
    r.format = "point-based";
    REQUIRE(r.empty());
    r.points.push_back({1.0f, 2.0f, 3.0f});
    REQUIRE_FALSE(r.empty());
    REQUIRE(r.num_points() == 1);
    REQUIRE(r.points[0][0] == 1.0f);
    REQUIRE(r.points[0][1] == 2.0f);
    REQUIRE(r.points[0][2] == 3.0f);
}

TEST_CASE("QueryResult: vector-based format empty", "[query][query_result]") {
    QueryResult r;
    r.format = "vector-based";
    REQUIRE(r.empty());
    QueryResult::Vector v{};
    v.x1 = 0; v.y1 = 0; v.x2 = 1; v.y2 = 1; v.z = 0; v.timestamp = 0; v.dataindex = 0;
    r.vectors.push_back(v);
    REQUIRE_FALSE(r.empty());
    REQUIRE(r.num_vectors() == 1);
    REQUIRE(r.vectors[0].x2 == 1.0f);
}

TEST_CASE("QueryResult: has_multiple_signals", "[query][query_result]") {
    QueryResult r;
    REQUIRE_FALSE(r.has_multiple_signals());
    r.laser_temporal_data.resize(1);
    REQUIRE(r.has_multiple_signals());
}

TEST_CASE("QueryResult: has_contours", "[query][query_result]") {
    QueryResult r;
    REQUIRE_FALSE(r.has_contours());
    r.contours.resize(1);
    REQUIRE(r.has_contours());
}

TEST_CASE("QueryResult: num_points and num_vectors", "[query][query_result]") {
    QueryResult r;
    r.points.resize(3);
    r.vectors.resize(2);
    REQUIRE(r.num_points() == 3);
    REQUIRE(r.num_vectors() == 2);
}

TEST_CASE("QueryResult: metadata fields", "[query][query_result]") {
    QueryResult r;
    r.model_id = "test_model";
    r.signal_type = "laser_monitoring_data";
    r.format = "point-based";
    REQUIRE(r.model_id == "test_model");
    REQUIRE(r.signal_type == "laser_monitoring_data");
    REQUIRE(r.format == "point-based");
}
