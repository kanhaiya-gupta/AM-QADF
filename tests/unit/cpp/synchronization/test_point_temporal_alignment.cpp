/**
 * C++ unit tests for PointTemporalAlignment (point_temporal_alignment.cpp).
 * Built when EIGEN_AVAILABLE. Tests alignSourcesByLayer, groupIndicesByLayer,
 * alignSourcesByLayerFromLayerArraysOnly.
 */

#ifdef EIGEN_AVAILABLE

#include <catch2/catch_test_macros.hpp>
#include "am_qadf_native/synchronization/point_temporal_alignment.hpp"
#include <Eigen/Dense>
#include <vector>

using namespace am_qadf_native::synchronization;

TEST_CASE("PointTemporalAlignment: alignSourcesByLayer size mismatch throws", "[synchronization][point_temporal_alignment]") {
    PointTemporalAlignment aligner;
    Eigen::MatrixXd p1(2, 3);
    p1 << 0, 0, 0, 1, 1, 1;
    std::vector<Eigen::MatrixXd> points = { p1 };
    std::vector<std::vector<int>> layers = { { 0, 1 }, { 0 } };  // second has size 1
    REQUIRE_THROWS_AS(aligner.alignSourcesByLayer(points, layers), std::invalid_argument);
}

TEST_CASE("PointTemporalAlignment: alignSourcesByLayer one source two layers", "[synchronization][point_temporal_alignment]") {
    PointTemporalAlignment aligner;
    Eigen::MatrixXd p(3, 3);
    p << 0, 0, 0, 1, 1, 1, 2, 2, 2;
    std::vector<Eigen::MatrixXd> points = { p };
    std::vector<std::vector<int>> layers = { { 0, 0, 1 } };
    LayerAlignmentResult r = aligner.alignSourcesByLayer(points, layers);
    REQUIRE(r.unique_layers.size() == 2);
    REQUIRE(r.unique_layers[0] == 0);
    REQUIRE(r.unique_layers[1] == 1);
    REQUIRE(r.indices_per_layer_per_source.size() == 2);
    REQUIRE(r.indices_per_layer_per_source[0][0].size() == 2);
    REQUIRE(r.indices_per_layer_per_source[1][0].size() == 1);
}

TEST_CASE("PointTemporalAlignment: alignSourcesByLayer two sources", "[synchronization][point_temporal_alignment]") {
    PointTemporalAlignment aligner;
    Eigen::MatrixXd p1(2, 3);
    p1 << 0, 0, 0, 1, 1, 1;
    Eigen::MatrixXd p2(2, 3);
    p2 << 2, 2, 2, 3, 3, 3;
    std::vector<Eigen::MatrixXd> points = { p1, p2 };
    std::vector<std::vector<int>> layers = { { 0, 1 }, { 0, 1 } };
    LayerAlignmentResult r = aligner.alignSourcesByLayer(points, layers);
    REQUIRE(r.unique_layers.size() == 2);
    REQUIRE(r.indices_per_layer_per_source.size() == 2);
    REQUIRE(r.indices_per_layer_per_source[0].size() == 2);
    REQUIRE(r.indices_per_layer_per_source[0][0].size() == 1);
    REQUIRE(r.indices_per_layer_per_source[0][1].size() == 1);
}

TEST_CASE("PointTemporalAlignment: groupIndicesByLayer", "[synchronization][point_temporal_alignment]") {
    PointTemporalAlignment aligner;
    std::vector<int> layer_indices = { 0, 1, 0, 1 };
    std::vector<int> unique_layers;
    std::vector<std::vector<int>> indices_per_layer;
    aligner.groupIndicesByLayer(layer_indices, unique_layers, indices_per_layer);
    REQUIRE(unique_layers.size() == 2);
    REQUIRE(indices_per_layer.size() == 2);
    REQUIRE(indices_per_layer[0].size() == 2);
    REQUIRE(indices_per_layer[1].size() == 2);
}

TEST_CASE("PointTemporalAlignment: alignSourcesByLayerFromLayerArraysOnly size mismatch throws", "[synchronization][point_temporal_alignment]") {
    PointTemporalAlignment aligner;
    int buf[] = { 0, 1 };
    std::vector<const int*> ptrs = { buf };
    std::vector<std::size_t> sizes = { 2 };
    std::vector<std::size_t> wrong_sizes = { 2, 3 };
    REQUIRE_THROWS_AS(aligner.alignSourcesByLayerFromLayerArraysOnly(ptrs, wrong_sizes), std::invalid_argument);
}

TEST_CASE("PointTemporalAlignment: alignSourcesByLayerFromLayerArraysOnly one source", "[synchronization][point_temporal_alignment]") {
    PointTemporalAlignment aligner;
    int buf[] = { 0, 1, 0 };
    std::vector<const int*> ptrs = { buf };
    std::vector<std::size_t> sizes = { 3 };
    LayerAlignmentResult r = aligner.alignSourcesByLayerFromLayerArraysOnly(ptrs, sizes);
    REQUIRE(r.unique_layers.size() == 2);
    REQUIRE(r.indices_per_layer_per_source[0][0].size() == 2);
    REQUIRE(r.indices_per_layer_per_source[1][0].size() == 1);
}

#endif // EIGEN_AVAILABLE
