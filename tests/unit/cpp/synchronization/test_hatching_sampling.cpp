/**
 * Placeholder for hatching_sampling.hpp. No .cpp implementation in repo yet;
 * API declares sample_hatching_points_by_z (double and float overloads).
 * Built when EIGEN_AVAILABLE. Add real tests once implementation exists.
 */

#ifdef EIGEN_AVAILABLE

#include <catch2/catch_test_macros.hpp>
// When implementation is added: #include "am_qadf_native/synchronization/hatching_sampling.hpp"

TEST_CASE("HatchingSampling: placeholder until sample_hatching_points_by_z is implemented", "[synchronization][hatching_sampling]") {
    REQUIRE(true);
}

#endif // EIGEN_AVAILABLE
