#ifndef AM_QADF_NATIVE_SYNCHRONIZATION_POINT_TRANSFORMATION_SAMPLING_HPP
#define AM_QADF_NATIVE_SYNCHRONIZATION_POINT_TRANSFORMATION_SAMPLING_HPP

#ifdef EIGEN_AVAILABLE
#include <Eigen/Dense>
#include <utility>
#include <vector>

namespace am_qadf_native {
namespace synchronization {

// Sampling for transformation: bbox corners only (8 points → 56 triplets for fit-on-3, validate-on-8).
struct TransformationSampling {

    // Enumerate all C(8,3)=56 ways to choose 3 corresponding pairs from 8 bbox corners.
    // Each pair is (source_3x3, target_3x3). Used by TransformationComputer::computeTransformationFromBboxCorners.
    // source_8 and target_8 must be 8x3 (e.g. BoundingBox::corners()).
    static std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> enumerateTripletSamplesFrom8(
        const Eigen::MatrixXd& source_8,
        const Eigen::MatrixXd& target_8
    );
};

}  // namespace synchronization
}  // namespace am_qadf_native

#endif  // EIGEN_AVAILABLE
#endif  // AM_QADF_NATIVE_SYNCHRONIZATION_POINT_TRANSFORMATION_SAMPLING_HPP
