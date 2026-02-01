#ifdef EIGEN_AVAILABLE

#include "am_qadf_native/synchronization/point_transformation_sampling.hpp"
#include <stdexcept>
#include <vector>

namespace am_qadf_native {
namespace synchronization {

std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>>
TransformationSampling::enumerateTripletSamplesFrom8(
    const Eigen::MatrixXd& source_8,
    const Eigen::MatrixXd& target_8
) {
    if (source_8.rows() != 8 || target_8.rows() != 8 ||
        source_8.cols() != 3 || target_8.cols() != 3) {
        throw std::invalid_argument(
            "enumerateTripletSamplesFrom8: source and target must be 8x3 (bbox corners)"
        );
    }
    std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> out;
    out.reserve(56);  // C(8,3) = 56
    for (int i = 0; i < 6; ++i) {
        for (int j = i + 1; j < 7; ++j) {
            for (int k = j + 1; k < 8; ++k) {
                Eigen::MatrixXd src_3(3, 3), tgt_3(3, 3);
                src_3.row(0) = source_8.row(i);
                src_3.row(1) = source_8.row(j);
                src_3.row(2) = source_8.row(k);
                tgt_3.row(0) = target_8.row(i);
                tgt_3.row(1) = target_8.row(j);
                tgt_3.row(2) = target_8.row(k);
                out.emplace_back(src_3, tgt_3);
            }
        }
    }
    return out;
}

}  // namespace synchronization
}  // namespace am_qadf_native

#endif  // EIGEN_AVAILABLE
