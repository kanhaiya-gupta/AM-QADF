#include "am_qadf_native/signal_mapping/interpolation_base.hpp"
#include <openvdb/openvdb.h>
#include <openvdb/math/Coord.h>
#include <openvdb/math/Vec3.h>
#include <cmath>

namespace am_qadf_native {
namespace signal_mapping {

std::vector<Point> numpy_to_points(
    const float* data,
    size_t num_points,
    const float* bbox_min
) {
    // C++ only: convert numpy array (N, 3) to std::vector<Point> efficiently
    std::vector<Point> points;
    points.reserve(num_points);
    
    if (bbox_min) {
        // Adjust points relative to bbox_min (vectorized in C++)
        for (size_t i = 0; i < num_points; ++i) {
            size_t idx = i * 3;
            points.emplace_back(
                data[idx] - bbox_min[0],
                data[idx + 1] - bbox_min[1],
                data[idx + 2] - bbox_min[2]
            );
        }
    } else {
        // No offset
        for (size_t i = 0; i < num_points; ++i) {
            size_t idx = i * 3;
            points.emplace_back(
                data[idx],
                data[idx + 1],
                data[idx + 2]
            );
        }
    }
    
    return points;
}

openvdb::Coord InterpolationBase::worldToGridCoord(
    FloatGridPtr grid,
    float x, float y, float z
) const {
    auto coord = grid->transform().worldToIndexCellCentered(
        openvdb::Vec3R(x, y, z)
    );
    return openvdb::Coord(
        static_cast<int>(coord.x()),
        static_cast<int>(coord.y()),
        static_cast<int>(coord.z())
    );
}

} // namespace signal_mapping
} // namespace am_qadf_native
