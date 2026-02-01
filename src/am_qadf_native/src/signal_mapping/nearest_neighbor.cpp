#include "am_qadf_native/signal_mapping/nearest_neighbor.hpp"
#include "am_qadf_native/signal_mapping/interpolation_base.hpp"
#include <openvdb/openvdb.h>
#include <openvdb/math/Coord.h>
#include <openvdb/math/Vec3.h>
#include <cmath>
#include <vector>
#include <stdexcept>

namespace am_qadf_native {
namespace signal_mapping {

void NearestNeighborMapper::map(
    FloatGridPtr grid,
    const std::vector<Point>& points,
    const std::vector<float>& values
) {
    if (points.size() != values.size()) {
        throw std::invalid_argument("Points and values size mismatch");
    }
    
    for (size_t i = 0; i < points.size(); ++i) {
        // Node-centered: floor(index) so world (0.5,0.5,0.5) -> voxel (0,0,0)
        openvdb::Coord nearest = grid->transform().worldToIndexNodeCentered(
            openvdb::Vec3R(points[i].x, points[i].y, points[i].z)
        );
        grid->tree().setValue(nearest, values[i]);
    }
}

} // namespace signal_mapping
} // namespace am_qadf_native
