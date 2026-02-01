#include "am_qadf_native/signal_mapping/linear_interpolation.hpp"
#include "am_qadf_native/signal_mapping/interpolation_base.hpp"
#include <openvdb/openvdb.h>
#include <openvdb/math/Coord.h>
#include <openvdb/math/Vec3.h>
#include <cmath>
#include <vector>
#include <stdexcept>

namespace am_qadf_native {
namespace signal_mapping {

void LinearMapper::map(
    FloatGridPtr grid,
    const std::vector<Point>& points,
    const std::vector<float>& values
) {
    if (points.size() != values.size()) {
        throw std::invalid_argument("Points and values size mismatch");
    }
    
    for (size_t i = 0; i < points.size(); ++i) {
        openvdb::Vec3R world_coord(points[i].x, points[i].y, points[i].z);
        openvdb::Vec3R coord = grid->transform().worldToIndex(world_coord);
        
        // Distribute value to 8 neighboring voxels using trilinear weights
        // Convert Coord to Vec3R for trilinear interpolation
        openvdb::Vec3R coord_vec(
            static_cast<double>(coord.x()),
            static_cast<double>(coord.y()),
            static_cast<double>(coord.z())
        );
        distributeTrilinear(grid, coord_vec, values[i]);
    }
}

void LinearMapper::distributeTrilinear(
    FloatGridPtr grid,
    const openvdb::Vec3R& coord,
    float value
) {
    // Get 8 corner voxels
    openvdb::Coord c000(
        static_cast<int>(std::floor(coord.x())),
        static_cast<int>(std::floor(coord.y())),
        static_cast<int>(std::floor(coord.z()))
    );
    
    openvdb::Coord c001(c000.x(), c000.y(), c000.z() + 1);
    openvdb::Coord c010(c000.x(), c000.y() + 1, c000.z());
    openvdb::Coord c011(c000.x(), c000.y() + 1, c000.z() + 1);
    openvdb::Coord c100(c000.x() + 1, c000.y(), c000.z());
    openvdb::Coord c101(c000.x() + 1, c000.y(), c000.z() + 1);
    openvdb::Coord c110(c000.x() + 1, c000.y() + 1, c000.z());
    openvdb::Coord c111(c000.x() + 1, c000.y() + 1, c000.z() + 1);
    
    // Calculate trilinear weights (inverse of BoxSampler)
    float wx = coord.x() - c000.x();
    float wy = coord.y() - c000.y();
    float wz = coord.z() - c000.z();
    
    // Distribute value to 8 neighbors with weights
    auto& tree = grid->tree();
    tree.setValue(c000, tree.getValue(c000) + value * (1-wx) * (1-wy) * (1-wz));
    tree.setValue(c001, tree.getValue(c001) + value * (1-wx) * (1-wy) * wz);
    tree.setValue(c010, tree.getValue(c010) + value * (1-wx) * wy * (1-wz));
    tree.setValue(c011, tree.getValue(c011) + value * (1-wx) * wy * wz);
    tree.setValue(c100, tree.getValue(c100) + value * wx * (1-wy) * (1-wz));
    tree.setValue(c101, tree.getValue(c101) + value * wx * (1-wy) * wz);
    tree.setValue(c110, tree.getValue(c110) + value * wx * wy * (1-wz));
    tree.setValue(c111, tree.getValue(c111) + value * wx * wy * wz);
}

} // namespace signal_mapping
} // namespace am_qadf_native
