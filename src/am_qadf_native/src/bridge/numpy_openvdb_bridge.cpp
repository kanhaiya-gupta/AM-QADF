#include "am_qadf_native/bridge/numpy_openvdb_bridge.hpp"
#include <openvdb/openvdb.h>
#include <openvdb/math/Transform.h>
#include <openvdb/math/Coord.h>
#include <pybind11/numpy.h>
#include <stdexcept>

namespace am_qadf_native {
namespace bridge {

FloatGridPtr numpyToOpenVDB(
    pybind11::array_t<float> array,
    float voxel_size
) {
    openvdb::initialize();

    if (array.ndim() != 3) {
        throw std::invalid_argument("Array must be 3D");
    }

    auto shape = array.shape();
    int depth = static_cast<int>(shape[0]);
    int height = static_cast<int>(shape[1]);
    int width = static_cast<int>(shape[2]);

    auto grid = openvdb::FloatGrid::create();
    grid->setTransform(
        openvdb::math::Transform::createLinearTransform(voxel_size)
    );

    auto array_ptr = array.unchecked<3>();
    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float value = array_ptr(z, y, x);
                if (value != 0.0f) {
                    grid->tree().setValue(openvdb::Coord(x, y, z), value);
                }
            }
        }
    }

    return grid;
}

pybind11::array_t<float> openVDBToNumPy(FloatGridPtr grid) {
    if (!grid) {
        return pybind11::array_t<float>({0, 0, 0});
    }

    openvdb::CoordBBox bbox;
    if (!grid->tree().evalActiveVoxelBoundingBox(bbox)) {
        return pybind11::array_t<float>({0, 0, 0});
    }

    int width = bbox.max().x() - bbox.min().x() + 1;
    int height = bbox.max().y() - bbox.min().y() + 1;
    int depth = bbox.max().z() - bbox.min().z() + 1;

    auto result = pybind11::array_t<float>({depth, height, width});
    auto result_ptr = result.mutable_unchecked<3>();

    for (auto iter = grid->beginValueOn(); iter; ++iter) {
        openvdb::Coord coord = iter.getCoord();
        int x = coord.x() - bbox.min().x();
        int y = coord.y() - bbox.min().y();
        int z = coord.z() - bbox.min().z();

        if (x >= 0 && x < width && y >= 0 && y < height && z >= 0 && z < depth) {
            result_ptr(z, y, x) = *iter;
        }
    }

    return result;
}

} // namespace bridge
} // namespace am_qadf_native
