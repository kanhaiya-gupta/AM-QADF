#ifndef AM_QADF_NATIVE_BRIDGE_NUMPY_OPENVDB_BRIDGE_HPP
#define AM_QADF_NATIVE_BRIDGE_NUMPY_OPENVDB_BRIDGE_HPP

#include <openvdb/openvdb.h>
#include <openvdb/Grid.h>
#include <pybind11/numpy.h>
#include <memory>

namespace am_qadf_native {
namespace bridge {

// Python bridge: NumPy array <-> OpenVDB grid (for small/convenience use; bulk data should use file or raw buffers).
// Lives outside voxelization so core voxelization stays 100% C++.

using FloatGridPtr = openvdb::FloatGrid::Ptr;

FloatGridPtr numpyToOpenVDB(
    pybind11::array_t<float> array,
    float voxel_size
);

pybind11::array_t<float> openVDBToNumPy(FloatGridPtr grid);

} // namespace bridge
} // namespace am_qadf_native

#endif
