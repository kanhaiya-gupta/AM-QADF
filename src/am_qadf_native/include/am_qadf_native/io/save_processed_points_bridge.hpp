#ifndef AM_QADF_NATIVE_IO_SAVE_PROCESSED_POINTS_BRIDGE_HPP
#define AM_QADF_NATIVE_IO_SAVE_PROCESSED_POINTS_BRIDGE_HPP

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "am_qadf_native/synchronization/point_bounds.hpp"
#include <string>

namespace am_qadf_native {
namespace io {

/** Save transformed points to MongoDB from Python dicts; all iteration and padding in C++. Zero-copy buffers. */
void save_transformed_points_to_mongodb(
    const std::string& model_id,
    const pybind11::list& source_types,
    const pybind11::dict& transformed_points,
    const pybind11::dict& signals,
    const pybind11::dict& layer_indices_per_source,
    pybind11::object timestamps_per_source,
    const pybind11::dict& transformations,
    const synchronization::BoundingBox& unified_bounds,
    const std::string& mongo_uri,
    const std::string& db_name,
    int batch_size
);

}  // namespace io
}  // namespace am_qadf_native

#endif
