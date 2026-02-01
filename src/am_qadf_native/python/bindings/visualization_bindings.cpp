#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "am_qadf_native/visualization/hatching_visualization_data.hpp"
#include "am_qadf_native/visualization/point_cloud_visualization_data.hpp"
#include <cstring>

namespace py = pybind11;
using namespace am_qadf_native::visualization;

namespace {

static py::tuple wrap_point_cloud_result(const PointCloudVisualizationResult& result) {
    py::ssize_t n_pos = static_cast<py::ssize_t>(result.positions.size());
    py::ssize_t n_scal = static_cast<py::ssize_t>(result.scalars.size());
    py::ssize_t n_rgb = static_cast<py::ssize_t>(result.vertex_colors_rgb.size());

    py::array_t<float> positions_array(n_pos);
    if (n_pos > 0) {
        std::memcpy(positions_array.mutable_data(), result.positions.data(),
                    static_cast<size_t>(n_pos) * sizeof(float));
    }
    py::array_t<float> scalars_array(n_scal);
    if (n_scal > 0) {
        std::memcpy(scalars_array.mutable_data(), result.scalars.data(),
                    static_cast<size_t>(n_scal) * sizeof(float));
    }
    py::array_t<float> vertex_colors_rgb_array(n_rgb);
    if (n_rgb > 0) {
        std::memcpy(vertex_colors_rgb_array.mutable_data(), result.vertex_colors_rgb.data(),
                    static_cast<size_t>(n_rgb) * sizeof(float));
    }
    return py::make_tuple(
        positions_array,
        scalars_array,
        result.active_scalar_name,
        vertex_colors_rgb_array,
        result.scalar_bar_min,
        result.scalar_bar_max
    );
}

py::tuple wrap_get_hatching_visualization_data(
    const std::string& model_id,
    int layer_start,
    int layer_end,
    const std::string& scalar_name,
    const std::string& uri,
    const std::string& db_name
) {
    HatchingVisualizationResult result = get_hatching_visualization_data(
        model_id, layer_start, layer_end, scalar_name, uri, db_name);

    py::ssize_t n_pos = static_cast<py::ssize_t>(result.positions.size());
    py::ssize_t n_scal = static_cast<py::ssize_t>(result.scalars.size());
    py::ssize_t n_types = static_cast<py::ssize_t>(result.segment_types.size());
    py::ssize_t n_rgb = static_cast<py::ssize_t>(result.vertex_colors_rgb.size());

    py::array_t<float> positions_array(n_pos);
    if (n_pos > 0) {
        std::memcpy(positions_array.mutable_data(), result.positions.data(),
                    static_cast<size_t>(n_pos) * sizeof(float));
    }

    py::array_t<float> scalars_array(n_scal);
    if (n_scal > 0) {
        std::memcpy(scalars_array.mutable_data(), result.scalars.data(),
                    static_cast<size_t>(n_scal) * sizeof(float));
    }

    py::array_t<int> segment_types_array(n_types);
    if (n_types > 0) {
        std::memcpy(segment_types_array.mutable_data(), result.segment_types.data(),
                    static_cast<size_t>(n_types) * sizeof(int));
    }

    py::array_t<float> vertex_colors_rgb_array(n_rgb);
    if (n_rgb > 0) {
        std::memcpy(vertex_colors_rgb_array.mutable_data(), result.vertex_colors_rgb.data(),
                    static_cast<size_t>(n_rgb) * sizeof(float));
    }

    return py::make_tuple(
        positions_array,
        scalars_array,
        result.active_scalar_name,
        segment_types_array,
        vertex_colors_rgb_array,
        result.scalar_bar_min,
        result.scalar_bar_max
    );
}

}  // namespace

void bind_visualization(py::module& m) {
    py::module_ m_vis = m.def_submodule("visualization",
        "Visualization data (C++ only): ready-to-render positions and scalars from query.");

    m_vis.def("get_hatching_visualization_data", &wrap_get_hatching_visualization_data,
              py::arg("model_id"),
              py::arg("layer_start") = -1,
              py::arg("layer_end") = -1,
              py::arg("scalar_name") = "laser_power",
              py::arg("uri") = "mongodb://localhost:27017",
              py::arg("db_name") = "am_qadf",
              R"doc(
Get ready-to-visualize hatching buffers (C++ only; no Python fallback).

Calls the query client in C++, sorts vectors by type and spatial order,
then returns flat positions and scalars. Python should only build PyVista
PolyData from these arrays.

Args:
    model_id: Model UUID.
    layer_start: First layer index (-1 = no filter).
    layer_end: Last layer index (-1 = no filter).
    scalar_name: "laser_power", "scan_speed", or "length".
    uri: MongoDB connection URI.
    db_name: MongoDB database name.

Returns:
    tuple: (positions, scalars, active_scalar_name, segment_types, vertex_colors_rgb, scalar_bar_min, scalar_bar_max)
    - positions: float array shape (6 * n_segments,).
    - scalars: float array shape (n_segments,).
    - active_scalar_name: "laser_power", "scan_speed", or "length".
    - segment_types: int array shape (n_segments,) 0=hatch, 1=contour.
    - vertex_colors_rgb: float array shape (6 * n_segments,) per-vertex RGB (no Python color calc).
    - scalar_bar_min/max: heatmap range when not path_only; NaN otherwise.
)doc");

    m_vis.def("get_laser_monitoring_visualization_data",
              [](const std::string& model_id, int layer_start, int layer_end,
                 const std::string& scalar_name, const std::string& uri, const std::string& db_name) {
                  return wrap_point_cloud_result(get_laser_monitoring_visualization_data(
                      model_id, layer_start, layer_end, scalar_name, uri, db_name));
              },
              py::arg("model_id"),
              py::arg("layer_start") = -1,
              py::arg("layer_end") = -1,
              py::arg("scalar_name") = "actual_power",
              py::arg("uri") = "mongodb://localhost:27017",
              py::arg("db_name") = "am_qadf",
              R"doc(
Get ready-to-visualize point cloud buffers for laser monitoring (C++ only).

Returns positions (3*n), scalars (n), active_scalar_name, vertex_colors_rgb (3*n), scalar_bar_min/max.
)doc");

    m_vis.def("get_ispm_visualization_data",
              [](const std::string& model_id, int layer_start, int layer_end,
                 const std::string& source_type, const std::string& scalar_name,
                 const std::string& uri, const std::string& db_name) {
                  return wrap_point_cloud_result(get_ispm_visualization_data(
                      model_id, layer_start, layer_end, source_type, scalar_name, uri, db_name));
              },
              py::arg("model_id"),
              py::arg("layer_start") = -1,
              py::arg("layer_end") = -1,
              py::arg("source_type"),
              py::arg("scalar_name") = "",
              py::arg("uri") = "mongodb://localhost:27017",
              py::arg("db_name") = "am_qadf",
              R"doc(
Get ready-to-visualize point cloud buffers for ISPM (C++ only).

source_type: "ispm_thermal", "ispm_optical", "ispm_acoustic", "ispm_strain", "ispm_plume".
Returns positions (3*n), scalars (n), active_scalar_name, vertex_colors_rgb (3*n), scalar_bar_min/max.
)doc");
}
