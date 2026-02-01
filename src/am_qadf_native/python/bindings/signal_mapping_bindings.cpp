#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "am_qadf_native/signal_mapping/nearest_neighbor.hpp"
#include "am_qadf_native/signal_mapping/linear_interpolation.hpp"
#include "am_qadf_native/signal_mapping/idw_interpolation.hpp"
#include "am_qadf_native/signal_mapping/kde_interpolation.hpp"
#include "am_qadf_native/signal_mapping/rbf_interpolation.hpp"
#include "am_qadf_native/signal_mapping/interpolation_base.hpp"

namespace py = pybind11;
using namespace am_qadf_native::signal_mapping;

void bind_signal_mapping(py::module& m) {
    // Create submodule so Python can do: from am_qadf_native.signal_mapping import IDWMapper, Point, ...
    py::module signal_mapping = m.def_submodule("signal_mapping", "Signal mapping C++ implementations (IDW, KDE, Linear, NearestNeighbor, RBF)");

    // Point struct
    py::class_<Point>(signal_mapping, "Point")
        .def(py::init<float, float, float>(),
             py::arg("x") = 0.0f, py::arg("y") = 0.0f, py::arg("z") = 0.0f)
        .def_readwrite("x", &Point::x)
        .def_readwrite("y", &Point::y)
        .def_readwrite("z", &Point::z);

    // Efficient numpy array to std::vector<Point> converter (C++ only, no Python loops)
    signal_mapping.def("numpy_to_points", [](py::array_t<float> points_array, py::array_t<float> bbox_min_array = py::none()) {
        // Validate input shape
        if (points_array.ndim() != 2 || points_array.shape(1) != 3) {
            throw std::invalid_argument("Points array must be shape (N, 3)");
        }
        
        // Get raw pointer to data (C++ access, no copy)
        auto points_buf = points_array.request();
        const float* data = static_cast<const float*>(points_buf.ptr);
        size_t num_points = points_array.shape(0);
        
        // Get bbox_min if provided
        const float* bbox_min = nullptr;
        if (!bbox_min_array.is_none()) {
            if (bbox_min_array.ndim() != 1 || bbox_min_array.shape(0) != 3) {
                throw std::invalid_argument("bbox_min must be shape (3,)");
            }
            auto bbox_buf = bbox_min_array.request();
            bbox_min = static_cast<const float*>(bbox_buf.ptr);
        }
        
        // Call C++ converter (all processing in C++, no Python loops)
        return numpy_to_points(data, num_points, bbox_min);
    }, py::arg("points_array"), py::arg("bbox_min") = py::none(),
       "Convert numpy array (N, 3) to std::vector<Point> efficiently in C++");

    // NearestNeighborMapper
    py::class_<NearestNeighborMapper>(signal_mapping, "NearestNeighborMapper")
        .def(py::init<>())
        .def("map", &NearestNeighborMapper::map);

    // LinearMapper
    py::class_<LinearMapper>(signal_mapping, "LinearMapper")
        .def(py::init<>())
        .def("map", &LinearMapper::map);

    // IDWMapper
    py::class_<IDWMapper>(signal_mapping, "IDWMapper")
        .def(py::init<float, int>(), py::arg("power") = 2.0f, py::arg("k_neighbors") = 10)
        .def("map", &IDWMapper::map)
        .def("set_power", &IDWMapper::setPower)
        .def("set_k_neighbors", &IDWMapper::setKNeighbors);

    // KDEMapper
    py::class_<KDEMapper>(signal_mapping, "KDEMapper")
        .def(py::init<float, const std::string&>(),
             py::arg("bandwidth") = 1.0f, py::arg("kernel_type") = "gaussian")
        .def("map", &KDEMapper::map)
        .def("set_bandwidth", &KDEMapper::setBandwidth)
        .def("set_kernel_type", &KDEMapper::setKernelType);

    // RBFMapper
    py::class_<RBFMapper>(signal_mapping, "RBFMapper")
        .def(py::init<const std::string&, float>(),
             py::arg("kernel_type") = "gaussian", py::arg("epsilon") = 1.0f)
        .def("map", &RBFMapper::map)
        .def("set_kernel_type", &RBFMapper::setKernelType)
        .def("set_epsilon", &RBFMapper::setEpsilon);
}
