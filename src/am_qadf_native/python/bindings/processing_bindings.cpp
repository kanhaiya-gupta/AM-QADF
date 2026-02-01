#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include "am_qadf_native/processing/signal_processing.hpp"
#include "am_qadf_native/processing/signal_generation.hpp"

namespace py = pybind11;
using namespace am_qadf_native::processing;

void bind_processing(py::module& m) {
    // Create processing submodule so "from am_qadf_native.processing import SignalGeneration" works
    auto processing_m = m.def_submodule("processing", "Signal processing and generation (C++).");

    // SignalProcessing
    py::class_<SignalProcessing>(processing_m, "SignalProcessing")
        .def(py::init<>())
        .def("normalize", &SignalProcessing::normalize,
             py::arg("values"), py::arg("min_value") = 0.0f, py::arg("max_value") = 1.0f)
        .def("moving_average", &SignalProcessing::movingAverage,
             py::arg("values"), py::arg("window_size") = 5)
        .def("derivative", &SignalProcessing::derivative)
        .def("integral", &SignalProcessing::integral)
        .def("fft", &SignalProcessing::fft)
        .def("ifft", &SignalProcessing::ifft)
        .def("frequency_filter", &SignalProcessing::frequencyFilter);

    // SignalGeneration
    py::class_<SignalGeneration>(processing_m, "SignalGeneration")
        .def(py::init<>())
        .def("generate_synthetic", &SignalGeneration::generateSynthetic,
             py::arg("points"), py::arg("signal_type"),
             py::arg("amplitude") = 1.0f, py::arg("frequency") = 1.0f)
        .def("generate_gaussian", &SignalGeneration::generateGaussian)
        .def("generate_sine_wave", &SignalGeneration::generateSineWave)
        .def("generate_random", &SignalGeneration::generateRandom)
        .def("generate_from_expression", &SignalGeneration::generateFromExpression);
}
