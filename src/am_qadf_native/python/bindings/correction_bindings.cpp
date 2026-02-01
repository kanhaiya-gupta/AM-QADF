#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "am_qadf_native/correction/signal_noise_reduction.hpp"
#include "am_qadf_native/correction/spatial_noise_filtering.hpp"
#include "am_qadf_native/correction/geometric_correction.hpp"
#include "am_qadf_native/correction/calibration.hpp"
#include "am_qadf_native/correction/validation.hpp"

namespace py = pybind11;
using namespace am_qadf_native::correction;

void bind_correction(py::module& m) {
    // Create correction submodule so "from am_qadf_native.correction import Calibration" works
    auto correction_m = m.def_submodule("correction", "Correction operations (calibration, validation, noise reduction, etc.)");

    // CalibrationData struct (required by Python wrapper)
    py::class_<CalibrationData>(correction_m, "CalibrationData")
        .def(py::init<>())
        .def_readwrite("sensor_id", &CalibrationData::sensor_id)
        .def_readwrite("calibration_type", &CalibrationData::calibration_type)
        .def_readwrite("parameters", &CalibrationData::parameters)
        .def_readwrite("reference_points", &CalibrationData::reference_points)
        .def_readwrite("measured_points", &CalibrationData::measured_points);

    // ValidationResult struct (required by Python wrapper)
    py::class_<ValidationResult>(correction_m, "ValidationResult")
        .def(py::init<>())
        .def_readwrite("is_valid", &ValidationResult::is_valid)
        .def_readwrite("errors", &ValidationResult::errors)
        .def_readwrite("warnings", &ValidationResult::warnings)
        .def_readwrite("metrics", &ValidationResult::metrics);

    // SignalNoiseReduction
    py::class_<SignalNoiseReduction>(correction_m, "SignalNoiseReduction")
        .def(py::init<>())
        .def("reduce_noise", &SignalNoiseReduction::reduceNoise,
             py::arg("raw_data"), py::arg("method"), py::arg("sigma") = 1.0f)
        .def("apply_gaussian_filter", &SignalNoiseReduction::applyGaussianFilter)
        .def("apply_savitzky_golay", &SignalNoiseReduction::applySavitzkyGolay)
        .def("remove_outliers", &SignalNoiseReduction::removeOutliers);

    // SpatialNoiseFilter
    py::class_<SpatialNoiseFilter>(correction_m, "SpatialNoiseFilter")
        .def(py::init<>())
        .def("apply", &SpatialNoiseFilter::apply,
             py::arg("grid"), py::arg("method"), py::arg("kernel_size") = 3,
             py::arg("sigma_spatial") = 1.0f, py::arg("sigma_color") = 0.1f)
        .def("apply_median_filter", &SpatialNoiseFilter::applyMedianFilter)
        .def("apply_bilateral_filter", &SpatialNoiseFilter::applyBilateralFilter)
        .def("apply_gaussian_filter", &SpatialNoiseFilter::applyGaussianFilter);

    // GeometricCorrection
    py::class_<GeometricCorrection>(correction_m, "GeometricCorrection")
        .def(py::init<>())
        .def("correct_distortions", &GeometricCorrection::correctDistortions)
        .def("correct_lens_distortion", &GeometricCorrection::correctLensDistortion)
        .def("correct_sensor_misalignment", &GeometricCorrection::correctSensorMisalignment);

    // Calibration
    py::class_<Calibration>(correction_m, "Calibration")
        .def(py::init<>())
        .def("load_from_file", &Calibration::loadFromFile)
        .def("save_to_file", &Calibration::saveToFile)
        .def("compute_calibration", &Calibration::computeCalibration)
        .def("validate_calibration", &Calibration::validateCalibration);

    // Validation
    py::class_<Validation>(correction_m, "Validation")
        .def(py::init<>())
        .def("validate_grid", &Validation::validateGrid)
        .def("validate_signal_data", &Validation::validateSignalData)
        .def("validate_coordinates", &Validation::validateCoordinates)
        .def("check_consistency", &Validation::checkConsistency);
}
