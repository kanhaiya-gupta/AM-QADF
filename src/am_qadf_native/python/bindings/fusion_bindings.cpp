#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "am_qadf_native/fusion/grid_fusion.hpp"
#include "am_qadf_native/fusion/fusion_strategies.hpp"
#include "am_qadf_native/fusion/fusion_quality.hpp"

namespace py = pybind11;
using namespace am_qadf_native::fusion;

void bind_fusion(py::module& m) {
    // Create fusion submodule
    auto fusion_m = m.def_submodule("fusion", "Grid fusion operations for OpenVDB");
    
    // GridFusion
    py::class_<GridFusion>(fusion_m, "GridFusion")
        .def(py::init<>())
        .def("fuse", &GridFusion::fuse, py::arg("grids"), py::arg("strategy") = "weighted_average")
        .def("fuse_weighted", &GridFusion::fuseWeighted);
    
    // Fusion strategies
    py::class_<FusionStrategy>(fusion_m, "FusionStrategy");
    
    py::class_<WeightedAverageStrategy, FusionStrategy>(fusion_m, "WeightedAverageStrategy")
        .def(py::init<const std::vector<float>&>())
        .def("fuse_values", &WeightedAverageStrategy::fuseValues);
    
    py::class_<MaxStrategy, FusionStrategy>(fusion_m, "MaxStrategy")
        .def(py::init<>())
        .def("fuse_values", &MaxStrategy::fuseValues);
    
    py::class_<MinStrategy, FusionStrategy>(fusion_m, "MinStrategy")
        .def(py::init<>())
        .def("fuse_values", &MinStrategy::fuseValues);
    
    py::class_<MedianStrategy, FusionStrategy>(fusion_m, "MedianStrategy")
        .def(py::init<>())
        .def("fuse_values", &MedianStrategy::fuseValues);

    // Fusion quality (C++ core for scale)
    py::class_<FusionQualityResult>(fusion_m, "FusionQualityResult")
        .def(py::init<>())
        .def_readwrite("fusion_accuracy", &FusionQualityResult::fusion_accuracy)
        .def_readwrite("signal_consistency", &FusionQualityResult::signal_consistency)
        .def_readwrite("fusion_completeness", &FusionQualityResult::fusion_completeness)
        .def_readwrite("quality_score", &FusionQualityResult::quality_score)
        .def_readwrite("coverage_ratio", &FusionQualityResult::coverage_ratio)
        .def_readwrite("per_signal_accuracy", &FusionQualityResult::per_signal_accuracy)
        .def_readwrite("residual_mean", &FusionQualityResult::residual_mean)
        .def_readwrite("residual_max", &FusionQualityResult::residual_max)
        .def_readwrite("residual_std", &FusionQualityResult::residual_std)
        .def_readwrite("has_residual_summary", &FusionQualityResult::has_residual_summary);

    py::class_<FusionQualityAssessor>(fusion_m, "FusionQualityAssessor")
        .def(py::init<>())
        .def("assess", &FusionQualityAssessor::assess,
             py::arg("fused_grid"),
             py::arg("source_grids"),
             py::arg("weights") = std::map<std::string, float>());
}
