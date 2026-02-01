#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "am_qadf_native/io/openvdb_reader.hpp"
#include "am_qadf_native/io/vdb_writer.hpp"
#include "am_qadf_native/io/paraview_exporter.hpp"
#include "am_qadf_native/io/mongodb_writer.hpp"
#include "am_qadf_native/io/mongocxx_instance.hpp"
#include "am_qadf_native/io/save_processed_points_bridge.hpp"
#include "am_qadf_native/synchronization/point_bounds.hpp"

namespace py = pybind11;
using namespace am_qadf_native::io;
using namespace am_qadf_native::synchronization;

void bind_io(py::module& m) {
    // Create io submodule
    auto io_m = m.def_submodule("io", "I/O operations for OpenVDB grids");
    
    // BoundingBox (io namespace)
    py::class_<am_qadf_native::io::BoundingBox>(io_m, "BoundingBox")
        .def(py::init<>())
        .def_readwrite("x_min", &am_qadf_native::io::BoundingBox::x_min)
        .def_readwrite("y_min", &am_qadf_native::io::BoundingBox::y_min)
        .def_readwrite("z_min", &am_qadf_native::io::BoundingBox::z_min)
        .def_readwrite("x_max", &am_qadf_native::io::BoundingBox::x_max)
        .def_readwrite("y_max", &am_qadf_native::io::BoundingBox::y_max)
        .def_readwrite("z_max", &am_qadf_native::io::BoundingBox::z_max);
    
    // OpenVDBReader
    py::class_<OpenVDBReader>(io_m, "OpenVDBReader")
        .def(py::init<>())
        .def("extract_features", &OpenVDBReader::extractFeatures)
        .def("extract_region", &OpenVDBReader::extractRegion)
        .def("extract_samples", &OpenVDBReader::extractSamples,
             py::arg("vdb_file"), py::arg("n_samples"), py::arg("strategy") = "uniform")
        .def("extract_full_grid", &OpenVDBReader::extractFullGrid)
        .def("process_chunks", &OpenVDBReader::processChunks)
        .def("load_grid_by_name", &OpenVDBReader::loadGridByName)
        .def("load_all_grids", &OpenVDBReader::loadAllGrids);
    
    // VDBWriter - bind to both main module and io submodule for compatibility
    auto vdb_writer = py::class_<VDBWriter>(io_m, "VDBWriter")
        .def(py::init<>())
        .def("write", &VDBWriter::write)
        .def("write_multiple", &VDBWriter::writeMultiple)
        .def("write_multiple_with_names", [](VDBWriter& self, 
             const std::string& filename,
             const py::dict& grids_dict) {
            // Convert dict to vectors
            std::vector<FloatGridPtr> grids;
            std::vector<std::string> grid_names;
            
            for (auto item : grids_dict) {
                std::string name = py::cast<std::string>(item.first);
                FloatGridPtr grid = py::cast<FloatGridPtr>(item.second);
                grid_names.push_back(name);
                grids.push_back(grid);
            }
            
            self.writeMultipleWithNames(grids, grid_names, filename);
        }, py::arg("filename"), py::arg("grids_dict"),
           "Write multiple grids with names from a dictionary {name: grid}")
        .def("write_multiple_with_names", &VDBWriter::writeMultipleWithNames,
             py::arg("grids"), py::arg("grid_names"), py::arg("filename"),
             "Write multiple grids with names from vectors")
        .def("write_compressed", &VDBWriter::writeCompressed)
        .def("append", &VDBWriter::append);
    
    // Also expose VDBWriter in main module for backward compatibility
    m.attr("VDBWriter") = io_m.attr("VDBWriter");
    
    // ParaViewExporter
    py::class_<ParaViewExporter>(io_m, "ParaViewExporter")
        .def(py::init<>())
        .def("export_to_paraview", &ParaViewExporter::exportToParaView)
        .def("export_multiple_to_paraview", &ParaViewExporter::exportMultipleToParaView)
        .def("export_with_metadata", &ParaViewExporter::exportWithMetadata);
    
    // ============================================
    // MongoDB C driver init (call early in main thread, e.g. before Jupyter uses writer)
    // ============================================
    io_m.def("ensure_mongocxx_initialized", []() {
        (void)get_mongocxx_instance();
    }, "Initialize the MongoDB C driver in the current thread. Call once before using MongoDBWriter (e.g. in Jupyter) to avoid kernel crashes.");

    io_m.def("save_transformed_points_to_mongodb", &save_transformed_points_to_mongodb,
             py::arg("model_id"), py::arg("source_types"),
             py::arg("transformed_points"), py::arg("signals"),
             py::arg("layer_indices_per_source"), py::arg("timestamps_per_source"),
             py::arg("transformations"), py::arg("unified_bounds"),
             py::arg("mongo_uri"), py::arg("db_name"),
             py::arg("batch_size") = 10000,
             "Save transformed points to MongoDB from dicts; iteration and padding in C++, zero-copy buffers. Use for scale (billions of points).");

    // ============================================
    // MongoDBWriter (for saving processed data)
    // ============================================
    
    py::class_<MongoDBWriter>(io_m, "MongoDBWriter")
        .def(py::init<const std::string&, const std::string&>(),
             py::arg("uri"), py::arg("db_name"),
             "Create MongoDB writer for saving processed data")
        .def("delete_processed_data", &MongoDBWriter::deleteProcessedData,
             py::arg("model_id"), py::arg("source_type"),
             py::arg("processing_run_id") = "",
             "Delete existing processed data for a model_id. "
             "If processing_run_id is provided, only deletes that run. "
             "Returns number of documents deleted.")
        .def("save_processed_points", &MongoDBWriter::saveProcessedPoints,
             py::arg("model_id"), py::arg("source_type"),
             py::arg("transformed_points"), py::arg("signal_values"),
             py::arg("layer_indices"), py::arg("timestamps"),
             py::arg("unified_bounds"), py::arg("transformation_matrix"),
             py::arg("quality_metrics"), py::arg("processing_run_id") = "",
             py::arg("delete_existing") = false, py::arg("batch_size") = 10000,
             "Save processed points to MongoDB (one document per point, batched). "
             "Collection: Processed_<source_type>_data\n\n"
             "OVERWRITE BEHAVIOR: Uses upsert to overwrite existing documents with same "
             "model_id + layer_index + spatial_coordinates. Same sample from same data source "
             "gets overwritten, no duplication.\n\n"
             "delete_existing: If True, deletes ALL existing processed data for model_id before upserting.")
        .def("save_processed_points_multiple_signals", &MongoDBWriter::saveProcessedPointsMultipleSignals,
             py::arg("model_id"), py::arg("source_type"),
             py::arg("transformed_points"), py::arg("signal_values_map"),
             py::arg("layer_indices"), py::arg("timestamps"),
             py::arg("unified_bounds"), py::arg("transformation_matrix"),
             py::arg("quality_metrics"), py::arg("processing_run_id") = "",
             py::arg("delete_existing") = false, py::arg("batch_size") = 10000,
             "Save processed points with multiple signals (one document per point, batched).\n\n"
             "OVERWRITE BEHAVIOR: Uses upsert to overwrite existing documents with same "
             "model_id + layer_index + spatial_coordinates. Same sample from same data source "
             "gets overwritten, no duplication.\n\n"
             "delete_existing: If True, deletes ALL existing processed data for model_id before upserting.")
        .def("save_transformation_metadata", &MongoDBWriter::saveTransformationMetadata,
             py::arg("model_id"), py::arg("source_type"), py::arg("target_type"),
             py::arg("transformation_matrix"), py::arg("quality_metrics"),
             "Save transformation metadata to MongoDB")
        .def("save_unified_bounds", &MongoDBWriter::saveUnifiedBounds,
             py::arg("model_id"), py::arg("unified_bounds"), py::arg("source_types"),
             "Save unified bounds metadata to MongoDB");
}
