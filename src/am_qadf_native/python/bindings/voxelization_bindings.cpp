#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <map>
#include <string>
#include <openvdb/openvdb.h>
#include <openvdb/Grid.h>
#include "am_qadf_native/voxelization/openvdb_grid.hpp"
#include "am_qadf_native/bridge/numpy_openvdb_bridge.hpp"
#include "am_qadf_native/voxelization/stl_voxelizer.hpp"
#include "am_qadf_native/voxelization/hatching_voxelizer.hpp"
#include "am_qadf_native/voxelization/unified_grid_factory.hpp"
#include "am_qadf_native/synchronization/point_bounds.hpp"
#include "am_qadf_native/synchronization/point_coordinate_transform.hpp"
#include <pybind11/eigen.h>

namespace py = pybind11;
using namespace am_qadf_native::voxelization;

void bind_voxelization(py::module& m) {
    // Register OpenVDB FloatGrid as a Pybind11 type so FloatGridPtr can be returned
    // FloatGridPtr is std::shared_ptr<openvdb::FloatGrid>
    // We register it as an opaque type - Pybind11 will handle std::shared_ptr automatically
    py::class_<openvdb::FloatGrid, std::shared_ptr<openvdb::FloatGrid>>(m, "OpenVDBFloatGrid")
        .def("__repr__", [](const std::shared_ptr<openvdb::FloatGrid>&) {
            return "<OpenVDB FloatGrid>";
        });
    
    // Register OpenVDB Vec3fGrid as a Pybind11 type so Vec3fGridPtr can be returned
    // Vec3fGridPtr is std::shared_ptr<openvdb::Vec3fGrid>
    // We register it as an opaque type - Pybind11 will handle std::shared_ptr automatically
    py::class_<openvdb::Vec3fGrid, std::shared_ptr<openvdb::Vec3fGrid>>(m, "OpenVDBVec3fGrid")
        .def("__repr__", [](const std::shared_ptr<openvdb::Vec3fGrid>&) {
            return "<OpenVDB Vec3fGrid>";
        });
    
    // UniformVoxelGrid
    py::class_<UniformVoxelGrid>(m, "UniformVoxelGrid")
        .def(py::init<float, float, float, float>(),
             py::arg("voxel_size"),
             py::arg("bbox_min_x") = 0.0f,
             py::arg("bbox_min_y") = 0.0f,
             py::arg("bbox_min_z") = 0.0f)
        .def("get_grid", &UniformVoxelGrid::getGrid)
        .def("set_signal_name", &UniformVoxelGrid::setSignalName)
        .def("add_point", &UniformVoxelGrid::addPoint)
        .def("add_point_at_voxel", &UniformVoxelGrid::addPointAtVoxel)
        .def("aggregate_at_voxel", &UniformVoxelGrid::aggregateAtVoxel)
        .def("get_value", &UniformVoxelGrid::getValue)
        .def("get_value_at_world", &UniformVoxelGrid::getValueAtWorld)
        .def("voxel_to_world", [](const UniformVoxelGrid& grid, int i, int j, int k) {
            float x, y, z;
            grid.voxelToWorld(i, j, k, x, y, z);
            return std::make_tuple(x, y, z);
        })
        .def("copy_from_grid", &UniformVoxelGrid::copyFromGrid)
        .def("populate_from_array", &UniformVoxelGrid::populateFromArray)
        .def("get_width", &UniformVoxelGrid::getWidth)
        .def("get_height", &UniformVoxelGrid::getHeight)
        .def("get_depth", &UniformVoxelGrid::getDepth)
        .def("get_voxel_size", &UniformVoxelGrid::getVoxelSize)
        .def("get_statistics", &UniformVoxelGrid::getStatistics)
        .def("save_to_file", &UniformVoxelGrid::saveToFile);
    
    // Statistics struct
    py::class_<UniformVoxelGrid::Statistics>(m, "UniformVoxelGridStatistics")
        .def_readwrite("total_voxels", &UniformVoxelGrid::Statistics::total_voxels)
        .def_readwrite("filled_voxels", &UniformVoxelGrid::Statistics::filled_voxels)
        .def_readwrite("fill_ratio", &UniformVoxelGrid::Statistics::fill_ratio)
        .def_readwrite("mean", &UniformVoxelGrid::Statistics::mean)
        .def_readwrite("min", &UniformVoxelGrid::Statistics::min)
        .def_readwrite("max", &UniformVoxelGrid::Statistics::max)
        .def_readwrite("std", &UniformVoxelGrid::Statistics::std);
    
    // MultiResolutionVoxelGrid
    py::class_<MultiResolutionVoxelGrid>(m, "MultiResolutionVoxelGrid")
        .def(py::init<const std::vector<float>&, float>())
        .def("get_grid", &MultiResolutionVoxelGrid::getGrid)
        .def("get_resolution", &MultiResolutionVoxelGrid::getResolution)
        .def("prolongate", &MultiResolutionVoxelGrid::prolongate)
        .def("restrict", &MultiResolutionVoxelGrid::restrict)
        .def("get_num_levels", &MultiResolutionVoxelGrid::getNumLevels)
        .def("save_to_file", &MultiResolutionVoxelGrid::saveToFile)
        .def("save_level_to_file", &MultiResolutionVoxelGrid::saveLevelToFile);
    
    // AdaptiveResolutionVoxelGrid (add_spatial_region: pybind11 does not convert list to float[3], so use 7 floats)
    py::class_<AdaptiveResolutionVoxelGrid>(m, "AdaptiveResolutionVoxelGrid")
        .def(py::init<float>())
        .def("add_spatial_region", [](AdaptiveResolutionVoxelGrid& self,
            float bbox_min_x, float bbox_min_y, float bbox_min_z,
            float bbox_max_x, float bbox_max_y, float bbox_max_z,
            float resolution) {
            float bbox_min[3] = {bbox_min_x, bbox_min_y, bbox_min_z};
            float bbox_max[3] = {bbox_max_x, bbox_max_y, bbox_max_z};
            self.addSpatialRegion(bbox_min, bbox_max, resolution);
        }, py::arg("bbox_min_x"), py::arg("bbox_min_y"), py::arg("bbox_min_z"),
           py::arg("bbox_max_x"), py::arg("bbox_max_y"), py::arg("bbox_max_z"),
           py::arg("resolution"))
        .def("add_temporal_range", &AdaptiveResolutionVoxelGrid::addTemporalRange)
        .def("get_resolution_for_point", &AdaptiveResolutionVoxelGrid::getResolutionForPoint)
        .def("get_or_create_grid", &AdaptiveResolutionVoxelGrid::getOrCreateGrid)
        .def("add_point", &AdaptiveResolutionVoxelGrid::addPoint)
        .def("get_all_grids", &AdaptiveResolutionVoxelGrid::getAllGrids)
        .def("save_to_file", &AdaptiveResolutionVoxelGrid::saveToFile);
    
    // VoxelGridFactory
    py::class_<VoxelGridFactory>(m, "VoxelGridFactory")
        .def_static("create_uniform", &VoxelGridFactory::createUniform)
        .def_static("create_multi_resolution", &VoxelGridFactory::createMultiResolution)
        .def_static("create_adaptive", &VoxelGridFactory::createAdaptive);
    
    // UnifiedGridFactory (for point transformation pipeline)
    {
        using namespace am_qadf_native::synchronization;
        py::class_<UnifiedGridFactory>(m, "UnifiedGridFactory")
            .def(py::init<>())
            .def("create_grid", [](UnifiedGridFactory& self,
                 const BoundingBox& unified_bounds,
                 const CoordinateSystem& reference_system,
                 float voxel_size,
                 float background_value = 0.0f) {
                return self.createGrid(unified_bounds, reference_system, voxel_size, background_value);
            }, py::arg("unified_bounds"), py::arg("reference_system"), py::arg("voxel_size"),
               py::arg("background_value") = 0.0f,
               "Create grid with unified bounds and reference transform")
            .def("create_grids", [](UnifiedGridFactory& self,
                 const BoundingBox& unified_bounds,
                 const CoordinateSystem& reference_system,
                 float voxel_size,
                 size_t count,
                 float background_value = 0.0f) {
                return self.createGrids(unified_bounds, reference_system, voxel_size, count, background_value);
            }, py::arg("unified_bounds"), py::arg("reference_system"), py::arg("voxel_size"),
               py::arg("count"), py::arg("background_value") = 0.0f,
               "Create multiple grids with same bounds and transform")
            .def("create_from_reference", &UnifiedGridFactory::createFromReference,
                 py::arg("reference_grid"), py::arg("unified_bounds"),
                 py::arg("background_value") = 0.0f,
                 "Create grid from existing grid's transform");
    }
    
    // NumPy <-> OpenVDB bridge (Python-only; voxelization core stays C++)
    m.def("numpy_to_openvdb", &am_qadf_native::bridge::numpyToOpenVDB);
    m.def("openvdb_to_numpy", &am_qadf_native::bridge::openVDBToNumPy);
    
    // GridType enum
    py::enum_<GridType>(m, "GridType")
        .value("UNIFORM", GridType::UNIFORM)
        .value("MULTI_RESOLUTION", GridType::MULTI_RESOLUTION)
        .value("ADAPTIVE", GridType::ADAPTIVE);
    
    // Note: CoordinateSystem and CoordinateTransformer are in synchronization namespace
    // They are bound in synchronization_bindings.cpp
    
    // STLVoxelizer
    py::class_<STLVoxelizer>(m, "STLVoxelizer")
        .def(py::init<>())
        .def("voxelize_stl", [](STLVoxelizer& self, const std::string& stl_path, float voxel_size, 
                                float half_width, bool unsigned_distance) {
            FloatGridPtr grid = self.voxelizeSTL(stl_path, voxel_size, half_width, unsigned_distance);
            // Return as shared_ptr - Pybind11 should handle std::shared_ptr automatically
            return grid;
        }, py::arg("stl_path"), py::arg("voxel_size"),
           py::arg("half_width") = 3.0f, py::arg("unsigned_distance") = false,
           py::return_value_policy::automatic)
        .def("voxelize_stl_with_signals", [](STLVoxelizer& self, const std::string& stl_path, 
                                              float voxel_size, const std::vector<std::array<float, 3>>& points,
                                              const std::vector<float>& signal_values, float half_width) {
            FloatGridPtr grid = self.voxelizeSTLWithSignals(stl_path, voxel_size, points, signal_values, half_width);
            return grid;
        }, py::arg("stl_path"), py::arg("voxel_size"),
           py::arg("points"), py::arg("signal_values"),
           py::arg("half_width") = 3.0f,
           py::return_value_policy::automatic)
        .def("get_stl_bounding_box", [](STLVoxelizer& self, const std::string& stl_path) {
            std::array<float, 3> bbox_min = {0.0f, 0.0f, 0.0f};
            std::array<float, 3> bbox_max = {0.0f, 0.0f, 0.0f};
            self.getSTLBoundingBox(stl_path, bbox_min, bbox_max);
            return std::make_tuple(bbox_min, bbox_max);
        }, py::arg("stl_path"));
    
    // HatchingPoint struct
    py::class_<HatchingPoint>(m, "HatchingPoint")
        .def(py::init<>())
        .def_readwrite("x", &HatchingPoint::x)
        .def_readwrite("y", &HatchingPoint::y)
        .def_readwrite("z", &HatchingPoint::z)
        .def_readwrite("power", &HatchingPoint::power)
        .def_readwrite("velocity", &HatchingPoint::velocity)
        .def_readwrite("energy", &HatchingPoint::energy);
    
    // HatchingVector struct (NEW: vector-based format)
    py::class_<HatchingVector>(m, "HatchingVector")
        .def(py::init<>())
        .def_readwrite("x1", &HatchingVector::x1)
        .def_readwrite("y1", &HatchingVector::y1)
        .def_readwrite("z1", &HatchingVector::z1)
        .def_readwrite("x2", &HatchingVector::x2)
        .def_readwrite("y2", &HatchingVector::y2)
        .def_readwrite("z2", &HatchingVector::z2)
        .def_readwrite("power", &HatchingVector::power)
        .def_readwrite("velocity", &HatchingVector::velocity)
        .def_readwrite("energy", &HatchingVector::energy);
    
    // HatchingVoxelizer
    py::class_<HatchingVoxelizer>(m, "HatchingVoxelizer")
        .def(py::init<>())
        .def("voxelize_hatching_paths", &HatchingVoxelizer::voxelizeHatchingPaths,
             py::arg("points"), py::arg("voxel_size"),
             py::arg("line_width") = 0.1f,
             py::arg("bbox_min") = std::array<float, 3>{0.0f, 0.0f, 0.0f},
             py::arg("bbox_max") = std::array<float, 3>{0.0f, 0.0f, 0.0f})
        .def("voxelize_contour_paths", &HatchingVoxelizer::voxelizeContourPaths,
             py::arg("points"), py::arg("voxel_size"),
             py::arg("line_width") = 0.1f,
             py::arg("bbox_min") = std::array<float, 3>{0.0f, 0.0f, 0.0f},
             py::arg("bbox_max") = std::array<float, 3>{0.0f, 0.0f, 0.0f})
        .def("voxelize_multi_layer_hatching", &HatchingVoxelizer::voxelizeMultiLayerHatching,
             py::arg("layers_points"), py::arg("voxel_size"),
             py::arg("line_width") = 0.1f,
             py::arg("bbox_min") = std::array<float, 3>{0.0f, 0.0f, 0.0f},
             py::arg("bbox_max") = std::array<float, 3>{0.0f, 0.0f, 0.0f})
        .def("voxelize_vectors", &HatchingVoxelizer::voxelizeVectors,
             py::arg("vectors"), py::arg("voxel_size"),
             py::arg("line_width") = 0.1f,
             py::arg("bbox_min") = std::array<float, 3>{0.0f, 0.0f, 0.0f},
             py::arg("bbox_max") = std::array<float, 3>{0.0f, 0.0f, 0.0f})
        .def("get_direction_grid", &HatchingVoxelizer::getDirectionGrid,
             "Get direction vector grid for arrow visualization in ParaView")
        .def("voxelize_vectors_from_arrays", &HatchingVoxelizer::voxelizeVectorsFromArrays,
             py::arg("x1"), py::arg("y1"), py::arg("z1"),
             py::arg("x2"), py::arg("y2"), py::arg("z2"),
             py::arg("power"), py::arg("velocity"), py::arg("energy"),
             py::arg("voxel_size"),
             py::arg("line_widths") = std::vector<float>(),
             py::arg("default_line_width") = 0.1f,
             py::arg("hatch_spacings") = std::vector<float>(),
             py::arg("bbox_min") = std::array<float, 3>{0.0f, 0.0f, 0.0f},
             py::arg("bbox_max") = std::array<float, 3>{0.0f, 0.0f, 0.0f})
        .def("voxelize_vectors_from_python_data", [](HatchingVoxelizer& self,
             py::list vectors_list,
             py::list vectordata_list,
             float voxel_size,
             float line_width,
             std::array<float, 3> bbox_min,
             std::array<float, 3> bbox_max) {
            // Helper function to safely convert Python value to float
            // Handles strings, floats, and ints
            auto safe_float_cast = [](py::handle value) -> float {
                try {
                    // Try direct float cast first (for float/int)
                    return py::cast<float>(value);
                } catch (const py::cast_error&) {
                    // If that fails, try converting to string then to float
                    try {
                        std::string str_val = py::cast<std::string>(value);
                        return std::stof(str_val);
                    } catch (const std::exception&) {
                        // If all else fails, return 0.0
                        return 0.0f;
                    }
                }
            };
            
            // Helper function to safely convert Python value to int
            // Handles strings, ints, and floats
            auto safe_int_cast = [](py::handle value) -> int {
                try {
                    // Try direct int cast first (for int/float)
                    return py::cast<int>(value);
                } catch (const py::cast_error&) {
                    // If that fails, try converting to string then to int
                    try {
                        std::string str_val = py::cast<std::string>(value);
                        return std::stoi(str_val);
                    } catch (const std::exception&) {
                        // If all else fails, return -1 (invalid index)
                        return -1;
                    }
                }
            };
            
            // ALL parsing happens in C++ - no Python loops
            std::vector<float> x1, y1, z1, x2, y2, z2;
            std::vector<float> power, velocity, energy;
            std::vector<float> line_widths;  // Per-vector beam widths
            std::vector<float> hatch_spacings;  // Per-vector hatch spacing (for path signal)
            
            // Build lookup map for vectordata (case-insensitive)
            std::map<int, py::dict> vectordata_map;
            for (auto item : vectordata_list) {
                py::dict vd = py::cast<py::dict>(item);
                int dataindex = -1;
                
                // Case-insensitive lookup for dataindex
                for (auto pair : vd) {
                    std::string key = py::cast<std::string>(pair.first);
                    std::transform(key.begin(), key.end(), key.begin(), ::tolower);
                    if (key == "dataindex") {
                        dataindex = safe_int_cast(pair.second);
                        break;
                    }
                }
                if (dataindex != -1) {
                    vectordata_map[dataindex] = vd;
                }
            }
            
            // Parse vectors (all in C++)
            for (auto vec_item : vectors_list) {
                py::dict vec = py::cast<py::dict>(vec_item);
                
                // Extract coordinates (case-insensitive)
                float x1_val = 0.0f, y1_val = 0.0f, z_val = 0.0f;
                float x2_val = 0.0f, y2_val = 0.0f;
                
                for (auto pair : vec) {
                    std::string key = py::cast<std::string>(pair.first);
                    std::string key_lower = key;
                    std::transform(key_lower.begin(), key_lower.end(), key_lower.begin(), ::tolower);
                    float val = safe_float_cast(pair.second);
                    
                    if (key_lower == "x1") x1_val = val;
                    else if (key_lower == "y1") y1_val = val;
                    else if (key_lower == "z") z_val = val;
                    else if (key_lower == "x2") x2_val = val;
                    else if (key_lower == "y2") y2_val = val;
                }
                
                x1.push_back(x1_val);
                y1.push_back(y1_val);
                z1.push_back(z_val);
                x2.push_back(x2_val);
                y2.push_back(y2_val);
                z2.push_back(z_val);
                
                // Get signals from vectordata
                int dataindex = -1;
                for (auto pair : vec) {
                    std::string key = py::cast<std::string>(pair.first);
                    std::string key_lower = key;
                    std::transform(key_lower.begin(), key_lower.end(), key_lower.begin(), ::tolower);
                    if (key_lower == "dataindex") {
                        dataindex = safe_int_cast(pair.second);
                        break;
                    }
                }
                
                float power_val = 200.0f, velocity_val = 500.0f, energy_val = 0.0f;
                float beam_width = line_width;  // Default to global line_width
                float hatch_spacing = 0.0f;  // Default: 0.0 means use voxel_size for sharp edges
                if (vectordata_map.find(dataindex) != vectordata_map.end()) {
                    py::dict vd = vectordata_map[dataindex];
                    for (auto pair : vd) {
                        std::string key = py::cast<std::string>(pair.first);
                        std::string key_lower = key;
                        std::transform(key_lower.begin(), key_lower.end(), key_lower.begin(), ::tolower);
                        float val = safe_float_cast(pair.second);
                        
                        // Handle multiple power field name variations (case-insensitive)
                        // Power, power, P, p, laserpower, laser_power, etc.
                        if (key_lower == "power" || key_lower == "p" || 
                            key_lower == "laserpower" || key_lower == "laser_power" ||
                            key_lower == "laser power" || key_lower == "laserpower_w" ||
                            key_lower == "power_w" || key_lower == "wattage") {
                            power_val = val;
                        }
                        // Handle multiple velocity/speed field name variations (case-insensitive)
                        // Velocity, velocity, speed, Speed, V, v, scannerspeed, scanner_speed, etc.
                        else if (key_lower == "velocity" || key_lower == "v" ||
                                 key_lower == "speed" || key_lower == "s" ||
                                 key_lower == "scannerspeed" || key_lower == "scanner_speed" ||
                                 key_lower == "scanner speed" || key_lower == "scan_speed" ||
                                 key_lower == "scanning_speed" || key_lower == "feedrate" ||
                                 key_lower == "feed_rate" || key_lower == "feed rate") {
                            velocity_val = val;
                        }
                        // Handle energy field name variations
                        else if (key_lower == "energy" || key_lower == "e" ||
                                 key_lower == "energy_density" || key_lower == "energy_dens" ||
                                 key_lower == "energy density" || key_lower == "ed") {
                            energy_val = val;
                        }
                        // Handle beam width field name variations (use per-vector beam width)
                        else if (key_lower == "laser_beam_width" || key_lower == "beam_width" ||
                                 key_lower == "beam width" || key_lower == "beamwidth" ||
                                 key_lower == "spot_size" || key_lower == "spot_size" ||
                                 key_lower == "spot size" || key_lower == "diameter" ||
                                 key_lower == "line_width" || key_lower == "line width") {
                            beam_width = val;  // Use per-vector beam width from vectordata
                        }
                        // Handle hatch spacing field name variations (for path signal visibility)
                        else if (key_lower == "hatch_spacing" || key_lower == "hatchspacing" ||
                                 key_lower == "hatch spacing" || key_lower == "spacing" ||
                                 key_lower == "hatch_distance" || key_lower == "hatchdistance" ||
                                 key_lower == "hatch distance") {
                            hatch_spacing = val;  // Use per-vector hatch spacing for path signal
                        }
                    }
                    
                    // Calculate energy if not provided and we have power, velocity, and beam_width
                    if (energy_val == 0.0f && power_val > 0 && velocity_val > 0 && beam_width > 0) {
                        energy_val = power_val / (velocity_val * beam_width);
                    }
                }
                
                power.push_back(power_val);
                velocity.push_back(velocity_val);
                energy.push_back(energy_val);
                line_widths.push_back(beam_width);  // Store per-vector beam width
                hatch_spacings.push_back(hatch_spacing);  // Store per-vector hatch spacing
            }
            
            // Call C++ method with parsed arrays (including per-vector beam widths and hatch spacings)
            return self.voxelizeVectorsFromArrays(
                x1, y1, z1, x2, y2, z2,
                power, velocity, energy,
                voxel_size, line_widths, line_width, hatch_spacings, bbox_min, bbox_max
            );
        },
        py::arg("vectors_list"), py::arg("vectordata_list"), py::arg("voxel_size"),
        py::arg("line_width") = 0.1f,
        py::arg("bbox_min") = std::array<float, 3>{0.0f, 0.0f, 0.0f},
        py::arg("bbox_max") = std::array<float, 3>{0.0f, 0.0f, 0.0f});
}
