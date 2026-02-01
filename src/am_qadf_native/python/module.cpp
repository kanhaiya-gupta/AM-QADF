#include <pybind11/pybind11.h>
#include <openvdb/openvdb.h>

// Binding modules
void bind_voxelization(pybind11::module& m);
void bind_signal_mapping(pybind11::module& m);
void bind_fusion(pybind11::module& m);
void bind_synchronization(pybind11::module& m);
void bind_query(pybind11::module& m);
void bind_visualization(pybind11::module& m);
void bind_correction(pybind11::module& m);
void bind_processing(pybind11::module& m);
void bind_io(pybind11::module& m);

PYBIND11_MODULE(am_qadf_native, m) {
    m.doc() = "AM-QADF Native C++ Module with OpenVDB Integration";
    
    // Initialize OpenVDB
    openvdb::initialize();
    
    // Bind all modules
    bind_voxelization(m);
    bind_signal_mapping(m);
    bind_fusion(m);
    bind_synchronization(m);
    bind_query(m);
    bind_visualization(m);
    bind_correction(m);
    bind_processing(m);
    bind_io(m);
}
