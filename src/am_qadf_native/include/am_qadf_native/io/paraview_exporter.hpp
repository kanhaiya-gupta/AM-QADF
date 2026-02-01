#ifndef AM_QADF_NATIVE_IO_PARAVIEW_EXPORTER_HPP
#define AM_QADF_NATIVE_IO_PARAVIEW_EXPORTER_HPP

#include "vdb_writer.hpp"
#include <string>
#include <vector>
#include <memory>

namespace am_qadf_native {
namespace io {

using FloatGrid = openvdb::FloatGrid;
using FloatGridPtr = FloatGrid::Ptr;

// ParaView exporter
class ParaViewExporter {
private:
    VDBWriter writer_;
    
public:
    // Export single grid to ParaView (.vdb file)
    void exportToParaView(FloatGridPtr grid, const std::string& filename);
    
    // Export multiple grids to ParaView
    void exportMultipleToParaView(
        const std::vector<FloatGridPtr>& grids,
        const std::string& filename
    );
    
    // Export with metadata for ParaView
    void exportWithMetadata(
        FloatGridPtr grid,
        const std::string& filename,
        const std::map<std::string, std::string>& metadata
    );
};

} // namespace io
} // namespace am_qadf_native

#endif // AM_QADF_NATIVE_IO_PARAVIEW_EXPORTER_HPP
