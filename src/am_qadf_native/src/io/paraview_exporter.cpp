#include "am_qadf_native/io/paraview_exporter.hpp"
#include <openvdb/openvdb.h>
#include <map>
#include <string>
#include <vector>
#include <memory>

namespace am_qadf_native {
namespace io {

void ParaViewExporter::exportToParaView(FloatGridPtr grid, const std::string& filename) {
    // Export OpenVDB grid to .vdb file
    // ParaView can open .vdb files directly
    writer_.write(grid, filename);
}

void ParaViewExporter::exportMultipleToParaView(
    const std::vector<FloatGridPtr>& grids,
    const std::string& filename
) {
    writer_.writeMultiple(grids, filename);
}

void ParaViewExporter::exportWithMetadata(
    FloatGridPtr grid,
    const std::string& filename,
    const std::map<std::string, std::string>& metadata
) {
    // Set metadata on grid
    for (const auto& pair : metadata) {
        grid->insertMeta(pair.first, openvdb::StringMetadata(pair.second));
    }
    
    writer_.write(grid, filename);
}

} // namespace io
} // namespace am_qadf_native
