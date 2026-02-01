#ifndef AM_QADF_NATIVE_IO_VDB_WRITER_HPP
#define AM_QADF_NATIVE_IO_VDB_WRITER_HPP

#include <openvdb/openvdb.h>
#include <openvdb/io/File.h>
#include <string>
#include <vector>
#include <memory>

namespace am_qadf_native {
namespace io {

using FloatGrid = openvdb::FloatGrid;
using FloatGridPtr = FloatGrid::Ptr;

// OpenVDB writer
class VDBWriter {
public:
    // Write single grid to file
    void write(FloatGridPtr grid, const std::string& filename);
    
    // Write multiple grids to file
    void writeMultiple(
        const std::vector<FloatGridPtr>& grids,
        const std::string& filename
    );
    
    // Write multiple grids with names (for multi-signal storage)
    void writeMultipleWithNames(
        const std::vector<FloatGridPtr>& grids,
        const std::vector<std::string>& grid_names,
        const std::string& filename
    );
    
    // Write with compression
    void writeCompressed(
        FloatGridPtr grid,
        const std::string& filename,
        int compression_level = 6
    );
    
    // Append grid to existing file
    void append(FloatGridPtr grid, const std::string& filename);
};

} // namespace io
} // namespace am_qadf_native

#endif // AM_QADF_NATIVE_IO_VDB_WRITER_HPP
