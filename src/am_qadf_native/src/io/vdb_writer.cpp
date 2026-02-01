#include "am_qadf_native/io/vdb_writer.hpp"
#include <openvdb/io/File.h>
#include <openvdb/io/Compression.h>
#include <openvdb/tools/Prune.h>
#include <openvdb/math/Transform.h>
#include <vector>
#include <string>
#include <memory>

namespace am_qadf_native {
namespace io {

// Performance Optimization Notes:
// 1. BLOSC compression: Significantly faster than ZLIB for compression/decompression
//    while maintaining similar compression ratios. Ideal for large files (10GB+).
// 2. Grid pruning: Removes empty tiles and optimizes sparse structure, reducing
//    file size and improving load times in ParaView.
// 3. These optimizations are critical for manufacturing data which can be 10GB+.
//    Without compression, even 1GB files can be slow to load in ParaView.

void VDBWriter::write(FloatGridPtr grid, const std::string& filename) {
    // Optimize grid before writing (prune empty tiles, compact)
    openvdb::tools::prune(grid->tree());
    
    openvdb::io::File file(filename);
    // Use BLOSC compression for better performance (faster than ZLIB, similar compression ratio)
    file.setCompression(openvdb::io::COMPRESS_BLOSC);
    
    openvdb::GridPtrVec grids;
    grids.push_back(grid);
    file.write(grids);
    file.close();
}

void VDBWriter::writeMultiple(
    const std::vector<FloatGridPtr>& grids,
    const std::string& filename
) {
    // Optimize all grids before writing
    for (auto grid : grids) {
        openvdb::tools::prune(grid->tree());
    }
    
    openvdb::io::File file(filename);
    // Use BLOSC compression for better performance
    file.setCompression(openvdb::io::COMPRESS_BLOSC);
    
    openvdb::GridPtrVec grid_vec;
    for (auto grid : grids) {
        grid_vec.push_back(grid);
    }
    file.write(grid_vec);
    file.close();
}

void VDBWriter::writeMultipleWithNames(
    const std::vector<FloatGridPtr>& grids,
    const std::vector<std::string>& grid_names,
    const std::string& filename
) {
    if (grids.size() != grid_names.size()) {
        throw std::invalid_argument("Number of grids must match number of names");
    }
    
    if (grids.empty()) {
        return;
    }
    
    // Use first grid's transform as reference so all grids are written with identical
    // transform metadata. This ensures ParaView (and other viewers) show all volumes
    // in the same place; otherwise per-grid transform differences can cause signals
    // to appear at different locations in volume view.
    const openvdb::math::Transform& ref_xform = grids[0]->transform();
    openvdb::math::Transform::Ptr ref_xform_copy(
        new openvdb::math::Transform(ref_xform));
    
    // Optimize and normalize transform for all grids before writing
    for (size_t i = 0; i < grids.size(); ++i) {
        openvdb::tools::prune(grids[i]->tree());
        grids[i]->setName(grid_names[i]);
        grids[i]->setTransform(ref_xform_copy);
    }
    
    openvdb::io::File file(filename);
    file.setCompression(openvdb::io::COMPRESS_BLOSC);
    
    openvdb::GridPtrVec grid_vec;
    for (auto grid : grids) {
        grid_vec.push_back(grid);
    }
    file.write(grid_vec);
    file.close();
}

void VDBWriter::writeCompressed(
    FloatGridPtr grid,
    const std::string& filename,
    int compression_level
) {
    // Optimize grid before writing
    openvdb::tools::prune(grid->tree());
    
    openvdb::io::File file(filename);
    
    // Set compression flags
    // OpenVDB supports: BLOSC (fast, good compression), ZLIB (slower, good compression), B42A
    // BLOSC is recommended for large files due to faster compression/decompression
    // Note: compression_level parameter is not directly used by OpenVDB's setCompression,
    // but BLOSC internally uses good defaults
    file.setCompression(openvdb::io::COMPRESS_BLOSC);
    
    openvdb::GridPtrVec grids;
    grids.push_back(grid);
    file.write(grids);
    file.close();
}

void VDBWriter::append(FloatGridPtr grid, const std::string& filename) {
    // Append grid to existing file
    // Load existing grids, add new grid, write all
    
    // Optimize new grid before writing
    openvdb::tools::prune(grid->tree());
    
    openvdb::io::File file(filename);
    
    openvdb::GridPtrVec all_grids;
    
    // Try to open existing file and read grids
    try {
        file.open();
        auto existing_grids = file.getGrids();
        file.close();
        
        // Copy existing grids
        if (existing_grids) {
            for (auto& existing_grid : *existing_grids) {
                all_grids.push_back(existing_grid);
            }
        }
    } catch (...) {
        // File doesn't exist or cannot be opened - will create new file
        file.close();
    }
    
    // Add new grid
    all_grids.push_back(grid);
    
    // Use BLOSC compression for better performance
    file.setCompression(openvdb::io::COMPRESS_BLOSC);
    
    // Write all grids
    file.write(all_grids);
    file.close();
}

} // namespace io
} // namespace am_qadf_native
