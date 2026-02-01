#include "am_qadf_native/voxelization/stl_voxelizer.hpp"
#include <openvdb/tools/MeshToVolume.h>
#include <openvdb/tools/LevelSetUtil.h>
#include <openvdb/math/Transform.h>
#include <openvdb/math/Coord.h>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace am_qadf_native {
namespace voxelization {

FloatGridPtr STLVoxelizer::voxelizeSTL(
    const std::string& stl_path,
    float voxel_size,
    float half_width,
    bool unsigned_distance
) {
    // Load STL mesh
    std::vector<openvdb::Vec3s> points;
    std::vector<openvdb::Vec3I> triangles;
    
    if (!loadSTLMesh(stl_path, points, triangles)) {
        throw std::runtime_error("Failed to load STL file: " + stl_path);
    }
    
    if (points.empty() || triangles.empty()) {
        throw std::runtime_error("STL file is empty or invalid: " + stl_path);
    }
    
    // Create transform with specified voxel size
    openvdb::math::Transform::Ptr transform = 
        openvdb::math::Transform::createLinearTransform(voxel_size);
    
    // Create level set grid
    openvdb::FloatGrid::Ptr level_set;
    
    if (unsigned_distance) {
        // Create unsigned distance field using meshToVolume (supports flags)
        // Note: meshToVolume parameter order is (mesh, transform, exteriorWidth, interiorWidth, flags)
        openvdb::tools::QuadAndTriangleDataAdapter<openvdb::Vec3s, openvdb::Vec3I> 
            mesh(points, triangles);
        level_set = openvdb::tools::meshToVolume<openvdb::FloatGrid>(
            mesh, *transform, half_width, half_width,
            openvdb::tools::UNSIGNED_DISTANCE_FIELD
        );
    } else {
        // Create signed distance field using meshToLevelSet (expects vectors directly)
        level_set = openvdb::tools::meshToLevelSet<openvdb::FloatGrid>(
            *transform, points, triangles, half_width
        );
    }
    
    // Convert level set to binary occupancy grid (inside = 1.0, outside = 0.0)
    return levelSetToOccupancy(level_set);
}

FloatGridPtr STLVoxelizer::voxelizeSTLWithSignals(
    const std::string& stl_path,
    float voxel_size,
    const std::vector<std::array<float, 3>>& points,
    const std::vector<float>& signal_values,
    float half_width
) {
    // First voxelize the STL to get geometry
    FloatGridPtr occupancy_grid = voxelizeSTL(stl_path, voxel_size, half_width, false);
    
    if (points.size() != signal_values.size()) {
        throw std::invalid_argument("Points and signal_values must have same size");
    }
    
    // Fill grid with signal values at point locations
    auto& tree = occupancy_grid->tree();
    auto transform = occupancy_grid->transform();
    
    for (size_t i = 0; i < points.size(); ++i) {
        const auto& pt = points[i];
        float value = signal_values[i];
        
        // Convert world coordinates to voxel coordinates
        openvdb::Coord coord = transform.worldToIndexCellCentered(
            openvdb::Vec3R(pt[0], pt[1], pt[2])
        );
        
        // Only set value if voxel is inside geometry (occupancy > 0.5)
        if (tree.isValueOn(coord)) {
            float occupancy = tree.getValue(coord);
            if (occupancy > 0.5f) {
                // Set signal value
                tree.setValue(coord, value);
            }
        }
    }
    
    return occupancy_grid;
}

void STLVoxelizer::getSTLBoundingBox(
    const std::string& stl_path,
    std::array<float, 3>& bbox_min,
    std::array<float, 3>& bbox_max
) {
    std::vector<openvdb::Vec3s> points;
    std::vector<openvdb::Vec3I> triangles;
    
    if (!loadSTLMesh(stl_path, points, triangles)) {
        throw std::runtime_error("Failed to load STL file: " + stl_path);
    }
    
    if (points.empty()) {
        bbox_min = {0.0f, 0.0f, 0.0f};
        bbox_max = {0.0f, 0.0f, 0.0f};
        return;
    }
    
    // Compute bounding box
    float min_x = points[0].x(), min_y = points[0].y(), min_z = points[0].z();
    float max_x = points[0].x(), max_y = points[0].y(), max_z = points[0].z();
    
    for (const auto& pt : points) {
        min_x = std::min(min_x, pt.x());
        min_y = std::min(min_y, pt.y());
        min_z = std::min(min_z, pt.z());
        max_x = std::max(max_x, pt.x());
        max_y = std::max(max_y, pt.y());
        max_z = std::max(max_z, pt.z());
    }
    
    bbox_min = {min_x, min_y, min_z};
    bbox_max = {max_x, max_y, max_z};
}

bool STLVoxelizer::loadSTLMesh(
    const std::string& stl_path,
    std::vector<openvdb::Vec3s>& points,
    std::vector<openvdb::Vec3I>& triangles
) {
    std::ifstream file(stl_path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    // Check if binary STL (starts with "solid" in ASCII, or 80-byte header in binary)
    char header[80];
    file.read(header, 80);
    
    bool is_binary = true;
    std::string header_str(header, 80);
    // Check if it's ASCII STL (starts with "solid")
    if (header_str.find("solid") == 0 && header_str.find("endsolid") != std::string::npos) {
        is_binary = false;
    }
    
    file.seekg(0);
    
    if (is_binary) {
        // Binary STL format
        file.read(header, 80);  // Skip header
        
        uint32_t num_triangles;
        file.read(reinterpret_cast<char*>(&num_triangles), sizeof(uint32_t));
        
        points.reserve(num_triangles * 3);
        triangles.reserve(num_triangles);
        
        for (uint32_t i = 0; i < num_triangles; ++i) {
            // Skip normal (12 bytes)
            file.seekg(12, std::ios::cur);
            
            // Read 3 vertices (36 bytes)
            float v[9];  // 3 vertices * 3 coordinates
            file.read(reinterpret_cast<char*>(v), 36);
            
            // Add vertices to points list
            size_t base_idx = points.size();
            points.push_back(openvdb::Vec3s(v[0], v[1], v[2]));
            points.push_back(openvdb::Vec3s(v[3], v[4], v[5]));
            points.push_back(openvdb::Vec3s(v[6], v[7], v[8]));
            
            // Add triangle
            triangles.push_back(openvdb::Vec3I(
                static_cast<int>(base_idx),
                static_cast<int>(base_idx + 1),
                static_cast<int>(base_idx + 2)
            ));
            
            // Skip attribute byte count (2 bytes)
            file.seekg(2, std::ios::cur);
        }
    } else {
        // ASCII STL format
        file.seekg(0);
        std::string line;
        std::vector<openvdb::Vec3s> temp_points;
        
        while (std::getline(file, line)) {
            // Trim whitespace
            line.erase(0, line.find_first_not_of(" \t\r\n"));
            line.erase(line.find_last_not_of(" \t\r\n") + 1);
            
            if (line.empty()) continue;
            
            std::istringstream iss(line);
            std::string keyword;
            iss >> keyword;
            
            if (keyword == "vertex") {
                float x, y, z;
                iss >> x >> y >> z;
                temp_points.push_back(openvdb::Vec3s(x, y, z));
            } else if (keyword == "endfacet") {
                // End of triangle - add triangle if we have 3 vertices
                if (temp_points.size() >= 3) {
                    size_t base_idx = points.size();
                    // Add last 3 vertices (the triangle)
                    points.push_back(temp_points[temp_points.size() - 3]);
                    points.push_back(temp_points[temp_points.size() - 2]);
                    points.push_back(temp_points[temp_points.size() - 1]);
                    
                    triangles.push_back(openvdb::Vec3I(
                        static_cast<int>(base_idx),
                        static_cast<int>(base_idx + 1),
                        static_cast<int>(base_idx + 2)
                    ));
                }
                // Clear temp points for next facet
                temp_points.clear();
            } else if (keyword == "endsolid") {
                break;
            }
        }
    }
    
    return !points.empty() && !triangles.empty();
}

FloatGridPtr STLVoxelizer::levelSetToOccupancy(FloatGridPtr level_set) {
    // Convert signed distance field to binary occupancy
    // Inside (negative distance) = 1.0, Outside (positive distance) = 0.0
    // 
    // CRITICAL: We need to mark ALL interior voxels as active, not just those in the narrow band.
    // 
    // How OpenVDB Level Sets Work:
    // - Level sets store a signed distance field (SDF) in a narrow band around the surface
    // - Negative distance = inside mesh, Zero = on surface, Positive = outside mesh
    // - Only voxels within narrow band (half_width) are stored as "active"
    // - Inactive voxels use background value (typically large positive = exterior)
    //
    // Strategy to Fill ALL Interior Voxels:
    // 1. Get bounding box of active voxels (defines mesh region)
    // 2. Expand bounding box to cover entire mesh interior
    // 3. Iterate over ALL voxels in expanded bounding box
    // 4. For each voxel, query the level set's distance value
    //    - OpenVDB's level set can compute distance for any coordinate
    //    - For active voxels: returns stored distance
    //    - For inactive voxels: computes distance using the level set field
    // 5. Mark as occupied (1.0) if distance <= 0 (inside or on surface)
    
    auto occupancy = FloatGrid::create();
    occupancy->setTransform(level_set->transform().copy());
    occupancy->setName(level_set->getName());
    
    auto& occupancy_tree = occupancy->tree();
    const auto& level_set_tree = level_set->tree();
    
    // Get bounding box of active voxels (defines the mesh surface region)
    openvdb::CoordBBox active_bbox;
    if (!level_set_tree.evalActiveVoxelBoundingBox(active_bbox)) {
        // No active voxels - return empty occupancy grid
        return occupancy;
    }
    
    // Expand bounding box to ensure we cover the entire mesh interior
    // The expansion should be large enough to cover the largest interior dimension
    // We expand by a conservative amount to ensure we don't miss any interior voxels
    active_bbox.expand(openvdb::Coord(10, 10, 10));  // Expand by 10 voxels in each direction
    
    // Iterate over ALL voxels in the expanded bounding box
    // This ensures we mark all interior voxels, not just those in the narrow band
    for (auto iter = active_bbox.begin(); iter; ++iter) {
        const openvdb::Coord& coord = *iter;
        
        // Get distance value for this voxel
        // OpenVDB's level set getValue() computes the actual distance field value
        // For active voxels: returns stored distance
        // For inactive voxels: computes distance using the level set's distance field
        float distance = level_set->tree().getValue(coord);
        
        // Negative distance = inside mesh, zero = on surface, positive = outside
        // Mark as occupied if distance <= 0 (inside or on surface)
        if (distance <= 0.0f) {
            occupancy_tree.setValue(coord, 1.0f);
        }
        // For exterior voxels (distance > 0), we don't store them (keeps grid sparse)
    }
    
    // Result: All voxels inside the mesh (distance <= 0) are now marked as active
    // with value 1.0, regardless of whether they were in the narrow band or not.
    // This ensures complete interior filling for accurate 3D geometry visualization.
    
    return occupancy;
}

} // namespace voxelization
} // namespace am_qadf_native
