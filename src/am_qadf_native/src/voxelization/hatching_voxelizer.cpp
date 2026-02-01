#include "am_qadf_native/voxelization/hatching_voxelizer.hpp"
#include <openvdb/math/Transform.h>
#include <openvdb/math/Coord.h>
#include <openvdb/math/Vec3.h>
#include <openvdb/math/BBox.h>
#include <algorithm>
#include <cmath>
#include <limits>

namespace am_qadf_native {
namespace voxelization {

std::map<std::string, FloatGridPtr> HatchingVoxelizer::createSignalGrids(
    float voxel_size,
    const std::array<float, 3>& bbox_min,
    const std::array<float, 3>& /* bbox_max */
) {
    std::map<std::string, FloatGridPtr> grids;
    
    // Create transform: world = index * voxel_size + bbox_min (same as UniformVoxelGrid)
    // So fusion notebook and signal-mapped grids share the same transform when using union bounds.
    openvdb::math::Transform::Ptr transform =
        openvdb::math::Transform::createLinearTransform(static_cast<double>(voxel_size));
    transform->postTranslate(openvdb::Vec3d(bbox_min[0], bbox_min[1], bbox_min[2]));
    
    // Create grids for each signal
    auto power_grid = FloatGrid::create(0.0f);
    power_grid->setTransform(transform);
    power_grid->setName("power");
    grids["power"] = power_grid;
    
    auto velocity_grid = FloatGrid::create(0.0f);
    velocity_grid->setTransform(transform);
    velocity_grid->setName("velocity");
    grids["velocity"] = velocity_grid;
    
    auto energy_grid = FloatGrid::create(0.0f);
    energy_grid->setTransform(transform);
    energy_grid->setName("energy");
    grids["energy"] = energy_grid;
    
    // Create "path" signal grid - constant value 1.0 wherever path exists
    // Useful for uniform visualization without color variation
    auto path_grid = FloatGrid::create(0.0f);
    path_grid->setTransform(transform);
    path_grid->setName("path");
    grids["path"] = path_grid;
    
    return grids;
}

void HatchingVoxelizer::computeBoundingBox(
    const std::vector<HatchingPoint>& points,
    std::array<float, 3>& bbox_min,
    std::array<float, 3>& bbox_max
) {
    if (points.empty()) {
        bbox_min = {0.0f, 0.0f, 0.0f};
        bbox_max = {0.0f, 0.0f, 0.0f};
        return;
    }
    
    float min_x = points[0].x, min_y = points[0].y, min_z = points[0].z;
    float max_x = points[0].x, max_y = points[0].y, max_z = points[0].z;
    
    for (const auto& pt : points) {
        min_x = std::min(min_x, pt.x);
        min_y = std::min(min_y, pt.y);
        min_z = std::min(min_z, pt.z);
        max_x = std::max(max_x, pt.x);
        max_y = std::max(max_y, pt.y);
        max_z = std::max(max_z, pt.z);
    }
    
    bbox_min = {min_x, min_y, min_z};
    bbox_max = {max_x, max_y, max_z};
}

std::vector<openvdb::Coord> HatchingVoxelizer::getVoxelsAlongLine(
    const openvdb::math::Transform& transform,
    const std::array<float, 3>& start,
    const std::array<float, 3>& end,
    float line_width
) {
    std::vector<openvdb::Coord> voxels;
    
    // Convert world coordinates to voxel coordinates
    openvdb::Vec3R start_world(start[0], start[1], start[2]);
    openvdb::Vec3R end_world(end[0], end[1], end[2]);
    
    openvdb::Coord start_coord = transform.worldToIndexCellCentered(start_world);
    openvdb::Coord end_coord = transform.worldToIndexCellCentered(end_world);
    
    // 3D Bresenham-like line algorithm
    int dx = std::abs(end_coord.x() - start_coord.x());
    int dy = std::abs(end_coord.y() - start_coord.y());
    int dz = std::abs(end_coord.z() - start_coord.z());
    
    int xs = (start_coord.x() < end_coord.x()) ? 1 : -1;
    int ys = (start_coord.y() < end_coord.y()) ? 1 : -1;
    int zs = (start_coord.z() < end_coord.z()) ? 1 : -1;
    
    // Calculate number of voxels along line
    int num_steps = std::max({dx, dy, dz});
    if (num_steps == 0) {
        voxels.push_back(start_coord);
        return voxels;
    }
    
    // Iterate along line
    int x = start_coord.x();
    int y = start_coord.y();
    int z = start_coord.z();
    
    int p1 = 2 * dy - dx;
    int p2 = 2 * dx - dy;
    int p3 = 2 * dz - dx;
    
    voxels.push_back(openvdb::Coord(x, y, z));
    
    for (int i = 0; i < num_steps; ++i) {
        if (p1 >= 0) {
            y += ys;
            p1 -= 2 * dx;
        }
        if (p2 >= 0) {
            x += xs;
            p2 -= 2 * dy;
        }
        if (p3 >= 0) {
            z += zs;
            p3 -= 2 * dx;
        }
        p1 += 2 * dy;
        p2 += 2 * dx;
        p3 += 2 * dz;
        
        voxels.push_back(openvdb::Coord(x, y, z));
    }
    
    // Account for line width - add neighboring voxels
    // If line_width <= voxel_size, return only line voxels (sharp edges for visualization)
    // Otherwise, expand with radius (blurred edges for physical signals)
    if (line_width > 0.0f) {
        float voxel_size = transform.voxelSize()[0];
        
        // Sharp edges: if line_width <= voxel_size, return only line voxels (no expansion)
        if (line_width <= voxel_size) {
            return voxels;  // No expansion - sharp edges!
        }
        
        // Blurred edges: expand with radius for physical signals
        std::vector<openvdb::Coord> expanded_voxels;
        int radius_voxels = static_cast<int>(std::ceil(line_width / voxel_size));
        
        for (const auto& coord : voxels) {
            // Add voxel itself
            expanded_voxels.push_back(coord);
            
            // Add neighboring voxels within radius
            for (int dx = -radius_voxels; dx <= radius_voxels; ++dx) {
                for (int dy = -radius_voxels; dy <= radius_voxels; ++dy) {
                    for (int dz = -radius_voxels; dz <= radius_voxels; ++dz) {
                        if (dx == 0 && dy == 0 && dz == 0) continue;
                        
                        float dist = std::sqrt(dx*dx + dy*dy + dz*dz) * voxel_size;
                        if (dist <= line_width / 2.0f) {
                            expanded_voxels.push_back(openvdb::Coord(
                                coord.x() + dx,
                                coord.y() + dy,
                                coord.z() + dz
                            ));
                        }
                    }
                }
            }
        }
        
        // Remove duplicates
        std::sort(expanded_voxels.begin(), expanded_voxels.end());
        expanded_voxels.erase(
            std::unique(expanded_voxels.begin(), expanded_voxels.end()),
            expanded_voxels.end()
        );
        
        return expanded_voxels;
    }
    
    return voxels;
}

void HatchingVoxelizer::setValueAlongLine(
    FloatGridPtr grid,
    const std::array<float, 3>& start,
    const std::array<float, 3>& end,
    float start_value,
    float end_value,
    float line_width
) {
    auto& tree = grid->tree();
    const auto& transform = grid->transform();
    
    // Get all voxels along line
    std::vector<openvdb::Coord> voxels = getVoxelsAlongLine(transform, start, end, line_width);
    
    if (voxels.empty()) return;
    
    // Calculate line length for interpolation
    float dx = end[0] - start[0];
    float dy = end[1] - start[1];
    float dz = end[2] - start[2];
    float line_length = std::sqrt(dx*dx + dy*dy + dz*dz);
    
    // Set values in voxels (interpolate along line)
    for (const auto& coord : voxels) {
        // Convert voxel coord back to world to calculate position along line
        openvdb::Vec3R world_pos = transform.indexToWorld(coord);
        
        // Calculate parameter t along line (0 = start, 1 = end)
        float t = 0.0f;
        if (line_length > 1e-6f) {
            float dx_local = world_pos.x() - start[0];
            float dy_local = world_pos.y() - start[1];
            float dz_local = world_pos.z() - start[2];
            
            // Project onto line direction
            float proj = (dx_local * dx + dy_local * dy + dz_local * dz) / (line_length * line_length);
            t = std::max(0.0f, std::min(1.0f, proj));
        }
        
        // Interpolate signal value
        float value = start_value * (1.0f - t) + end_value * t;
        
        // Set value (use max to handle overlapping paths)
        float current_value = tree.getValue(coord);
        tree.setValue(coord, std::max(current_value, value));
    }
}

void HatchingVoxelizer::voxelizeLineSegment(
    const HatchingPoint& p1,
    const HatchingPoint& p2,
    FloatGridPtr power_grid,
    FloatGridPtr velocity_grid,
    FloatGridPtr energy_grid,
    FloatGridPtr path_grid,  // Path signal: constant value 1.0
    float line_width,         // Beam width for physical signals
    float hatch_spacing,      // Hatch spacing (0.0 = not used; >0 = limits expansion to respect gaps)
    float path_line_width,    // Path width (0.0 = sharp, uses voxel_size; >0 = blurred)
    Vec3fGridPtr direction_grid
) {
    std::array<float, 3> start = {p1.x, p1.y, p1.z};
    std::array<float, 3> end = {p2.x, p2.y, p2.z};
    
    // Calculate direction vector (normalized)
    float dx = end[0] - start[0];
    float dy = end[1] - start[1];
    float dz = end[2] - start[2];
    float length = std::sqrt(dx*dx + dy*dy + dz*dz);
    
    std::array<float, 3> direction = {0.0f, 0.0f, 0.0f};
    if (length > 1e-6f) {
        // Normalize direction vector
        direction[0] = dx / length;
        direction[1] = dy / length;
        direction[2] = dz / length;
    }
    
    // Calculate effective line width for physical signals considering hatch spacing
    // If hatch_spacing > 0, use min(line_width, hatch_spacing) to respect gaps
    // This ensures we don't blur power/energy across gaps when paths are far apart
    // Energy density formula: E = P / (v * h) where h = hatch_spacing
    // So hatch_spacing directly affects energy distribution
    float effective_line_width = line_width;
    if (hatch_spacing > 0.0f) {
        // Use min to respect hatch spacing boundaries (don't blur across gaps)
        // If hatch_spacing < line_width: paths overlap (realistic - intentional overlap)
        // If hatch_spacing > line_width: gaps will occur naturally (realistic - no blur across gap)
        // Using min ensures we don't expand beyond hatch spacing when it's smaller
        effective_line_width = std::min(line_width, hatch_spacing);
    }
    
    // Set signal values along line segment (physical signals use effective line width)
    // This respects hatch spacing for realistic power/energy visualization
    setValueAlongLine(power_grid, start, end, p1.power, p2.power, effective_line_width);
    setValueAlongLine(velocity_grid, start, end, p1.velocity, p2.velocity, effective_line_width);
    setValueAlongLine(energy_grid, start, end, p1.energy, p2.energy, effective_line_width);
    
    // Set path signal to constant value 1.0 wherever path exists
    // Use path_line_width for sharp edges (0.0 = sharp, only line voxels)
    if (path_grid) {
        // If path_line_width is 0.0 or negative, use voxel_size for sharp edges (1 voxel)
        // Otherwise use the specified path_line_width
        float actual_path_width = (path_line_width > 0.0f) 
                                  ? path_line_width 
                                  : power_grid->voxelSize()[0];  // Use voxel_size for sharp edges
        setValueAlongLine(path_grid, start, end, 1.0f, 1.0f, actual_path_width);
    }
    
    // Set direction vectors along line segment (for arrow visualization in ParaView)
    // Use effective_line_width for direction vectors (they represent physical beam direction)
    if (direction_grid && length > 1e-6f) {
        setDirectionAlongLine(direction_grid, start, end, direction, effective_line_width);
    }
}

void HatchingVoxelizer::setDirectionAlongLine(
    Vec3fGridPtr direction_grid,
    const std::array<float, 3>& start,
    const std::array<float, 3>& end,
    const std::array<float, 3>& direction,
    float line_width
) {
    auto& tree = direction_grid->tree();
    const auto& transform = direction_grid->transform();
    
    // Get all voxels along line
    std::vector<openvdb::Coord> voxels = getVoxelsAlongLine(transform, start, end, line_width);
    
    if (voxels.empty()) return;
    
    // Set direction vector in all voxels along the line
    // For overlapping vectors, we use the most recent direction (overwrite)
    openvdb::Vec3f dir_vec(direction[0], direction[1], direction[2]);
    
    for (const auto& coord : voxels) {
        // Set direction vector (normalized, unit length)
        tree.setValue(coord, dir_vec);
    }
}

std::map<std::string, FloatGridPtr> HatchingVoxelizer::voxelizeHatchingPaths(
    const std::vector<HatchingPoint>& points,
    float voxel_size,
    float line_width,
    const std::array<float, 3>& bbox_min,
    const std::array<float, 3>& bbox_max
) {
    if (points.empty()) {
        // Return minimal grids (power, velocity, energy, path) so callers get expected keys
        std::array<float, 3> zero = {0.0f, 0.0f, 0.0f};
        return createSignalGrids(voxel_size, zero, zero);
    }
    
    // Compute bounding box if not provided
    std::array<float, 3> actual_bbox_min = bbox_min;
    std::array<float, 3> actual_bbox_max = bbox_max;
    
    if (bbox_min[0] == 0.0f && bbox_min[1] == 0.0f && bbox_min[2] == 0.0f &&
        bbox_max[0] == 0.0f && bbox_max[1] == 0.0f && bbox_max[2] == 0.0f) {
        computeBoundingBox(points, actual_bbox_min, actual_bbox_max);
    }
    
    // Create signal grids
    auto grids = createSignalGrids(voxel_size, actual_bbox_min, actual_bbox_max);
    FloatGridPtr power_grid = grids["power"];
    FloatGridPtr velocity_grid = grids["velocity"];
    FloatGridPtr energy_grid = grids["energy"];
    FloatGridPtr path_grid = grids["path"];
    
    // Create direction vector grid (same transform as signal grids for fusion alignment)
    direction_grid_ = Vec3fGrid::create(openvdb::Vec3f(0.0f, 0.0f, 0.0f));
    direction_grid_->setTransform(openvdb::math::Transform::Ptr(new openvdb::math::Transform(power_grid->transform())));
    direction_grid_->setName("direction");
    
    // Voxelize each line segment
    // Use path_line_width = 0.0 for sharp path edges (default: sharp)
    // Use hatch_spacing = 0.0 (not available in this method - would need to add parameter)
    for (size_t i = 0; i < points.size() - 1; ++i) {
        voxelizeLineSegment(
            points[i],
            points[i + 1],
            power_grid,
            velocity_grid,
            energy_grid,
            path_grid,
            line_width,
            0.0f,  // hatch_spacing = 0.0 (not available)
            0.0f,  // path_line_width = 0.0 means sharp edges (only line voxels)
            direction_grid_
        );
    }
    
    return grids;
}

std::map<std::string, FloatGridPtr> HatchingVoxelizer::voxelizeContourPaths(
    const std::vector<HatchingPoint>& points,
    float voxel_size,
    float line_width,
    const std::array<float, 3>& bbox_min,
    const std::array<float, 3>& bbox_max
) {
    // Contours are similar to hatching but typically closed loops
    // Close the loop if first and last points are not the same
    std::vector<HatchingPoint> closed_points = points;
    
    if (points.size() >= 3) {
        float dist = std::sqrt(
            (points[0].x - points.back().x) * (points[0].x - points.back().x) +
            (points[0].y - points.back().y) * (points[0].y - points.back().y) +
            (points[0].z - points.back().z) * (points[0].z - points.back().z)
        );
        
        // If not closed (distance > threshold), close it
        if (dist > voxel_size * 0.1f) {
            closed_points.push_back(points[0]);
        }
    }
    
    return voxelizeHatchingPaths(closed_points, voxel_size, line_width, bbox_min, bbox_max);
}

std::map<std::string, FloatGridPtr> HatchingVoxelizer::voxelizeMultiLayerHatching(
    const std::vector<std::vector<HatchingPoint>>& layers_points,
    float voxel_size,
    float line_width,
    const std::array<float, 3>& bbox_min,
    const std::array<float, 3>& bbox_max
) {
    if (layers_points.empty()) {
        return {};
    }
    
    // Compute bounding box from all layers
    std::array<float, 3> actual_bbox_min = bbox_min;
    std::array<float, 3> actual_bbox_max = bbox_max;
    
    if (bbox_min[0] == 0.0f && bbox_min[1] == 0.0f && bbox_min[2] == 0.0f &&
        bbox_max[0] == 0.0f && bbox_max[1] == 0.0f && bbox_max[2] == 0.0f) {
        // Collect all points from all layers
        std::vector<HatchingPoint> all_points;
        for (const auto& layer_points : layers_points) {
            all_points.insert(all_points.end(), layer_points.begin(), layer_points.end());
        }
        computeBoundingBox(all_points, actual_bbox_min, actual_bbox_max);
    }
    
    // Create signal grids
    auto grids = createSignalGrids(voxel_size, actual_bbox_min, actual_bbox_max);
    FloatGridPtr power_grid = grids["power"];
    FloatGridPtr velocity_grid = grids["velocity"];
    FloatGridPtr energy_grid = grids["energy"];
    FloatGridPtr path_grid = grids["path"];
    
    // Create direction vector grid (same transform as signal grids for fusion alignment)
    direction_grid_ = Vec3fGrid::create(openvdb::Vec3f(0.0f, 0.0f, 0.0f));
    direction_grid_->setTransform(openvdb::math::Transform::Ptr(new openvdb::math::Transform(power_grid->transform())));
    direction_grid_->setName("direction");
    
    // Voxelize each layer
    // Use path_line_width = 0.0 for sharp path edges (default: sharp)
    for (const auto& layer_points : layers_points) {
        for (size_t i = 0; i < layer_points.size() - 1; ++i) {
            voxelizeLineSegment(
                layer_points[i],
                layer_points[i + 1],
                power_grid,
                velocity_grid,
                energy_grid,
                path_grid,
                line_width,
                0.0f,  // hatch_spacing = 0.0 (not available)
                0.0f,  // path_line_width = 0.0 means sharp edges (only line voxels)
                direction_grid_
            );
        }
    }
    
    return grids;
}

std::map<std::string, FloatGridPtr> HatchingVoxelizer::voxelizeVectors(
    const std::vector<HatchingVector>& vectors,
    float voxel_size,
    float line_width,
    const std::array<float, 3>& bbox_min,
    const std::array<float, 3>& bbox_max
) {
    if (vectors.empty()) {
        std::array<float, 3> zero = {0.0f, 0.0f, 0.0f};
        return createSignalGrids(voxel_size, zero, zero);
    }
    
    // Compute bounding box from vectors if not provided
    std::array<float, 3> actual_bbox_min = bbox_min;
    std::array<float, 3> actual_bbox_max = bbox_max;
    
    if (bbox_min[0] == 0.0f && bbox_min[1] == 0.0f && bbox_min[2] == 0.0f &&
        bbox_max[0] == 0.0f && bbox_max[1] == 0.0f && bbox_max[2] == 0.0f) {
        // Compute bounding box from all vector endpoints
        float min_x = vectors[0].x1, min_y = vectors[0].y1, min_z = vectors[0].z1;
        float max_x = vectors[0].x1, max_y = vectors[0].y1, max_z = vectors[0].z1;
        
        for (const auto& vec : vectors) {
            min_x = std::min({min_x, vec.x1, vec.x2});
            min_y = std::min({min_y, vec.y1, vec.y2});
            min_z = std::min({min_z, vec.z1, vec.z2});
            max_x = std::max({max_x, vec.x1, vec.x2});
            max_y = std::max({max_y, vec.y1, vec.y2});
            max_z = std::max({max_z, vec.z1, vec.z2});
        }
        
        actual_bbox_min = {min_x, min_y, min_z};
        actual_bbox_max = {max_x, max_y, max_z};
    }
    
    // Create signal grids
    auto grids = createSignalGrids(voxel_size, actual_bbox_min, actual_bbox_max);
    FloatGridPtr power_grid = grids["power"];
    FloatGridPtr velocity_grid = grids["velocity"];
    FloatGridPtr energy_grid = grids["energy"];
    FloatGridPtr path_grid = grids["path"];
    
    // Create direction vector grid (same transform as signal grids for fusion alignment)
    direction_grid_ = Vec3fGrid::create(openvdb::Vec3f(0.0f, 0.0f, 0.0f));
    direction_grid_->setTransform(openvdb::math::Transform::Ptr(new openvdb::math::Transform(power_grid->transform())));
    direction_grid_->setName("direction");
    
    // Voxelize each vector independently (each vector is a separate line segment)
    // This is the key difference: vectors are NOT connected to each other
    for (const auto& vec : vectors) {
        // Create HatchingPoint structs for start and end
        HatchingPoint p1, p2;
        p1.x = vec.x1; p1.y = vec.y1; p1.z = vec.z1;
        p1.power = vec.power; p1.velocity = vec.velocity; p1.energy = vec.energy;
        
        p2.x = vec.x2; p2.y = vec.y2; p2.z = vec.z2;
        p2.power = vec.power; p2.velocity = vec.velocity; p2.energy = vec.energy;
        
        // Voxelize this single line segment (including direction vectors)
        // Use path_line_width = 0.0 for sharp path edges (default: sharp)
        // Use hatch_spacing = 0.0 (not available in this method - would need to add parameter)
        voxelizeLineSegment(p1, p2, power_grid, velocity_grid, energy_grid, path_grid, line_width, 0.0f, 0.0f, direction_grid_);
    }
    
    return grids;
}

std::map<std::string, FloatGridPtr> HatchingVoxelizer::voxelizeVectorsFromArrays(
    const std::vector<float>& x1,
    const std::vector<float>& y1,
    const std::vector<float>& z1,
    const std::vector<float>& x2,
    const std::vector<float>& y2,
    const std::vector<float>& z2,
    const std::vector<float>& power,
    const std::vector<float>& velocity,
    const std::vector<float>& energy,
    float voxel_size,
    const std::vector<float>& line_widths,
    float default_line_width,
    const std::vector<float>& hatch_spacings,
    const std::array<float, 3>& bbox_min,
    const std::array<float, 3>& bbox_max
) {
    if (x1.empty() || x1.size() != y1.size() || x1.size() != z1.size() ||
        x1.size() != x2.size() || x1.size() != y2.size() || x1.size() != z2.size()) {
        return {};
    }
    
    size_t n_vectors = x1.size();
    
    // Compute bounding box from vectors if not provided
    std::array<float, 3> actual_bbox_min = bbox_min;
    std::array<float, 3> actual_bbox_max = bbox_max;
    
    if (bbox_min[0] == 0.0f && bbox_min[1] == 0.0f && bbox_min[2] == 0.0f &&
        bbox_max[0] == 0.0f && bbox_max[1] == 0.0f && bbox_max[2] == 0.0f) {
        // Compute bounding box from all vector endpoints
        float min_x = x1[0], min_y = y1[0], min_z = z1[0];
        float max_x = x1[0], max_y = y1[0], max_z = z1[0];
        
        for (size_t i = 0; i < n_vectors; ++i) {
            min_x = std::min({min_x, x1[i], x2[i]});
            min_y = std::min({min_y, y1[i], y2[i]});
            min_z = std::min({min_z, z1[i], z2[i]});
            max_x = std::max({max_x, x1[i], x2[i]});
            max_y = std::max({max_y, y1[i], y2[i]});
            max_z = std::max({max_z, z1[i], z2[i]});
        }
        
        actual_bbox_min = {min_x, min_y, min_z};
        actual_bbox_max = {max_x, max_y, max_z};
    }
    
    // Create signal grids
    auto grids = createSignalGrids(voxel_size, actual_bbox_min, actual_bbox_max);
    FloatGridPtr power_grid = grids["power"];
    FloatGridPtr velocity_grid = grids["velocity"];
    FloatGridPtr energy_grid = grids["energy"];
    FloatGridPtr path_grid = grids["path"];
    
    // Create direction vector grid (same transform as signal grids for fusion alignment)
    direction_grid_ = Vec3fGrid::create(openvdb::Vec3f(0.0f, 0.0f, 0.0f));
    direction_grid_->setTransform(openvdb::math::Transform::Ptr(new openvdb::math::Transform(power_grid->transform())));
    direction_grid_->setName("direction");
    
    // Voxelize each vector independently
    for (size_t i = 0; i < n_vectors; ++i) {
        // Create HatchingPoint structs for start and end
        HatchingPoint p1, p2;
        p1.x = x1[i]; p1.y = y1[i]; p1.z = z1[i];
        p2.x = x2[i]; p2.y = y2[i]; p2.z = z2[i];
        
        // Get signals (use provided arrays or defaults)
        p1.power = (i < power.size()) ? power[i] : 200.0f;
        p1.velocity = (i < velocity.size()) ? velocity[i] : 500.0f;
        p1.energy = (i < energy.size()) ? energy[i] : 0.0f;
        
        p2.power = p1.power;
        p2.velocity = p1.velocity;
        p2.energy = p1.energy;
        
        // Use per-vector line_width if available, otherwise use default
        float vector_line_width = (i < line_widths.size() && line_widths[i] > 0.0f) 
                                   ? line_widths[i] 
                                   : default_line_width;
        
        // Get hatch_spacing for this vector (affects both path and physical signals)
        float hatch_spacing = 0.0f;
        if (i < hatch_spacings.size() && hatch_spacings[i] > 0.0f) {
            hatch_spacing = hatch_spacings[i];
        }
        
        // Calculate path_line_width based on hatch_spacing (for realistic path visualization)
        // If hatch_spacing >= voxel_size: use voxel_size (sharp edges, gaps visible)
        // If hatch_spacing < voxel_size: use hatch_spacing (paths may overlap in voxels)
        // If hatch_spacing not available (0.0): use voxel_size (sharp edges, default)
        float path_line_width = 0.0f;  // Default: sharp edges
        if (hatch_spacing > 0.0f) {
            if (hatch_spacing >= voxel_size) {
                // Hatch spacing >= voxel_size: use voxel_size for sharp edges (gaps will be visible)
                path_line_width = voxel_size;
            } else {
                // Hatch spacing < voxel_size: use hatch_spacing (paths may overlap in same voxel)
                path_line_width = hatch_spacing;
            }
        } else {
            // No hatch_spacing available: use voxel_size for sharp edges (default)
            path_line_width = voxel_size;
        }
        
        // Voxelize this single line segment with its specific beam width (including direction vectors)
        // hatch_spacing affects both path visualization and physical signals (power/energy)
        // path_line_width is calculated from hatch_spacing for realistic path visualization
        voxelizeLineSegment(p1, p2, power_grid, velocity_grid, energy_grid, path_grid, vector_line_width, hatch_spacing, path_line_width, direction_grid_);
    }
    
    return grids;
}

} // namespace voxelization
} // namespace am_qadf_native
