#include "am_qadf_native/voxelization/openvdb_grid.hpp"
#include "am_qadf_native/io/vdb_writer.hpp"
#include <openvdb/openvdb.h>
#include <openvdb/tools/MultiResGrid.h>
#include <openvdb/tools/GridOperators.h>
#include <openvdb/tools/GridTransformer.h>
#include <openvdb/math/Transform.h>
#include <openvdb/math/Coord.h>
#include <openvdb/math/Vec3.h>
#include <openvdb/math/BBox.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <limits>

namespace am_qadf_native {
namespace voxelization {

// ============================================
// UniformVoxelGrid Implementation
// ============================================

UniformVoxelGrid::UniformVoxelGrid(float voxel_size, float bbox_min_x, float bbox_min_y, float bbox_min_z) 
    : voxel_size_(voxel_size) {
    bbox_min_[0] = bbox_min_x;
    bbox_min_[1] = bbox_min_y;
    bbox_min_[2] = bbox_min_z;
    
    openvdb::initialize();
    grid_ = FloatGrid::create();
    
    // Create transform with voxel size and bbox offset
    // Transform: world = index * voxel_size + bbox_min
    // So: index = (world - bbox_min) / voxel_size
    auto transform = openvdb::math::Transform::createLinearTransform(voxel_size);
    // Pre-translate applies before scaling, so we translate by -bbox_min in index space
    // Then scale, then translate by bbox_min in world space
    // Actually, we want: world = index * voxel_size + bbox_min
    // So we use postTranslate to add bbox_min after scaling
    transform->postTranslate(openvdb::Vec3d(bbox_min_x, bbox_min_y, bbox_min_z));
    grid_->setTransform(transform);
}

FloatGridPtr UniformVoxelGrid::getGrid() const {
    return grid_;
}

void UniformVoxelGrid::setSignalName(const std::string& name) {
    grid_->setName(name);
}

void UniformVoxelGrid::addPoint(float x, float y, float z, float value) {
    // Node-centered so world (0.5,0.5,0.5) -> voxel (0,0,0)
    openvdb::Coord coord = grid_->transform().worldToIndexNodeCentered(
        openvdb::Vec3R(x, y, z)
    );
    grid_->tree().setValue(coord, value);
}

void UniformVoxelGrid::addPointAtVoxel(int i, int j, int k, float value) {
    // Add point at voxel indices - OpenVDB converts to world coordinates internally
    auto coord = openvdb::Coord(i, j, k);
    grid_->tree().setValue(coord, value);
}

void UniformVoxelGrid::aggregateAtVoxel(int i, int j, int k, const std::vector<float>& values, const std::string& aggregation_mode) {
    // Aggregate multiple values at a voxel (C++ - fast, no Python loops)
    if (values.empty()) {
        return;
    }
    
    float aggregated_value = 0.0f;
    
    if (aggregation_mode == "mean") {
        float sum = 0.0f;
        for (float v : values) {
            sum += v;
        }
        aggregated_value = sum / static_cast<float>(values.size());
    } else if (aggregation_mode == "max") {
        aggregated_value = *std::max_element(values.begin(), values.end());
    } else if (aggregation_mode == "min") {
        aggregated_value = *std::min_element(values.begin(), values.end());
    } else if (aggregation_mode == "sum") {
        for (float v : values) {
            aggregated_value += v;
        }
    } else if (aggregation_mode == "first") {
        aggregated_value = values[0];
    } else {
        // Default to mean
        float sum = 0.0f;
        for (float v : values) {
            sum += v;
        }
        aggregated_value = sum / static_cast<float>(values.size());
    }
    
    // Set aggregated value
    auto coord = openvdb::Coord(i, j, k);
    grid_->tree().setValue(coord, aggregated_value);
}

float UniformVoxelGrid::getValue(int i, int j, int k) const {
    // Get value at voxel coordinates (i, j, k)
    auto coord = openvdb::Coord(i, j, k);
    return grid_->tree().getValue(coord);
}

float UniformVoxelGrid::getValueAtWorld(float x, float y, float z) const {
    // Node-centered to match addPoint: world (0.5,0.5,0.5) -> voxel (0,0,0)
    openvdb::Coord coord = grid_->transform().worldToIndexNodeCentered(
        openvdb::Vec3R(x, y, z)
    );
    return grid_->tree().getValue(coord);
}

void UniformVoxelGrid::voxelToWorld(int i, int j, int k, float& x, float& y, float& z) const {
    // Convert voxel indices to world coordinates using OpenVDB Transform
    openvdb::Coord coord(i, j, k);
    openvdb::Vec3d world = grid_->transform().indexToWorld(coord);
    x = static_cast<float>(world.x());
    y = static_cast<float>(world.y());
    z = static_cast<float>(world.z());
}

void UniformVoxelGrid::copyFromGrid(FloatGridPtr source_grid) {
    // Copy all values from source grid to this grid (C++ - fast)
    if (!source_grid) {
        return;
    }
    
    // Copy all active voxels from source to target
    for (auto iter = source_grid->beginValueOn(); iter; ++iter) {
        openvdb::Coord coord = iter.getCoord();
        // Convert source grid's world coordinates to this grid's voxel coordinates
        openvdb::Vec3d world = source_grid->transform().indexToWorld(coord);
        openvdb::Coord target_coord = grid_->transform().worldToIndexCellCentered(world);
        grid_->tree().setValue(target_coord, *iter);
    }
}

void UniformVoxelGrid::populateFromArray(pybind11::array_t<float> array) {
    // Populate grid from numpy array (C++ - fast, no Python loops)
    if (array.ndim() != 3) {
        throw std::invalid_argument("Array must be 3D");
    }
    
    auto shape = array.shape();
    int depth = static_cast<int>(shape[0]);
    int height = static_cast<int>(shape[1]);
    int width = static_cast<int>(shape[2]);
    
    // Access array data directly in C++
    auto array_ptr = array.unchecked<3>();
    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float value = array_ptr(z, y, x);
                if (value != 0.0f) {  // Only store non-zero values (sparse)
                    // Convert array indices to world coordinates, then to voxel coordinates
                    // Array indices (z, y, x) correspond to voxel indices
                    // We need to account for bbox_min offset
                    float world_x = bbox_min_[0] + (x + 0.5f) * voxel_size_;
                    float world_y = bbox_min_[1] + (y + 0.5f) * voxel_size_;
                    float world_z = bbox_min_[2] + (z + 0.5f) * voxel_size_;
                    
                    // Use OpenVDB transform to convert world to voxel coordinates
                    openvdb::Coord coord = grid_->transform().worldToIndexCellCentered(
                        openvdb::Vec3R(world_x, world_y, world_z)
                    );
                    grid_->tree().setValue(coord, value);
                }
            }
        }
    }
}

int UniformVoxelGrid::getWidth() const {
    // Get bounding box of active voxels
    openvdb::CoordBBox bbox;
    if (grid_->tree().evalActiveVoxelBoundingBox(bbox)) {
        return bbox.max().x() - bbox.min().x() + 1;
    }
    return 0;
}

int UniformVoxelGrid::getHeight() const {
    openvdb::CoordBBox bbox;
    if (grid_->tree().evalActiveVoxelBoundingBox(bbox)) {
        return bbox.max().y() - bbox.min().y() + 1;
    }
    return 0;
}

int UniformVoxelGrid::getDepth() const {
    openvdb::CoordBBox bbox;
    if (grid_->tree().evalActiveVoxelBoundingBox(bbox)) {
        return bbox.max().z() - bbox.min().z() + 1;
    }
    return 0;
}

float UniformVoxelGrid::getVoxelSize() const {
    return voxel_size_;
}

UniformVoxelGrid::Statistics UniformVoxelGrid::getStatistics() const {
    // Compute statistics in C++ (fast, no Python loops)
    Statistics stats;
    
    // Get bounding box of active voxels
    openvdb::CoordBBox bbox;
    if (grid_->tree().evalActiveVoxelBoundingBox(bbox)) {
        int width = bbox.max().x() - bbox.min().x() + 1;
        int height = bbox.max().y() - bbox.min().y() + 1;
        int depth = bbox.max().z() - bbox.min().z() + 1;
        stats.total_voxels = width * height * depth;
    } else {
        stats.total_voxels = 0;
    }
    
    // Count filled voxels and compute statistics
    stats.filled_voxels = 0;
    float sum = 0.0f;
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();
    
    for (auto iter = grid_->beginValueOn(); iter; ++iter) {
        float value = *iter;
        if (value != 0.0f) {
            stats.filled_voxels++;
            sum += value;
            min_val = std::min(min_val, value);
            max_val = std::max(max_val, value);
        }
    }
    
    stats.fill_ratio = (stats.total_voxels > 0) ? 
        static_cast<float>(stats.filled_voxels) / static_cast<float>(stats.total_voxels) : 0.0f;
    
    if (stats.filled_voxels > 0) {
        stats.mean = sum / static_cast<float>(stats.filled_voxels);
        stats.min = (min_val == std::numeric_limits<float>::max()) ? 0.0f : min_val;
        stats.max = (max_val == std::numeric_limits<float>::lowest()) ? 0.0f : max_val;
        
        // Compute standard deviation
        float variance_sum = 0.0f;
        for (auto iter = grid_->beginValueOn(); iter; ++iter) {
            float value = *iter;
            if (value != 0.0f) {
                float diff = value - stats.mean;
                variance_sum += diff * diff;
            }
        }
        stats.std = std::sqrt(variance_sum / static_cast<float>(stats.filled_voxels));
    } else {
        stats.mean = 0.0f;
        stats.min = 0.0f;
        stats.max = 0.0f;
        stats.std = 0.0f;
    }
    
    return stats;
}

void UniformVoxelGrid::saveToFile(const std::string& filename) const {
    // Save grid directly to VDB file using OpenVDB's file I/O (C++ - fast, no NumPy conversion)
    am_qadf_native::io::VDBWriter writer;
    writer.write(grid_, filename);
}

// ============================================
// MultiResolutionVoxelGrid Implementation
// ============================================

MultiResolutionVoxelGrid::MultiResolutionVoxelGrid(
    const std::vector<float>& resolutions,
    float base_resolution
) : multi_grid_(resolutions.size(), 0.0f, base_resolution),
    resolutions_(resolutions) {
    // MultiResGrid is initialized with number of levels, default value, and base voxel size
    // Individual levels can be accessed via grid(level)
}

FloatGridPtr MultiResolutionVoxelGrid::getGrid(int level) const {
    // MultiResGrid::grid() returns ConstGridPtr for the grid at the specified level
    // We need to create a mutable copy
    auto const_grid = multi_grid_.grid(level);
    if (const_grid) {
        // Create a mutable copy of the grid
        // Note: This is expensive but necessary to return a mutable grid from a const method
        return openvdb::gridPtrCast<FloatGrid>(const_grid->deepCopy());
    }
    return nullptr;
}

float MultiResolutionVoxelGrid::getResolution(int level) const {
    if (level >= 0 && level < static_cast<int>(resolutions_.size())) {
        return resolutions_[level];
    }
    return 0.0f;
}

void MultiResolutionVoxelGrid::prolongate(int from_level, int to_level) {
    // Prolongation: Interpolate from coarse (from_level) to fine (to_level)
    // This is typically used in multi-grid methods to transfer data from coarse to fine grid
    
    if (from_level < 0 || to_level < 0 || 
        from_level >= static_cast<int>(resolutions_.size()) ||
        to_level >= static_cast<int>(resolutions_.size())) {
        return;  // Invalid levels
    }
    
    if (from_level >= to_level) {
        return;  // Prolongation only works from coarse to fine
    }
    
    // Get source grid (coarse)
    auto source_grid = getGrid(from_level);
    if (!source_grid) {
        return;
    }
    
    // Get target grid (fine)
    auto target_grid = getGrid(to_level);
    if (!target_grid) {
        return;
    }
    
    // Use OpenVDB's resampling to interpolate from coarse to fine
    // ResampleToMatch will interpolate values from source to target grid
    openvdb::tools::resampleToMatch<openvdb::tools::BoxSampler>(*source_grid, *target_grid);
}

void MultiResolutionVoxelGrid::restrict(int from_level, int to_level) {
    // Restriction: Downsample from fine (from_level) to coarse (to_level)
    // This is typically used in multi-grid methods to transfer data from fine to coarse grid
    
    if (from_level < 0 || to_level < 0 ||
        from_level >= static_cast<int>(resolutions_.size()) ||
        to_level >= static_cast<int>(resolutions_.size())) {
        return;  // Invalid levels
    }
    
    if (from_level <= to_level) {
        return;  // Restriction only works from fine to coarse
    }
    
    // Get source grid (fine)
    auto source_grid = getGrid(from_level);
    if (!source_grid) {
        return;
    }
    
    // Get target grid (coarse)
    auto target_grid = getGrid(to_level);
    if (!target_grid) {
        return;
    }
    
    // Use OpenVDB's resampling to downsample from fine to coarse
    // BoxSampler will average values when downsampling
    openvdb::tools::resampleToMatch<openvdb::tools::BoxSampler>(*source_grid, *target_grid);
}

int MultiResolutionVoxelGrid::getNumLevels() const {
    return static_cast<int>(resolutions_.size());
}

void MultiResolutionVoxelGrid::saveToFile(const std::string& filename) const {
    // Save all resolution levels to a single VDB file with named grids
    am_qadf_native::io::VDBWriter writer;
    
    std::vector<FloatGridPtr> grids;
    std::vector<std::string> grid_names;
    
    for (int level = 0; level < getNumLevels(); ++level) {
        auto grid = getGrid(level);
        if (grid) {
            grids.push_back(grid);
            grid_names.push_back("level_" + std::to_string(level));
        }
    }
    
    if (!grids.empty()) {
        writer.writeMultipleWithNames(grids, grid_names, filename);
    }
}

void MultiResolutionVoxelGrid::saveLevelToFile(int level, const std::string& filename) const {
    // Save a specific resolution level to VDB file
    auto grid = getGrid(level);
    if (grid) {
        am_qadf_native::io::VDBWriter writer;
        writer.write(grid, filename);
    }
}

// ============================================
// AdaptiveResolutionVoxelGrid Implementation
// ============================================

AdaptiveResolutionVoxelGrid::AdaptiveResolutionVoxelGrid(float base_resolution)
    : base_resolution_(base_resolution) {
}

void AdaptiveResolutionVoxelGrid::addSpatialRegion(
    float bbox_min[3], float bbox_max[3], float resolution
) {
    SpatialRegion region;
    std::copy(bbox_min, bbox_min + 3, region.bbox_min);
    std::copy(bbox_max, bbox_max + 3, region.bbox_max);
    region.resolution = resolution;
    spatial_regions_.push_back(region);
}

void AdaptiveResolutionVoxelGrid::addTemporalRange(
    float time_start, float time_end,
    int layer_start, int layer_end,
    float resolution
) {
    TemporalRange range;
    range.time_start = time_start;
    range.time_end = time_end;
    range.layer_start = layer_start;
    range.layer_end = layer_end;
    range.resolution = resolution;
    temporal_ranges_.push_back(range);
}

float AdaptiveResolutionVoxelGrid::getResolutionForPoint(
    float x, float y, float z,
    float timestamp,
    int layer
) const {
    float resolution = base_resolution_;
    
    // Check spatial regions
    for (const auto& region : spatial_regions_) {
        if (x >= region.bbox_min[0] && x <= region.bbox_max[0] &&
            y >= region.bbox_min[1] && y <= region.bbox_max[1] &&
            z >= region.bbox_min[2] && z <= region.bbox_max[2]) {
            resolution = std::min(resolution, region.resolution);
        }
    }
    
    // Check temporal ranges
    for (const auto& range : temporal_ranges_) {
        if (timestamp >= range.time_start && timestamp <= range.time_end &&
            layer >= range.layer_start && layer <= range.layer_end) {
            resolution = std::min(resolution, range.resolution);
        }
    }
    
    return resolution;
}

std::string AdaptiveResolutionVoxelGrid::makeRegionKey(const SpatialRegion& region) const {
    return std::to_string(region.bbox_min[0]) + "_" +
           std::to_string(region.bbox_max[0]) + "_" +
           std::to_string(region.resolution);
}

FloatGridPtr AdaptiveResolutionVoxelGrid::getOrCreateGrid(
    float x, float y, float z,
    float timestamp,
    int layer
) {
    float resolution = getResolutionForPoint(x, y, z, timestamp, layer);
    
    // Find or create spatial region for this resolution
    SpatialRegion region;
    region.bbox_min[0] = x - resolution * 10;
    region.bbox_max[0] = x + resolution * 10;
    region.bbox_min[1] = y - resolution * 10;
    region.bbox_max[1] = y + resolution * 10;
    region.bbox_min[2] = z - resolution * 10;
    region.bbox_max[2] = z + resolution * 10;
    region.resolution = resolution;
    
    std::string key = makeRegionKey(region);
    
    if (region_grids_.find(key) == region_grids_.end()) {
        // Create new grid for this region/resolution
        auto grid = FloatGrid::create();
        grid->setTransform(
            openvdb::math::Transform::createLinearTransform(resolution)
        );
        region_grids_[key] = grid;
    }
    
    return region_grids_[key];
}

void AdaptiveResolutionVoxelGrid::addPoint(
    float x, float y, float z, float value,
    float timestamp,
    int layer
) {
    auto grid = getOrCreateGrid(x, y, z, timestamp, layer);
    auto coord = grid->transform().worldToIndexCellCentered(
        openvdb::Vec3R(x, y, z)
    );
    grid->tree().setValue(coord, value);
}

std::vector<FloatGridPtr> AdaptiveResolutionVoxelGrid::getAllGrids() const {
    std::vector<FloatGridPtr> grids;
    for (const auto& pair : region_grids_) {
        grids.push_back(pair.second);
    }
    return grids;
}

void AdaptiveResolutionVoxelGrid::saveToFile(const std::string& filename) const {
    // Save all region grids to a single VDB file with named grids
    am_qadf_native::io::VDBWriter writer;
    
    std::vector<FloatGridPtr> grids;
    std::vector<std::string> grid_names;
    
    for (const auto& pair : region_grids_) {
        grids.push_back(pair.second);
        grid_names.push_back(pair.first);  // Use region key as name
    }
    
    if (!grids.empty()) {
        writer.writeMultipleWithNames(grids, grid_names, filename);
    }
}

// ============================================
// VoxelGridFactory Implementation
// ============================================

std::unique_ptr<UniformVoxelGrid> VoxelGridFactory::createUniform(
    float voxel_size, const std::string& signal_name
) {
    auto grid = std::make_unique<UniformVoxelGrid>(voxel_size);
    grid->setSignalName(signal_name);
    return grid;
}

std::unique_ptr<MultiResolutionVoxelGrid> VoxelGridFactory::createMultiResolution(
    const std::vector<float>& resolutions
) {
    if (resolutions.size() < 2u) {
        throw std::invalid_argument("MultiResGrid: at least two levels are required");
    }
    return std::make_unique<MultiResolutionVoxelGrid>(
        resolutions, resolutions.back()  // Base = finest
    );
}

std::unique_ptr<AdaptiveResolutionVoxelGrid> VoxelGridFactory::createAdaptive(
    float base_resolution
) {
    return std::make_unique<AdaptiveResolutionVoxelGrid>(base_resolution);
}

} // namespace voxelization
} // namespace am_qadf_native
