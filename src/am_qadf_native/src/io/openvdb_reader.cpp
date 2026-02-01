#include "am_qadf_native/io/openvdb_reader.hpp"
#include <openvdb/io/File.h>
#include <openvdb/tools/Statistics.h>
#include <openvdb/tools/GridOperators.h>
#include <openvdb/math/Coord.h>
#include <openvdb/math/Vec3.h>
#include <openvdb/math/Transform.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <random>
#include <numeric>
#include <cmath>
#include <limits>
#include <functional>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <algorithm>

namespace am_qadf_native {
namespace io {

FloatGridPtr OpenVDBReader::loadGrid(const std::string& vdb_file) {
    openvdb::io::File file(vdb_file);
    file.open();
    
    auto grids = file.getGrids();
    if (grids->empty()) {
        file.close();
        return FloatGrid::create();
    }
    
    // Get first FloatGrid - getGrids() returns GridPtrVec (vector of shared_ptr<GridBase>)
    auto grid = openvdb::gridPtrCast<FloatGrid>(grids->front());
    file.close();
    
    return grid;
}

pybind11::dict OpenVDBReader::extractFeatures(const std::string& vdb_file) {
    auto grid = loadGrid(vdb_file);
    
    float min_val, max_val, mean_val, std_val;
    computeStatistics(grid, min_val, max_val, mean_val, std_val);
    
    auto histogram = computeHistogram(grid);
    
    // Compute percentiles
    std::map<std::string, float> percentiles = computePercentiles(grid);
    
    // Count active voxels
    int voxel_count = static_cast<int>(grid->activeVoxelCount());
    
    pybind11::dict result;
    result["mean"] = mean_val;
    result["std"] = std_val;
    result["min"] = min_val;
    result["max"] = max_val;
    result["histogram"] = histogram;
    result["percentiles"] = percentiles;
    result["voxel_count"] = voxel_count;
    
    return result;
}

pybind11::array_t<float> OpenVDBReader::extractRegion(
    const std::string& vdb_file,
    const BoundingBox& bbox
) {
    auto grid = loadGrid(vdb_file);
    
    // Convert world coordinates to index coordinates
    auto min_coord = grid->transform().worldToIndexCellCentered(
        openvdb::Vec3R(bbox.x_min, bbox.y_min, bbox.z_min)
    );
    auto max_coord = grid->transform().worldToIndexCellCentered(
        openvdb::Vec3R(bbox.x_max, bbox.y_max, bbox.z_max)
    );
    
    int width = static_cast<int>(max_coord.x() - min_coord.x()) + 1;
    int height = static_cast<int>(max_coord.y() - min_coord.y()) + 1;
    int depth = static_cast<int>(max_coord.z() - min_coord.z()) + 1;
    
    // Create NumPy array
    auto result = pybind11::array_t<float>({depth, height, width});
    auto result_ptr = result.mutable_unchecked<3>();
    
    // Extract region data
    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                openvdb::Coord coord(
                    static_cast<int>(min_coord.x()) + x,
                    static_cast<int>(min_coord.y()) + y,
                    static_cast<int>(min_coord.z()) + z
                );
                
                float value = grid->tree().getValue(coord);
                result_ptr(z, y, x) = value;
            }
        }
    }
    
    return result;
}

pybind11::array_t<float> OpenVDBReader::extractSamples(
    const std::string& vdb_file,
    int n_samples,
    const std::string& strategy
) {
    auto grid = loadGrid(vdb_file);
    
    // Collect all active voxels
    std::vector<std::pair<openvdb::Coord, float>> active_voxels;
    for (auto iter = grid->beginValueOn(); iter; ++iter) {
        active_voxels.push_back(std::make_pair(iter.getCoord(), *iter));
    }
    
    if (active_voxels.empty()) {
        // Return empty 2D array with shape (0, 4)
        return pybind11::array_t<float>(std::vector<pybind11::ssize_t>{0, 4});
    }
    
    // Sample based on strategy
    std::vector<std::pair<openvdb::Coord, float>> samples;
    if (strategy == "random" && active_voxels.size() > static_cast<size_t>(n_samples)) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<size_t> dist(0, active_voxels.size() - 1);
        
        std::set<size_t> selected_indices;
        while (selected_indices.size() < static_cast<size_t>(n_samples)) {
            selected_indices.insert(dist(gen));
        }
        
        for (size_t idx : selected_indices) {
            samples.push_back(active_voxels[idx]);
        }
    } else {
        // Uniform sampling: take every Nth voxel
        size_t step = active_voxels.size() / n_samples;
        if (step == 0) step = 1;
        
        for (size_t i = 0; i < active_voxels.size() && samples.size() < static_cast<size_t>(n_samples); i += step) {
            samples.push_back(active_voxels[i]);
        }
    }
    
    // Convert to NumPy array
    std::vector<pybind11::ssize_t> shape_vec = {
        static_cast<pybind11::ssize_t>(samples.size()),
        static_cast<pybind11::ssize_t>(4)
    };
    pybind11::array_t<float> result(shape_vec);
    auto result_ptr = result.mutable_unchecked<2>();
    
    for (size_t i = 0; i < samples.size(); ++i) {
        auto world_coord = grid->transform().indexToWorld(samples[i].first);
        result_ptr(i, 0) = world_coord.x();
        result_ptr(i, 1) = world_coord.y();
        result_ptr(i, 2) = world_coord.z();
        result_ptr(i, 3) = samples[i].second;
    }
    
    return result;
}

pybind11::array_t<float> OpenVDBReader::extractFullGrid(const std::string& vdb_file) {
    auto grid = loadGrid(vdb_file);
    
    // Get bounding box of active voxels
    openvdb::CoordBBox bbox;
    if (!grid->tree().evalActiveVoxelBoundingBox(bbox)) {
        // No active voxels - return empty array
        return pybind11::array_t<float>({0, 0, 0});
    }
    
    // Compute dimensions
    int width = bbox.max().x() - bbox.min().x() + 1;
    int height = bbox.max().y() - bbox.min().y() + 1;
    int depth = bbox.max().z() - bbox.min().z() + 1;
    
    // Check if grid is too large (sanity check: > 1GB)
    size_t total_size = static_cast<size_t>(width) * height * depth * sizeof(float);
    if (total_size > 1024 * 1024 * 1024) {  // 1GB limit
        // Grid too large - return empty array
        return pybind11::array_t<float>({0, 0, 0});
    }
    
    // Create NumPy array
    auto result = pybind11::array_t<float>({depth, height, width});
    auto result_ptr = result.mutable_unchecked<3>();
    
    // Initialize with background value
    float background = grid->background();
    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                result_ptr(z, y, x) = background;
            }
        }
    }
    
    // Extract active voxels
    openvdb::Coord min_coord = bbox.min();
    for (auto iter = grid->beginValueOn(); iter; ++iter) {
        openvdb::Coord coord = iter.getCoord();
        int x = coord.x() - min_coord.x();
        int y = coord.y() - min_coord.y();
        int z = coord.z() - min_coord.z();
        
        if (x >= 0 && x < width && y >= 0 && y < height && z >= 0 && z < depth) {
            result_ptr(z, y, x) = *iter;
        }
    }
    
    return result;
}

void OpenVDBReader::processChunks(
    const std::string& vdb_file,
    std::function<void(pybind11::array_t<float>)> process_chunk,
    int chunk_size
) {
    auto grid = loadGrid(vdb_file);
    
    if (chunk_size <= 0) {
        chunk_size = 1000000;  // Default: 1M voxels per chunk
    }
    
    // Collect all active voxels
    std::vector<std::pair<openvdb::Coord, float>> active_voxels;
    for (auto iter = grid->beginValueOn(); iter; ++iter) {
        active_voxels.push_back(std::make_pair(iter.getCoord(), *iter));
    }
    
    if (active_voxels.empty()) {
        return;
    }
    
    // Process in chunks
    size_t num_chunks = (active_voxels.size() + chunk_size - 1) / chunk_size;
    
    for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
        size_t start_idx = chunk_idx * chunk_size;
        size_t end_idx = std::min(start_idx + chunk_size, active_voxels.size());
        
        // Create NumPy array for this chunk
        size_t chunk_length = end_idx - start_idx;
        std::vector<pybind11::ssize_t> shape_vec = {
            static_cast<pybind11::ssize_t>(chunk_length),
            static_cast<pybind11::ssize_t>(4)  // [x, y, z, value]
        };
        
        pybind11::array_t<float> chunk_array(shape_vec);
        auto chunk_ptr = chunk_array.mutable_unchecked<2>();
        
        // Fill chunk data
        for (size_t i = start_idx; i < end_idx; ++i) {
            const auto& voxel = active_voxels[i];
            auto world_coord = grid->transform().indexToWorld(voxel.first);
            
            size_t local_idx = i - start_idx;
            chunk_ptr(local_idx, 0) = world_coord.x();
            chunk_ptr(local_idx, 1) = world_coord.y();
            chunk_ptr(local_idx, 2) = world_coord.z();
            chunk_ptr(local_idx, 3) = voxel.second;
        }
        
        // Process chunk
        process_chunk(chunk_array);
    }
}

void OpenVDBReader::computeStatistics(
    FloatGridPtr grid,
    float& min_val,
    float& max_val,
    float& mean_val,
    float& std_val
) const {
    min_val = std::numeric_limits<float>::max();
    max_val = std::numeric_limits<float>::lowest();
    double sum = 0.0;
    size_t count = 0;
    
    for (auto iter = grid->beginValueOn(); iter; ++iter) {
        float val = *iter;
        min_val = std::min(min_val, val);
        max_val = std::max(max_val, val);
        sum += val;
        count++;
    }
    
    mean_val = count > 0 ? static_cast<float>(sum / count) : 0.0f;
    
    // Compute standard deviation
    double sum_sq_diff = 0.0;
    for (auto iter = grid->beginValueOn(); iter; ++iter) {
        float diff = *iter - mean_val;
        sum_sq_diff += diff * diff;
    }
    std_val = count > 0 ? static_cast<float>(std::sqrt(sum_sq_diff / count)) : 0.0f;
}

std::vector<int> OpenVDBReader::computeHistogram(
    FloatGridPtr grid,
    int num_bins
) const {
    // Compute histogram
    float min_val, max_val, mean_val, std_val;
    computeStatistics(grid, min_val, max_val, mean_val, std_val);
    
    if (max_val == min_val) {
        return std::vector<int>(num_bins, 0);
    }
    
    float bin_width = (max_val - min_val) / num_bins;
    std::vector<int> histogram(num_bins, 0);
    
    for (auto iter = grid->beginValueOn(); iter; ++iter) {
        float val = *iter;
        int bin = static_cast<int>((val - min_val) / bin_width);
        bin = std::max(0, std::min(bin, num_bins - 1));
        histogram[bin]++;
    }
    
    return histogram;
}

std::map<std::string, float> OpenVDBReader::computePercentiles(
    FloatGridPtr grid,
    const std::vector<float>& percentile_values
) const {
    std::map<std::string, float> percentiles;
    
    // Collect all active voxel values
    std::vector<float> values;
    values.reserve(grid->activeVoxelCount());
    
    for (auto iter = grid->beginValueOn(); iter; ++iter) {
        values.push_back(*iter);
    }
    
    if (values.empty()) {
        return percentiles;
    }
    
    // Sort values
    std::sort(values.begin(), values.end());
    
    // Compute requested percentiles
    for (float p : percentile_values) {
        if (p < 0.0f || p > 100.0f) {
            continue;  // Skip invalid percentiles
        }
        
        float index = (p / 100.0f) * (values.size() - 1);
        size_t lower_idx = static_cast<size_t>(std::floor(index));
        size_t upper_idx = static_cast<size_t>(std::ceil(index));
        
        float percentile_val;
        if (lower_idx == upper_idx) {
            percentile_val = values[lower_idx];
        } else {
            // Linear interpolation
            float weight = index - lower_idx;
            percentile_val = values[lower_idx] * (1.0f - weight) + values[upper_idx] * weight;
        }
        
        // Store as string key (e.g., "10", "50", "90")
        std::string key = std::to_string(static_cast<int>(p));
        percentiles[key] = percentile_val;
    }
    
    return percentiles;
}

FloatGridPtr OpenVDBReader::loadGridByName(const std::string& vdb_file, const std::string& grid_name) {
    openvdb::io::File file(vdb_file);
    file.open();
    
    auto grids = file.getGrids();
    if (!grids || grids->empty()) {
        file.close();
        return FloatGrid::create();
    }
    
    // Find grid by name
    for (auto& grid_base : *grids) {
        if (grid_base->getName() == grid_name) {
            // Try to cast to FloatGrid
            auto grid = openvdb::gridPtrCast<FloatGrid>(grid_base);
            file.close();
            return grid ? grid : FloatGrid::create();
        }
    }
    
    file.close();
    // Grid not found - return empty grid
    return FloatGrid::create();
}

std::map<std::string, FloatGridPtr> OpenVDBReader::loadAllGrids(const std::string& vdb_file) {
    std::map<std::string, FloatGridPtr> result;
    
    openvdb::io::File file(vdb_file);
    file.open();
    
    auto grids = file.getGrids();
    if (!grids || grids->empty()) {
        file.close();
        return result;
    }
    
    // Extract all FloatGrids by name
    for (auto& grid_base : *grids) {
        auto grid = openvdb::gridPtrCast<FloatGrid>(grid_base);
        if (grid) {
            std::string grid_name = grid_base->getName();
            if (grid_name.empty()) {
                // If no name, use default name
                grid_name = "grid_" + std::to_string(result.size());
            }
            result[grid_name] = grid;
        }
    }
    
    file.close();
    return result;
}

} // namespace io
} // namespace am_qadf_native
