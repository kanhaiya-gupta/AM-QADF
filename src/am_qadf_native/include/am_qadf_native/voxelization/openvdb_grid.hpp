#ifndef AM_QADF_NATIVE_VOXELIZATION_OPENVDB_GRID_HPP
#define AM_QADF_NATIVE_VOXELIZATION_OPENVDB_GRID_HPP

#include <openvdb/openvdb.h>
#include <openvdb/Grid.h>
#include <openvdb/tools/MultiResGrid.h>
#include <pybind11/numpy.h>
#include <string>
#include <map>
#include <vector>
#include <memory>

namespace am_qadf_native {
namespace voxelization {

// Base grid type - FloatGrid for all signals
using FloatGrid = openvdb::FloatGrid;
using FloatGridPtr = FloatGrid::Ptr;

// Spatial region structure
struct SpatialRegion {
    float bbox_min[3];
    float bbox_max[3];
    float resolution;
};

// Temporal range structure
struct TemporalRange {
    float time_start;
    float time_end;
    int layer_start;
    int layer_end;
    float resolution;
};

// ============================================
// 1. UNIFORM GRID (Default)
// ============================================
class UniformVoxelGrid {
private:
    FloatGridPtr grid_;
    float voxel_size_;
    float bbox_min_[3];  // Bounding box offset for world coordinates
    
public:
    // Constructor with voxel size and optional bounding box offset
    UniformVoxelGrid(float voxel_size, float bbox_min_x = 0.0f, float bbox_min_y = 0.0f, float bbox_min_z = 0.0f);
    
    FloatGridPtr getGrid() const;
    void setSignalName(const std::string& name);
    
    // Add point data (world coordinates - OpenVDB handles transformation)
    void addPoint(float x, float y, float z, float value);
    
    // Add point at voxel indices (OpenVDB converts to world coordinates)
    void addPointAtVoxel(int i, int j, int k, float value);
    
    // Aggregate multiple values at a voxel (C++ - fast)
    void aggregateAtVoxel(int i, int j, int k, const std::vector<float>& values, const std::string& aggregation_mode);
    
    // Get value at voxel coordinates (i, j, k)
    float getValue(int i, int j, int k) const;
    
    // Get value at world coordinates (OpenVDB handles transformation)
    float getValueAtWorld(float x, float y, float z) const;
    
    // Convert voxel indices to world coordinates (using OpenVDB Transform)
    void voxelToWorld(int i, int j, int k, float& x, float& y, float& z) const;
    
    // Copy values from another FloatGrid (C++ - fast)
    void copyFromGrid(FloatGridPtr source_grid);
    
    // Populate from numpy array (C++ - fast, no Python loops)
    void populateFromArray(pybind11::array_t<float> array);
    
    // Get grid dimensions
    int getWidth() const;
    int getHeight() const;
    int getDepth() const;
    float getVoxelSize() const;
    
    // Get statistics (C++ - fast)
    struct Statistics {
        int total_voxels;
        int filled_voxels;
        float fill_ratio;
        float mean;
        float min;
        float max;
        float std;
    };
    Statistics getStatistics() const;
    
    // Save grid directly to VDB file (C++ - no NumPy conversion needed)
    void saveToFile(const std::string& filename) const;
};

// ============================================
// 2. MULTI-RESOLUTION GRID (Built-in Tool)
// ============================================
class MultiResolutionVoxelGrid {
private:
    // MultiResGrid requires the tree type as its template parameter
    // Use FloatTree directly (FloatGrid is Grid<FloatTree>)
    using MultiResGrid = openvdb::tools::MultiResGrid<openvdb::FloatTree>;
    MultiResGrid multi_grid_;
    std::vector<float> resolutions_;
    
public:
    MultiResolutionVoxelGrid(const std::vector<float>& resolutions, float base_resolution);
    
    // Get grid at specific level
    FloatGridPtr getGrid(int level) const;
    
    // Get resolution for level
    float getResolution(int level) const;
    
    // Prolongation (coarse → fine)
    void prolongate(int from_level, int to_level);
    
    // Restriction (fine → coarse)
    void restrict(int from_level, int to_level);
    
    // Get number of levels
    int getNumLevels() const;
    
    // Save all resolution levels to VDB file (C++ - no NumPy conversion needed)
    // Each level is saved with name "level_0", "level_1", etc.
    void saveToFile(const std::string& filename) const;
    
    // Save specific level to VDB file
    void saveLevelToFile(int level, const std::string& filename) const;
};

// ============================================
// 3. ADAPTIVE RESOLUTION GRID (Custom)
// ============================================
class AdaptiveResolutionVoxelGrid {
private:
    // Store multiple grids for different regions/resolutions
    std::map<std::string, FloatGridPtr> region_grids_;
    
    // Resolution maps
    std::vector<SpatialRegion> spatial_regions_;
    std::vector<TemporalRange> temporal_ranges_;
    
    float base_resolution_;
    
    // Generate key for region grid
    std::string makeRegionKey(const SpatialRegion& region) const;
    
public:
    AdaptiveResolutionVoxelGrid(float base_resolution);
    
    // Add spatial region with specific resolution
    void addSpatialRegion(float bbox_min[3], float bbox_max[3], float resolution);
    
    // Add temporal range with specific resolution
    void addTemporalRange(float time_start, float time_end,
                         int layer_start, int layer_end, float resolution);
    
    // Get resolution for point based on spatial/temporal maps
    float getResolutionForPoint(float x, float y, float z,
                                float timestamp = 0.0f, int layer = 0) const;
    
    // Get or create grid for specific region/resolution
    FloatGridPtr getOrCreateGrid(float x, float y, float z,
                                 float timestamp = 0.0f, int layer = 0);
    
    // Add point to appropriate grid
    void addPoint(float x, float y, float z, float value,
                  float timestamp = 0.0f, int layer = 0);
    
    // Get all grids (for fusion/export)
    std::vector<FloatGridPtr> getAllGrids() const;
    
    // Save all region grids to VDB file (C++ - no NumPy conversion needed)
    // Each region grid is saved with its region key as the name
    void saveToFile(const std::string& filename) const;
};

// ============================================
// 4. UNIFIED GRID MANAGER (Factory Pattern)
// ============================================
enum class GridType {
    UNIFORM,
    MULTI_RESOLUTION,
    ADAPTIVE
};

class VoxelGridFactory {
public:
    // Create uniform grid
    static std::unique_ptr<UniformVoxelGrid> createUniform(
        float voxel_size, const std::string& signal_name);
    
    // Create multi-resolution grid
    static std::unique_ptr<MultiResolutionVoxelGrid> createMultiResolution(
        const std::vector<float>& resolutions);
    
    // Create adaptive grid
    static std::unique_ptr<AdaptiveResolutionVoxelGrid> createAdaptive(
        float base_resolution);
};

} // namespace voxelization
} // namespace am_qadf_native

#endif // AM_QADF_NATIVE_VOXELIZATION_OPENVDB_GRID_HPP
