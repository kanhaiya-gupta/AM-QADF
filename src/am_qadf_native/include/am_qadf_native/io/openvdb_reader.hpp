#ifndef AM_QADF_NATIVE_IO_OPENVDB_READER_HPP
#define AM_QADF_NATIVE_IO_OPENVDB_READER_HPP

#include <openvdb/openvdb.h>
#include <openvdb/io/File.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <string>
#include <memory>
#include <vector>
#include <array>
#include <map>

namespace am_qadf_native {
namespace io {

using FloatGrid = openvdb::FloatGrid;
using FloatGridPtr = FloatGrid::Ptr;

// Bounding box structure
struct BoundingBox {
    float x_min, y_min, z_min;
    float x_max, y_max, z_max;
};

// OpenVDB reader for analytics/anomaly detection
class OpenVDBReader {
public:
    // Extract features (statistics, histograms) - for analytics
    pybind11::dict extractFeatures(const std::string& vdb_file);
    
    // Extract specific region as NumPy array (zero-copy) - for spatial analysis
    pybind11::array_t<float> extractRegion(
        const std::string& vdb_file,
        const BoundingBox& bbox  // (x_min, y_min, z_min, x_max, y_max, z_max)
    );
    
    // Extract sparse samples (for anomaly detection)
    pybind11::array_t<float> extractSamples(
        const std::string& vdb_file,
        int n_samples,           // Number of samples to extract
        const std::string& strategy = "uniform"  // "uniform", "random", "stratified"
    );
    
    // Extract full grid to NumPy array (for small grids)
    pybind11::array_t<float> extractFullGrid(const std::string& vdb_file);
    
    // Stream chunks for very large data
    void processChunks(
        const std::string& vdb_file,
        std::function<void(pybind11::array_t<float>)> process_chunk,
        int chunk_size = 1000000
    );
    
    // Load grid by name from file (for multi-grid files)
    FloatGridPtr loadGridByName(const std::string& vdb_file, const std::string& grid_name);
    
    // Load all grids from file (returns map of grid_name -> FloatGridPtr)
    std::map<std::string, FloatGridPtr> loadAllGrids(const std::string& vdb_file);
    
private:
    // Helper: Load grid from file (first grid)
    FloatGridPtr loadGrid(const std::string& vdb_file);
    
    // Helper: Compute statistics
    void computeStatistics(FloatGridPtr grid, float& min_val, float& max_val,
                          float& mean_val, float& std_val) const;
    
    // Helper: Compute histogram
    std::vector<int> computeHistogram(FloatGridPtr grid, int num_bins = 100) const;
    
    // Helper: Compute percentiles
    std::map<std::string, float> computePercentiles(
        FloatGridPtr grid,
        const std::vector<float>& percentile_values = {10.0f, 50.0f, 90.0f}
    ) const;
};

} // namespace io
} // namespace am_qadf_native

#endif // AM_QADF_NATIVE_IO_OPENVDB_READER_HPP
