#ifndef AM_QADF_NATIVE_VOXELIZATION_HATCHING_VOXELIZER_HPP
#define AM_QADF_NATIVE_VOXELIZATION_HATCHING_VOXELIZER_HPP

#include <openvdb/openvdb.h>
#include <openvdb/Grid.h>
#include <openvdb/tools/LevelSetUtil.h>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <array>

namespace am_qadf_native {
namespace voxelization {

using FloatGrid = openvdb::FloatGrid;
using FloatGridPtr = FloatGrid::Ptr;
using Vec3fGrid = openvdb::Vec3fGrid;
using Vec3fGridPtr = Vec3fGrid::Ptr;

// Hatching path point with process parameters (signals)
// These are geometric points along hatching/contour paths
struct HatchingPoint {
    float x, y, z;      // 3D position (geometry)
    float power;        // Signal: laser power (W)
    float velocity;     // Signal: scan velocity (mm/s)
    float energy;       // Signal: energy density (J/mm²)
};

// Hatching vector (line segment) with process parameters
// Each vector is a separate line segment: (x1, y1, z1) -> (x2, y2, z2)
// This matches the pyslm structure where hatches are stored as pairs
struct HatchingVector {
    float x1, y1, z1;   // Start point
    float x2, y2, z2;   // End point
    float power;        // Signal: laser power (W)
    float velocity;     // Signal: scan velocity (mm/s)
    float energy;       // Signal: energy density (J/mm²)
};

// Hatching Voxelizer - converts geometric hatching/contour paths to OpenVDB grids
// Hatching paths are 3D line segments (geometry) with process parameters (signals)
// This voxelizes the geometry into grids and fills voxels with signal values
class HatchingVoxelizer {
public:
    // Voxelize hatching paths (geometric line segments) to OpenVDB grids
    // Points form connected line segments: p0->p1->p2->...
    // Each point has associated signals (power, velocity, energy)
    // Returns map of signal names to grids: {"power": grid, "velocity": grid, "energy": grid}
    // Also returns direction vectors as Vec3fGrid for arrow visualization in ParaView
    std::map<std::string, FloatGridPtr> voxelizeHatchingPaths(
        const std::vector<HatchingPoint>& points,
        float voxel_size,
        float line_width = 0.1f,  // Width/radius of hatching line in mm (laser beam width)
        const std::array<float, 3>& bbox_min = {0.0f, 0.0f, 0.0f},
        const std::array<float, 3>& bbox_max = {0.0f, 0.0f, 0.0f}
    );
    
    // Voxelize contour paths (closed loops, similar to hatching)
    std::map<std::string, FloatGridPtr> voxelizeContourPaths(
        const std::vector<HatchingPoint>& points,
        float voxel_size,
        float line_width = 0.1f,
        const std::array<float, 3>& bbox_min = {0.0f, 0.0f, 0.0f},
        const std::array<float, 3>& bbox_max = {0.0f, 0.0f, 0.0f}
    );
    
    // Voxelize multiple layers of hatching paths
    // layers_points: vector of point arrays, one per layer
    std::map<std::string, FloatGridPtr> voxelizeMultiLayerHatching(
        const std::vector<std::vector<HatchingPoint>>& layers_points,
        float voxel_size,
        float line_width = 0.1f,
        const std::array<float, 3>& bbox_min = {0.0f, 0.0f, 0.0f},
        const std::array<float, 3>& bbox_max = {0.0f, 0.0f, 0.0f}
    );
    
    // Voxelize hatching vectors (NEW: vector-based format)
    // Each vector is a separate line segment (x1,y1,z1) -> (x2,y2,z2)
    // This matches the pyslm structure and MongoDB vector format
    // Vectors are NOT connected to each other - each is independent
    // Returns scalar grids: {"power": grid, "velocity": grid, "energy": grid}
    // Also creates direction vector grid for arrow visualization in ParaView
    std::map<std::string, FloatGridPtr> voxelizeVectors(
        const std::vector<HatchingVector>& vectors,
        float voxel_size,
        float line_width = 0.1f,
        const std::array<float, 3>& bbox_min = {0.0f, 0.0f, 0.0f},
        const std::array<float, 3>& bbox_max = {0.0f, 0.0f, 0.0f}
    );
    
    // Get direction vector grid (for arrow visualization in ParaView)
    // Returns Vec3fGrid with normalized direction vectors (dx, dy, dz) for each voxel
    Vec3fGridPtr getDirectionGrid() const { return direction_grid_; }
    
    // Voxelize vectors from raw arrays (for Python interface - avoids Python-side conversion)
    // Arrays must have same length. Handles case-insensitive field names internally.
    // x1, y1, z1, x2, y2, z2: coordinates for each vector
    // power, velocity, energy: signals for each vector (optional, can be empty)
    // line_widths: per-vector line widths (optional, if empty uses default_line_width for all)
    // hatch_spacings: per-vector hatch spacing (for path signal visibility, 0.0 = use voxel_size for sharp edges)
    // Also creates direction vector grid for arrow visualization in ParaView
    std::map<std::string, FloatGridPtr> voxelizeVectorsFromArrays(
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
        float default_line_width = 0.1f,
        const std::vector<float>& hatch_spacings = std::vector<float>(),
        const std::array<float, 3>& bbox_min = {0.0f, 0.0f, 0.0f},
        const std::array<float, 3>& bbox_max = {0.0f, 0.0f, 0.0f}
    );
    
private:
    // Helper: Create empty grids for all signals
    std::map<std::string, FloatGridPtr> createSignalGrids(
        float voxel_size,
        const std::array<float, 3>& bbox_min,
        const std::array<float, 3>& bbox_max
    );
    
    // Helper: Voxelize a line segment (geometry) with signal values
    // Interpolates signals along the line segment
    // Also stores direction vectors for arrow visualization
    // line_width: beam width for physical signals (power, velocity, energy)
    // hatch_spacing: distance between paths (affects effective expansion for realistic gaps)
    // path_line_width: width for path signal (0.0 = sharp edges, only line voxels; >0 = blurred edges)
    void voxelizeLineSegment(
        const HatchingPoint& p1,
        const HatchingPoint& p2,
        FloatGridPtr power_grid,
        FloatGridPtr velocity_grid,
        FloatGridPtr energy_grid,
        FloatGridPtr path_grid,  // Path signal: constant value 1.0
        float line_width,         // Beam width for physical signals
        float hatch_spacing = 0.0f,  // Hatch spacing (0.0 = not used; >0 = limits expansion to respect gaps)
        float path_line_width = 0.0f,  // Path width (0.0 = sharp, uses voxel_size; >0 = blurred)
        Vec3fGridPtr direction_grid = nullptr  // Optional: store direction vectors
    );
    
    // Helper: Set signal value in grid along line segment with width
    // Uses Bresenham-like algorithm to find all voxels along the line
    void setValueAlongLine(
        FloatGridPtr grid,
        const std::array<float, 3>& start,
        const std::array<float, 3>& end,
        float start_value,
        float end_value,
        float line_width
    );
    
    // Helper: Compute bounding box from points if not provided
    void computeBoundingBox(
        const std::vector<HatchingPoint>& points,
        std::array<float, 3>& bbox_min,
        std::array<float, 3>& bbox_max
    );
    
    // Helper: Get all voxel coordinates along a line segment
    std::vector<openvdb::Coord> getVoxelsAlongLine(
        const openvdb::math::Transform& transform,
        const std::array<float, 3>& start,
        const std::array<float, 3>& end,
        float line_width
    );
    
    // Helper: Set direction vectors along a line segment
    void setDirectionAlongLine(
        Vec3fGridPtr direction_grid,
        const std::array<float, 3>& start,
        const std::array<float, 3>& end,
        const std::array<float, 3>& direction,
        float line_width
    );
    
    // Member variable to store direction grid
    mutable Vec3fGridPtr direction_grid_;
};

} // namespace voxelization
} // namespace am_qadf_native

#endif // AM_QADF_NATIVE_VOXELIZATION_HATCHING_VOXELIZER_HPP
