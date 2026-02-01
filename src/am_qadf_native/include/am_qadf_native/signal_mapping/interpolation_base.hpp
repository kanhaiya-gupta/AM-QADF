#ifndef AM_QADF_NATIVE_SIGNAL_MAPPING_INTERPOLATION_BASE_HPP
#define AM_QADF_NATIVE_SIGNAL_MAPPING_INTERPOLATION_BASE_HPP

#include <openvdb/openvdb.h>
#include <openvdb/Grid.h>
#include <vector>
#include <array>

namespace am_qadf_native {
namespace signal_mapping {

using FloatGrid = openvdb::FloatGrid;
using FloatGridPtr = FloatGrid::Ptr;

// Point structure
struct Point {
    float x, y, z;
    Point(float x = 0.0f, float y = 0.0f, float z = 0.0f) : x(x), y(y), z(z) {}
};

// Helper function: Convert numpy array to std::vector<Point> (C++ only, no Python loops)
// This is exposed in Python bindings for efficient conversion
std::vector<Point> numpy_to_points(
    const float* data,
    size_t num_points,
    const float* bbox_min = nullptr
);

// Base interpolation class
class InterpolationBase {
public:
    virtual ~InterpolationBase() = default;
    
    // Map points to grid
    virtual void map(
        FloatGridPtr grid,
        const std::vector<Point>& points,
        const std::vector<float>& values
    ) = 0;
    
protected:
    // Helper: Convert world coordinates to grid coordinates
    openvdb::Coord worldToGridCoord(FloatGridPtr grid, float x, float y, float z) const;
};

} // namespace signal_mapping
} // namespace am_qadf_native

#endif // AM_QADF_NATIVE_SIGNAL_MAPPING_INTERPOLATION_BASE_HPP
