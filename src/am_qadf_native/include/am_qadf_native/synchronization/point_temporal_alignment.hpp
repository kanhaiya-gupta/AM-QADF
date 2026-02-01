#ifndef AM_QADF_NATIVE_SYNCHRONIZATION_POINT_TEMPORAL_ALIGNMENT_HPP
#define AM_QADF_NATIVE_SYNCHRONIZATION_POINT_TEMPORAL_ALIGNMENT_HPP

#ifdef EIGEN_AVAILABLE
#include <Eigen/Dense>
#include <vector>
#include <set>
#include <cstddef>

#ifdef __has_include
#if __has_include(<pybind11/pybind11.h>)
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#define AM_QADF_POINT_TEMPORAL_ALIGNMENT_PYBIND11_AVAILABLE 1
#endif
#endif

namespace am_qadf_native {
namespace synchronization {

/**
 * Result of aligning multiple point sources by layer.
 * - unique_layers: sorted list of layer indices that appear in any source
 * - indices_per_layer_per_source: [layer_idx][source_idx] = row indices into that source's points
 *   that belong to that layer (caller slices points and signals using these indices)
 */
struct LayerAlignmentResult {
    std::vector<int> unique_layers;
    std::vector<std::vector<std::vector<int>>> indices_per_layer_per_source;
};

/**
 * Point temporal alignment: align point sets by layer (and optionally time) so that
 * "same layer" is well-defined across sources for per-layer fusion and mapping.
 * Operates on points (and optional signals) after spatial transformation.
 */
class PointTemporalAlignment {
public:
    PointTemporalAlignment() = default;

    /**
     * Align multiple sources by layer index.
     * For each layer that appears in any source, collect the point indices per source
     * that belong to that layer. Output enables "for each layer L, get points from
     * each source that belong to L" for fusion.
     *
     * @param points_per_source  Points matrix per source (N_s x 3)
     * @param layer_indices_per_source  Layer index per point per source; length must match rows of points
     * @return unique_layers (sorted) and indices_per_layer_per_source[l][s] = indices for layer l, source s
     */
    LayerAlignmentResult alignSourcesByLayer(
        const std::vector<Eigen::MatrixXd>& points_per_source,
        const std::vector<std::vector<int>>& layer_indices_per_source
    );

    /**
     * Zero-copy variant: layer indices as contiguous buffers (e.g. numpy int32 arrays).
     * Use this for large data to avoid copying billions of ints from Python.
     * @param layer_ptrs  Pointer to first element per source (length = num_sources)
     * @param layer_sizes  Number of elements per source (must match points_per_source[s].rows())
     */
    LayerAlignmentResult alignSourcesByLayerFromPointers(
        const std::vector<Eigen::MatrixXd>& points_per_source,
        const std::vector<const int*>& layer_ptrs,
        const std::vector<std::size_t>& layer_sizes
    );

    /**
     * Zero-copy, no points: align by layer using only layer index buffers.
     * No point data is read; use from Python with numpy int32 arrays only (no Eigen copy).
     * Single-pass grouping per source. Use for billions of points.
     * @param layer_ptrs  Pointer to first element per source (length = num_sources)
     * @param layer_sizes  Number of elements per source
     */
    LayerAlignmentResult alignSourcesByLayerFromLayerArraysOnly(
        const std::vector<const int*>& layer_ptrs,
        const std::vector<std::size_t>& layer_sizes
    );

    /**
     * Group a single source's points by layer (convenience).
     * @return unique_layers (sorted) and indices_per_layer[l] = row indices for that layer
     */
    void groupIndicesByLayer(
        const std::vector<int>& layer_indices,
        std::vector<int>& unique_layers_out,
        std::vector<std::vector<int>>& indices_per_layer_out
    ) const;
};

#ifdef AM_QADF_POINT_TEMPORAL_ALIGNMENT_PYBIND11_AVAILABLE
/** Align by layer from Python dicts; order/validation/conversion in C++. Returns (LayerAlignmentResult, source_order_used). */
pybind11::object align_points_by_layer_from_dicts(
    PointTemporalAlignment& self,
    const pybind11::dict& transformed_points,
    const pybind11::dict& layer_indices_per_source,
    pybind11::object source_order);
#endif

}  // namespace synchronization
}  // namespace am_qadf_native

#endif  // EIGEN_AVAILABLE
#endif  // AM_QADF_NATIVE_SYNCHRONIZATION_POINT_TEMPORAL_ALIGNMENT_HPP
