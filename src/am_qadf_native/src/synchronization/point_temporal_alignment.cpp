#ifdef EIGEN_AVAILABLE

#include "am_qadf_native/synchronization/point_temporal_alignment.hpp"
#include <algorithm>
#include <set>
#include <stdexcept>
#include <cstddef>
#include <unordered_map>

namespace am_qadf_native {
namespace synchronization {

LayerAlignmentResult PointTemporalAlignment::alignSourcesByLayer(
    const std::vector<Eigen::MatrixXd>& points_per_source,
    const std::vector<std::vector<int>>& layer_indices_per_source
) {
    LayerAlignmentResult result;

    const size_t num_sources = points_per_source.size();
    if (layer_indices_per_source.size() != num_sources) {
        throw std::invalid_argument(
            "alignSourcesByLayer: points_per_source and layer_indices_per_source must have same size"
        );
    }

    for (size_t s = 0; s < num_sources; ++s) {
        const int rows = static_cast<int>(points_per_source[s].rows());
        const size_t n = layer_indices_per_source[s].size();
        if (n != static_cast<size_t>(rows)) {
            throw std::invalid_argument(
                "alignSourcesByLayer: layer_indices size must match points rows for each source"
            );
        }
    }

    std::set<int> layers_set;
    for (const auto& layer_indices : layer_indices_per_source) {
        for (int layer : layer_indices) {
            layers_set.insert(layer);
        }
    }
    result.unique_layers.assign(layers_set.begin(), layers_set.end());
    std::sort(result.unique_layers.begin(), result.unique_layers.end());

    result.indices_per_layer_per_source.resize(result.unique_layers.size());
    for (size_t l = 0; l < result.unique_layers.size(); ++l) {
        const int layer = result.unique_layers[l];
        result.indices_per_layer_per_source[l].resize(num_sources);
        for (size_t s = 0; s < num_sources; ++s) {
            std::vector<int>& indices = result.indices_per_layer_per_source[l][s];
            const auto& layers = layer_indices_per_source[s];
            for (size_t i = 0; i < layers.size(); ++i) {
                if (layers[i] == layer) {
                    indices.push_back(static_cast<int>(i));
                }
            }
        }
    }

    return result;
}

LayerAlignmentResult PointTemporalAlignment::alignSourcesByLayerFromPointers(
    const std::vector<Eigen::MatrixXd>& points_per_source,
    const std::vector<const int*>& layer_ptrs,
    const std::vector<std::size_t>& layer_sizes
) {
    const size_t num_sources = points_per_source.size();
    if (layer_ptrs.size() != num_sources || layer_sizes.size() != num_sources) {
        throw std::invalid_argument(
            "alignSourcesByLayerFromPointers: points_per_source, layer_ptrs, layer_sizes must have same size"
        );
    }
    for (size_t s = 0; s < num_sources; ++s) {
        const std::size_t rows = static_cast<std::size_t>(points_per_source[s].rows());
        if (layer_sizes[s] != rows) {
            throw std::invalid_argument(
                "alignSourcesByLayerFromPointers: layer_sizes[s] must match points_per_source[s].rows()"
            );
        }
    }
    return alignSourcesByLayerFromLayerArraysOnly(layer_ptrs, layer_sizes);
}

LayerAlignmentResult PointTemporalAlignment::alignSourcesByLayerFromLayerArraysOnly(
    const std::vector<const int*>& layer_ptrs,
    const std::vector<std::size_t>& layer_sizes
) {
    LayerAlignmentResult result;
    const size_t num_sources = layer_ptrs.size();
    if (layer_sizes.size() != num_sources) {
        throw std::invalid_argument(
            "alignSourcesByLayerFromLayerArraysOnly: layer_ptrs and layer_sizes must have same size"
        );
    }

    // Single-pass per source: build layer_id -> indices for that source
    using LayerToIndices = std::unordered_map<int, std::vector<int>>;
    std::vector<LayerToIndices> per_source(num_sources);
    std::set<int> layers_set;

    for (size_t s = 0; s < num_sources; ++s) {
        const int* ptr = layer_ptrs[s];
        const std::size_t n = layer_sizes[s];
        LayerToIndices& map_s = per_source[s];
        for (std::size_t i = 0; i < n; ++i) {
            const int layer = ptr[i];
            map_s[layer].push_back(static_cast<int>(i));
            layers_set.insert(layer);
        }
    }

    result.unique_layers.assign(layers_set.begin(), layers_set.end());
    std::sort(result.unique_layers.begin(), result.unique_layers.end());

    result.indices_per_layer_per_source.resize(result.unique_layers.size());
    for (size_t l = 0; l < result.unique_layers.size(); ++l) {
        const int layer = result.unique_layers[l];
        result.indices_per_layer_per_source[l].resize(num_sources);
        for (size_t s = 0; s < num_sources; ++s) {
            auto it = per_source[s].find(layer);
            if (it != per_source[s].end()) {
                result.indices_per_layer_per_source[l][s] = it->second;
            }
            // else leave empty (default)
        }
    }
    return result;
}

void PointTemporalAlignment::groupIndicesByLayer(
    const std::vector<int>& layer_indices,
    std::vector<int>& unique_layers_out,
    std::vector<std::vector<int>>& indices_per_layer_out
) const {
    std::set<int> layers_set(layer_indices.begin(), layer_indices.end());
    unique_layers_out.assign(layers_set.begin(), layers_set.end());
    std::sort(unique_layers_out.begin(), unique_layers_out.end());

    indices_per_layer_out.resize(unique_layers_out.size());
    for (size_t l = 0; l < unique_layers_out.size(); ++l) {
        const int layer = unique_layers_out[l];
        std::vector<int>& indices = indices_per_layer_out[l];
        for (size_t i = 0; i < layer_indices.size(); ++i) {
            if (layer_indices[i] == layer) {
                indices.push_back(static_cast<int>(i));
            }
        }
    }
}

#ifdef AM_QADF_POINT_TEMPORAL_ALIGNMENT_PYBIND11_AVAILABLE
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

pybind11::object align_points_by_layer_from_dicts(
    PointTemporalAlignment& self,
    const pybind11::dict& transformed_points,
    const pybind11::dict& layer_indices_per_source,
    pybind11::object source_order)
{
    py::list order_list;
    if (!source_order.is_none() && py::len(source_order) > 0) {
        order_list = py::cast<py::list>(source_order);
    } else {
        for (auto& item : transformed_points)
            order_list.append(item.first);
    }

    std::vector<py::array_t<int32_t>> layer_arrays_held;
    std::vector<const int*> ptrs;
    std::vector<size_t> sizes;
    py::list order_used;

    for (py::ssize_t i = 0; i < py::len(order_list); ++i) {
        py::object st = order_list[i];
        py::object po = transformed_points.attr("get")(st);
        py::object lo = layer_indices_per_source.attr("get")(st);
        if (po.is_none() || lo.is_none())
            continue;

        py::array pts_arr = py::array::ensure(po);
        if (pts_arr.ndim() < 1)
            throw std::invalid_argument("transformed_points[source] must have at least one dimension");
        py::ssize_t n_pts = pts_arr.shape(0);

        py::array_t<int32_t> layer_arr = py::array_t<int32_t>::ensure(lo);
        if (static_cast<py::ssize_t>(layer_arr.size()) != n_pts)
            throw std::invalid_argument(
                "layer_indices for source must have length equal to points rows");

        layer_arrays_held.push_back(layer_arr);
        ptrs.push_back(layer_arr.data());
        sizes.push_back(static_cast<size_t>(layer_arr.size()));
        order_used.append(st);
    }

    if (ptrs.empty())
        throw std::invalid_argument("No valid source data for alignment");

    LayerAlignmentResult result = self.alignSourcesByLayerFromLayerArraysOnly(ptrs, sizes);
    return py::make_tuple(result, order_used);
}
#endif

}  // namespace synchronization
}  // namespace am_qadf_native

#endif  // EIGEN_AVAILABLE
