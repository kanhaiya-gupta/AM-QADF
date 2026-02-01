#ifdef EIGEN_AVAILABLE

#include "am_qadf_native/io/save_processed_points_bridge.hpp"
#include "am_qadf_native/io/mongodb_writer.hpp"
#include "am_qadf_native/io/mongocxx_instance.hpp"
#include "am_qadf_native/synchronization/point_bounds.hpp"
#include "am_qadf_native/synchronization/point_transformation_estimate.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstddef>

namespace py = pybind11;
namespace sync_ns = am_qadf_native::synchronization;

namespace am_qadf_native {
namespace io {

void save_transformed_points_to_mongodb(
    const std::string& model_id,
    const py::list& source_types,
    const py::dict& transformed_points,
    const py::dict& signals,
    const py::dict& layer_indices_per_source,
    py::object timestamps_per_source,
    const py::dict& transformations,
    const sync_ns::BoundingBox& unified_bounds,
    const std::string& mongo_uri,
    const std::string& db_name,
    int batch_size
) {
    (void)get_mongocxx_instance();
    MongoDBWriter writer(mongo_uri, db_name);

    const py::ssize_t n_sources = py::len(source_types);
    for (py::ssize_t s = 0; s < n_sources; ++s) {
        py::object st_obj = source_types[s];
        std::string st = st_obj.cast<std::string>();

        py::object tr_obj = transformations.attr("get")(st_obj);
        if (tr_obj.is_none())
            continue;
        py::dict tr = py::cast<py::dict>(tr_obj);
        py::object mat_obj = tr.attr("get")(py::str("matrix"));
        py::object qual_obj = tr.attr("get")(py::str("quality"));
        if (mat_obj.is_none() || qual_obj.is_none())
            continue;

        py::object pts_obj = transformed_points.attr("get")(st_obj);
        if (pts_obj.is_none())
            continue;
        py::array_t<double> pts = py::array_t<double>::ensure(pts_obj);
        if (pts.ndim() < 1)
            continue;
        py::ssize_t n_pts = pts.shape(0);
        if (n_pts == 0)
            continue;
        pts = py::array_t<double>::ensure(pts.reshape(std::vector<py::ssize_t>{n_pts, pts.size() / n_pts}));
        if (pts.shape(1) < 3)
            throw std::invalid_argument("transformed_points must have at least 3 columns");
        std::vector<double> pts_3_copy;
        const double* points_xyz;
        if (pts.shape(1) == 3) {
            points_xyz = pts.data();
        } else {
            pts_3_copy.resize(static_cast<std::size_t>(n_pts) * 3);
            for (py::ssize_t i = 0; i < n_pts; ++i) {
                pts_3_copy[static_cast<std::size_t>(i) * 3 + 0] = pts.data()[i * pts.shape(1) + 0];
                pts_3_copy[static_cast<std::size_t>(i) * 3 + 1] = pts.data()[i * pts.shape(1) + 1];
                pts_3_copy[static_cast<std::size_t>(i) * 3 + 2] = pts.data()[i * pts.shape(1) + 2];
            }
            points_xyz = pts_3_copy.data();
        }

        py::object sigs_obj = signals.attr("get")(st_obj);
        py::array_t<double> sig_arr;
        if (sigs_obj.is_none() || py::len(sigs_obj) == 0) {
            sig_arr = py::array_t<double>({n_pts});
            std::fill(sig_arr.mutable_data(), sig_arr.mutable_data() + n_pts, 0.0);
        } else {
            py::dict sigs = py::cast<py::dict>(sigs_obj);
            py::object first_val;
            for (auto item : sigs) {
                first_val = py::reinterpret_borrow<py::object>(item.second);
                break;
            }
            if (first_val.is_none()) {
                sig_arr = py::array_t<double>({n_pts});
                std::fill(sig_arr.mutable_data(), sig_arr.mutable_data() + n_pts, 0.0);
            } else {
                sig_arr = py::array_t<double>::ensure(first_val);
                sig_arr = py::array_t<double>::ensure(sig_arr.reshape(std::vector<py::ssize_t>{sig_arr.size()}));
                if (sig_arr.size() != static_cast<py::ssize_t>(n_pts)) {
                    double fill = sig_arr.size() > 0 ? sig_arr.data()[0] : 0.0;
                    py::array_t<double> padded(n_pts);
                    std::fill(padded.mutable_data(), padded.mutable_data() + n_pts, fill);
                    sig_arr = padded;
                }
            }
        }
        const double* signal = sig_arr.data();

        py::object layers_obj = layer_indices_per_source.attr("get")(st_obj);
        if (layers_obj.is_none())
            continue;
        py::array_t<std::int32_t> layer_arr = py::array_t<std::int32_t>::ensure(layers_obj);
        layer_arr = py::array_t<std::int32_t>::ensure(layer_arr.reshape(std::vector<py::ssize_t>{layer_arr.size()}));
        const std::size_t layer_len = static_cast<std::size_t>(layer_arr.size());

        std::vector<int> padded_layers;
        const int* layer_indices_ptr;
        if (layer_len == static_cast<std::size_t>(n_pts)) {
            layer_indices_ptr = layer_arr.data();
        } else {
            padded_layers.resize(static_cast<std::size_t>(n_pts));
            const int32_t* src = layer_arr.data();
            for (py::ssize_t i = 0; i < n_pts; ++i)
                padded_layers[static_cast<std::size_t>(i)] = layer_len > 0 ? static_cast<int>(src[i % layer_len]) : 0;
            layer_indices_ptr = padded_layers.data();
        }

        const char* timestamps_iso = nullptr;
        std::size_t timestamp_stride = 0;
        std::vector<char> timestamps_buf;
        if (!timestamps_per_source.is_none()) {
            py::object ts_obj = timestamps_per_source.attr("get")(st_obj);
            if (!ts_obj.is_none()) {
                if (py::isinstance<py::array>(ts_obj)) {
                    py::array ts_arr = py::cast<py::array>(ts_obj);
                    if (ts_arr.ndim() == 1 && ts_arr.dtype().kind() == 'S' && ts_arr.dtype().itemsize() > 0) {
                        timestamps_iso = reinterpret_cast<const char*>(ts_arr.data());
                        timestamp_stride = static_cast<std::size_t>(ts_arr.dtype().itemsize());
                    }
                }
                if (timestamps_iso == nullptr && py::isinstance<py::list>(ts_obj)) {
                    py::list ts_list = py::cast<py::list>(ts_obj);
                    const std::size_t ts_len = static_cast<std::size_t>(py::len(ts_list));
                    constexpr std::size_t kStampLen = 28;
                    timestamps_buf.resize(static_cast<std::size_t>(n_pts) * kStampLen, ' ');
                    for (py::ssize_t i = 0; i < n_pts; ++i) {
                        std::string t = ts_len > 0 ? py::cast<std::string>(ts_list[i % static_cast<py::ssize_t>(ts_len)]) : "";
                        if (t.size() > kStampLen) t.resize(kStampLen);
                        std::copy(t.begin(), t.end(), timestamps_buf.begin() + static_cast<std::size_t>(i) * kStampLen);
                    }
                    timestamps_iso = timestamps_buf.data();
                    timestamp_stride = kStampLen;
                }
            }
        }

        py::array_t<double> mat_arr = py::array_t<double>::ensure(mat_obj);
        if (mat_arr.size() < 16)
            throw std::invalid_argument("transformation matrix must have at least 16 elements");
        mat_arr = py::array_t<double>::ensure(mat_arr.reshape(std::vector<py::ssize_t>{4, 4}));
        Eigen::Matrix4d mat;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                mat(i, j) = mat_arr.data()[static_cast<py::ssize_t>(i) * 4 + j];
        sync_ns::TransformationQuality qual = qual_obj.cast<sync_ns::TransformationQuality>();

        writer.saveProcessedPointsFromBuffers(
            model_id, st,
            points_xyz, signal, layer_indices_ptr, static_cast<std::size_t>(n_pts),
            timestamps_iso, timestamp_stride,
            unified_bounds, mat, qual,
            "", false, batch_size
        );
    }
}

}  // namespace io
}  // namespace am_qadf_native

#endif  // EIGEN_AVAILABLE
