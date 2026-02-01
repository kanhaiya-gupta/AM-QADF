#ifdef EIGEN_AVAILABLE

#include "am_qadf_native/io/mongodb_writer.hpp"
#include "am_qadf_native/io/mongocxx_instance.hpp"
#include "am_qadf_native/synchronization/point_bounds.hpp"
#include "am_qadf_native/synchronization/point_transformation_estimate.hpp"
#include <mongocxx/uri.hpp>
#include <mongocxx/options/bulk_write.hpp>
#include <mongocxx/model/replace_one.hpp>
#include <bsoncxx/builder/stream/document.hpp>
#include <bsoncxx/builder/stream/array.hpp>
#include <bsoncxx/types.hpp>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <cmath>  // For std::abs (for coordinate comparison tolerance)

namespace {

// Thread-safe current timestamp string (avoids std::localtime in multithreaded environments e.g. Jupyter).
std::string currentTimestampString(const char* format) {
    auto now = std::chrono::system_clock::now();
    std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf{};
#if defined(_WIN32) || defined(_WIN64)
    (void)localtime_s(&tm_buf, &tt);
#else
    (void)localtime_r(&tt, &tm_buf);
#endif
    std::stringstream ss;
    ss << std::put_time(&tm_buf, format);
    return ss.str();
}

}  // namespace

namespace am_qadf_native {
namespace io {

using namespace synchronization;

MongoDBWriter::MongoDBWriter(const std::string& uri, const std::string& db_name) {
    (void)get_mongocxx_instance();
    mongocxx::uri mongo_uri(uri);
    client_ = mongocxx::client(mongo_uri);
    db_ = client_[db_name];
}

// Helper: Get collection name with "Processed_" prefix
std::string MongoDBWriter::getProcessedCollectionName(const std::string& source_type) {
    return "Processed_" + source_type + "_data";
}

// Helper: Convert Eigen::MatrixXd to BSON array
bsoncxx::builder::stream::array MongoDBWriter::pointsToBSONArray(const Eigen::MatrixXd& points) {
    bsoncxx::builder::stream::array array_builder;
    
    for (int i = 0; i < points.rows(); ++i) {
        array_builder << bsoncxx::builder::stream::open_array
                     << static_cast<double>(points(i, 0))
                     << static_cast<double>(points(i, 1))
                     << static_cast<double>(points(i, 2))
                     << bsoncxx::builder::stream::close_array;
    }
    
    return array_builder;
}

// Helper: Convert Eigen::VectorXd to BSON array
bsoncxx::builder::stream::array MongoDBWriter::vectorToBSONArray(const Eigen::VectorXd& vec) {
    bsoncxx::builder::stream::array array_builder;
    
    for (int i = 0; i < vec.size(); ++i) {
        array_builder << static_cast<double>(vec(i));
    }
    
    return array_builder;
}

// Helper: Convert Eigen::Matrix4d to BSON array (4x4 matrix as nested arrays)
bsoncxx::builder::stream::array MongoDBWriter::matrix4dToBSONArray(const Eigen::Matrix4d& matrix) {
    bsoncxx::builder::stream::array array_builder;
    
    for (int i = 0; i < 4; ++i) {
        array_builder << bsoncxx::builder::stream::open_array;
        for (int j = 0; j < 4; ++j) {
            array_builder << static_cast<double>(matrix(i, j));
        }
        array_builder << bsoncxx::builder::stream::close_array;
    }
    
    return array_builder;
}

// Delete existing processed data
int MongoDBWriter::deleteProcessedData(
    const std::string& model_id,
    const std::string& source_type,
    const std::string& processing_run_id
) {
    std::string collection_name = getProcessedCollectionName(source_type);
    auto collection = db_[collection_name];
    
    // Build filter query
    bsoncxx::builder::stream::document filter_builder;
    filter_builder << "model_id" << model_id;
    
    if (!processing_run_id.empty()) {
        filter_builder << "processing_run_id" << processing_run_id;
    }
    
    auto filter = filter_builder.extract();
    
    // Delete matching documents
    auto result = collection.delete_many(filter.view());
    if (result) {
        return static_cast<int>(result->deleted_count());
    }
    return 0;
}

// Save processed points (one document per point, batched)
void MongoDBWriter::saveProcessedPoints(
    const std::string& model_id,
    const std::string& source_type,
    const Eigen::MatrixXd& transformed_points,
    const Eigen::VectorXd& signal_values,
    const std::vector<int>& layer_indices,
    const std::vector<std::string>& timestamps,
    const BoundingBox& unified_bounds,
    const Eigen::Matrix4d& transformation_matrix,
    const TransformationQuality& quality_metrics,
    const std::string& processing_run_id,
    bool delete_existing,
    int batch_size
) {
    if (transformed_points.rows() != signal_values.size()) {
        throw std::invalid_argument("Number of points must match number of signal values");
    }
    
    if (transformed_points.rows() != static_cast<int>(layer_indices.size())) {
        throw std::invalid_argument("Number of points must match number of layer indices");
    }
    
    if (!timestamps.empty() && timestamps.size() != static_cast<size_t>(transformed_points.rows())) {
        throw std::invalid_argument("Number of timestamps must match number of points (or be empty)");
    }
    
    if (transformed_points.cols() != 3) {
        throw std::invalid_argument("Points matrix must have 3 columns (x, y, z)");
    }
    
    // Delete existing data if requested (prevents duplicates)
    if (delete_existing) {
        int deleted = deleteProcessedData(model_id, source_type, processing_run_id);
        // Note: deleted count could be 0 if no existing data, which is fine
    }
    
    // Compute bounds of the transformed points (source-specific bounds)
    BoundingBox source_bounds;
    if (transformed_points.rows() > 0) {
        source_bounds.min_x = source_bounds.max_x = transformed_points(0, 0);
        source_bounds.min_y = source_bounds.max_y = transformed_points(0, 1);
        source_bounds.min_z = source_bounds.max_z = transformed_points(0, 2);
        for (int i = 1; i < transformed_points.rows(); ++i) {
            source_bounds.min_x = std::min(source_bounds.min_x, transformed_points(i, 0));
            source_bounds.max_x = std::max(source_bounds.max_x, transformed_points(i, 0));
            source_bounds.min_y = std::min(source_bounds.min_y, transformed_points(i, 1));
            source_bounds.max_y = std::max(source_bounds.max_y, transformed_points(i, 1));
            source_bounds.min_z = std::min(source_bounds.min_z, transformed_points(i, 2));
            source_bounds.max_z = std::max(source_bounds.max_z, transformed_points(i, 2));
        }
    }
    
    // Get collection with "Processed_" prefix
    std::string collection_name = getProcessedCollectionName(source_type);
    auto collection = db_[collection_name];
    
    std::string default_timestamp = currentTimestampString("%Y-%m-%dT%H:%M:%S");
    
    int total_points = transformed_points.rows();
    int total_inserted = 0;
    
    // Process in batches
    for (int batch_start = 0; batch_start < total_points; batch_start += batch_size) {
        int batch_end = std::min(batch_start + batch_size, total_points);
        std::vector<bsoncxx::document::value> batch_docs;
        batch_docs.reserve(batch_end - batch_start);
        
        for (int i = batch_start; i < batch_end; ++i) {
            // Build one document per point (matching raw data structure)
            bsoncxx::builder::stream::document doc_builder;
            doc_builder << "model_id" << model_id
                        << "layer_index" << layer_indices[i]
                        << "timestamp" << (timestamps.empty() ? default_timestamp : timestamps[i])
                        << "spatial_coordinates" << bsoncxx::builder::stream::open_array
                            << static_cast<double>(transformed_points(i, 0))
                            << static_cast<double>(transformed_points(i, 1))
                            << static_cast<double>(transformed_points(i, 2))
                        << bsoncxx::builder::stream::close_array
                        << "signal_value" << static_cast<double>(signal_values(i))
                        << "coordinate_system" << "unified";  // All processed data is in unified coordinate system
            
            // Add processing_run_id if provided (helps track different processing runs)
            if (!processing_run_id.empty()) {
                doc_builder << "processing_run_id" << processing_run_id;
            }
            
            // Add transformation metadata (stored once per batch, or can be stored separately)
            // For efficiency, we store it in the first document of each batch
            if (i == batch_start) {
                doc_builder << "transformation_metadata" << bsoncxx::builder::stream::open_document
                    << "transformation_matrix" << matrix4dToBSONArray(transformation_matrix)
                    << "quality_metrics" << bsoncxx::builder::stream::open_document
                        << "rms_error" << quality_metrics.rms_error
                        << "alignment_quality" << quality_metrics.alignment_quality
                        << "confidence" << quality_metrics.confidence
                        << "max_error" << quality_metrics.max_error
                        << "mean_error" << quality_metrics.mean_error
                    << bsoncxx::builder::stream::close_document
                    << "source_bounds" << bsoncxx::builder::stream::open_document
                        << "min_x" << source_bounds.min_x
                        << "min_y" << source_bounds.min_y
                        << "min_z" << source_bounds.min_z
                        << "max_x" << source_bounds.max_x
                        << "max_y" << source_bounds.max_y
                        << "max_z" << source_bounds.max_z
                        << "width" << source_bounds.width()
                        << "height" << source_bounds.height()
                        << "depth" << source_bounds.depth()
                    << bsoncxx::builder::stream::close_document
                    << "unified_bounds" << bsoncxx::builder::stream::open_document
                        << "min_x" << unified_bounds.min_x
                        << "min_y" << unified_bounds.min_y
                        << "min_z" << unified_bounds.min_z
                        << "max_x" << unified_bounds.max_x
                        << "max_y" << unified_bounds.max_y
                        << "max_z" << unified_bounds.max_z
                        << "width" << unified_bounds.width()
                        << "height" << unified_bounds.height()
                        << "depth" << unified_bounds.depth()
                    << bsoncxx::builder::stream::close_document
                << bsoncxx::builder::stream::close_document;
            }
            
            batch_docs.push_back(doc_builder.extract());
        }
        
        // Upsert batch (overwrites existing documents with same model_id + layer_index + spatial_coordinates)
        if (!batch_docs.empty()) {
            // Use bulk_write with replace_one operations (upsert=true)
            // Keep filter documents alive so views passed to replace_one remain valid during bulk_write
            mongocxx::options::bulk_write bulk_opts;
            std::vector<mongocxx::model::write> writes;
            std::vector<bsoncxx::document::value> filter_values;
            writes.reserve(batch_docs.size());
            filter_values.reserve(batch_docs.size());
            
            for (const auto& doc : batch_docs) {
                // Create filter: model_id + layer_index + spatial_coordinates (unique compound key)
                bsoncxx::builder::stream::document filter_builder;
                filter_builder << "model_id" << model_id;
                
                // Extract layer_index and spatial_coordinates from document
                auto doc_view = doc.view();
                if (doc_view["layer_index"]) {
                    filter_builder << "layer_index" << doc_view["layer_index"].get_int32();
                }
                if (doc_view["spatial_coordinates"]) {
                    filter_builder << "spatial_coordinates" << doc_view["spatial_coordinates"].get_array();
                }
                
                filter_values.push_back(filter_builder.extract());
                const auto& filter = filter_values.back();
                
                // Create replace_one operation with upsert (filter and doc views valid until bulk_write returns)
                mongocxx::model::replace_one replace_op(filter.view(), doc_view);
                replace_op.upsert(true);
                writes.push_back(replace_op);
            }
            
            // Execute bulk write
            collection.bulk_write(writes, bulk_opts);
            total_inserted += batch_docs.size();
        }
    }
}

// Zero-copy variant: read from raw buffers
void MongoDBWriter::saveProcessedPointsFromBuffers(
    const std::string& model_id,
    const std::string& source_type,
    const double* points_xyz,
    const double* signal,
    const int* layer_indices,
    std::size_t n_pts,
    const char* timestamps_iso,
    std::size_t timestamp_stride,
    const BoundingBox& unified_bounds,
    const Eigen::Matrix4d& transformation_matrix,
    const TransformationQuality& quality_metrics,
    const std::string& processing_run_id,
    bool delete_existing,
    int batch_size
) {
    if (points_xyz == nullptr || signal == nullptr || layer_indices == nullptr) {
        throw std::invalid_argument("points_xyz, signal, layer_indices must not be null");
    }
    if (timestamps_iso != nullptr && timestamp_stride == 0) {
        throw std::invalid_argument("timestamp_stride must be > 0 when timestamps_iso is provided");
    }

    if (delete_existing) {
        deleteProcessedData(model_id, source_type, processing_run_id);
    }

    BoundingBox source_bounds;
    if (n_pts > 0) {
        source_bounds.min_x = source_bounds.max_x = points_xyz[0];
        source_bounds.min_y = source_bounds.max_y = points_xyz[1];
        source_bounds.min_z = source_bounds.max_z = points_xyz[2];
        for (std::size_t i = 1; i < n_pts; ++i) {
            double x = points_xyz[i * 3 + 0];
            double y = points_xyz[i * 3 + 1];
            double z = points_xyz[i * 3 + 2];
            source_bounds.min_x = std::min(source_bounds.min_x, x);
            source_bounds.max_x = std::max(source_bounds.max_x, x);
            source_bounds.min_y = std::min(source_bounds.min_y, y);
            source_bounds.max_y = std::max(source_bounds.max_y, y);
            source_bounds.min_z = std::min(source_bounds.min_z, z);
            source_bounds.max_z = std::max(source_bounds.max_z, z);
        }
    }

    std::string collection_name = getProcessedCollectionName(source_type);
    auto collection = db_[collection_name];
    std::string default_timestamp = currentTimestampString("%Y-%m-%dT%H:%M:%S");

    int total_points = static_cast<int>(n_pts);
    for (int batch_start = 0; batch_start < total_points; batch_start += batch_size) {
        int batch_end = std::min(batch_start + batch_size, total_points);
        std::vector<bsoncxx::document::value> batch_docs;
        batch_docs.reserve(batch_end - batch_start);

        for (int i = batch_start; i < batch_end; ++i) {
            std::size_t idx = static_cast<std::size_t>(i);
            std::string ts = (timestamps_iso != nullptr)
                ? std::string(timestamps_iso + idx * timestamp_stride, timestamp_stride)
                : default_timestamp;
            bsoncxx::builder::stream::document doc_builder;
            doc_builder << "model_id" << model_id
                        << "layer_index" << layer_indices[idx]
                        << "timestamp" << ts
                        << "spatial_coordinates" << bsoncxx::builder::stream::open_array
                            << points_xyz[idx * 3 + 0]
                            << points_xyz[idx * 3 + 1]
                            << points_xyz[idx * 3 + 2]
                        << bsoncxx::builder::stream::close_array
                        << "signal_value" << signal[idx]
                        << "coordinate_system" << "unified";
            if (!processing_run_id.empty()) {
                doc_builder << "processing_run_id" << processing_run_id;
            }
            if (i == batch_start) {
                doc_builder << "transformation_metadata" << bsoncxx::builder::stream::open_document
                    << "transformation_matrix" << matrix4dToBSONArray(transformation_matrix)
                    << "quality_metrics" << bsoncxx::builder::stream::open_document
                        << "rms_error" << quality_metrics.rms_error
                        << "alignment_quality" << quality_metrics.alignment_quality
                        << "confidence" << quality_metrics.confidence
                        << "max_error" << quality_metrics.max_error
                        << "mean_error" << quality_metrics.mean_error
                    << bsoncxx::builder::stream::close_document
                    << "source_bounds" << bsoncxx::builder::stream::open_document
                        << "min_x" << source_bounds.min_x << "min_y" << source_bounds.min_y << "min_z" << source_bounds.min_z
                        << "max_x" << source_bounds.max_x << "max_y" << source_bounds.max_y << "max_z" << source_bounds.max_z
                        << "width" << source_bounds.width() << "height" << source_bounds.height() << "depth" << source_bounds.depth()
                    << bsoncxx::builder::stream::close_document
                    << "unified_bounds" << bsoncxx::builder::stream::open_document
                        << "min_x" << unified_bounds.min_x << "min_y" << unified_bounds.min_y << "min_z" << unified_bounds.min_z
                        << "max_x" << unified_bounds.max_x << "max_y" << unified_bounds.max_y << "max_z" << unified_bounds.max_z
                        << "width" << unified_bounds.width() << "height" << unified_bounds.height() << "depth" << unified_bounds.depth()
                    << bsoncxx::builder::stream::close_document
                << bsoncxx::builder::stream::close_document;
            }
            batch_docs.push_back(doc_builder.extract());
        }

        if (!batch_docs.empty()) {
            mongocxx::options::bulk_write bulk_opts;
            std::vector<mongocxx::model::write> writes;
            std::vector<bsoncxx::document::value> filter_values;
            writes.reserve(batch_docs.size());
            filter_values.reserve(batch_docs.size());
            for (const auto& doc : batch_docs) {
                auto doc_view = doc.view();
                bsoncxx::builder::stream::document filter_builder;
                filter_builder << "model_id" << model_id;
                if (doc_view["layer_index"]) {
                    filter_builder << "layer_index" << doc_view["layer_index"].get_int32();
                }
                if (doc_view["spatial_coordinates"]) {
                    filter_builder << "spatial_coordinates" << doc_view["spatial_coordinates"].get_array();
                }
                filter_values.push_back(filter_builder.extract());
                mongocxx::model::replace_one replace_op(filter_values.back().view(), doc_view);
                replace_op.upsert(true);
                writes.push_back(replace_op);
            }
            collection.bulk_write(writes, bulk_opts);
        }
    }
}

// Save processed points with multiple signals (one document per point, batched)
void MongoDBWriter::saveProcessedPointsMultipleSignals(
    const std::string& model_id,
    const std::string& source_type,
    const Eigen::MatrixXd& transformed_points,
    const std::map<std::string, Eigen::VectorXd>& signal_values_map,
    const std::vector<int>& layer_indices,
    const std::vector<std::string>& timestamps,
    const BoundingBox& unified_bounds,
    const Eigen::Matrix4d& transformation_matrix,
    const TransformationQuality& quality_metrics,
    const std::string& processing_run_id,
    bool delete_existing,
    int batch_size
) {
    if (transformed_points.cols() != 3) {
        throw std::invalid_argument("Points matrix must have 3 columns (x, y, z)");
    }
    
    if (transformed_points.rows() != static_cast<int>(layer_indices.size())) {
        throw std::invalid_argument("Number of points must match number of layer indices");
    }
    
    if (!timestamps.empty() && timestamps.size() != static_cast<size_t>(transformed_points.rows())) {
        throw std::invalid_argument("Number of timestamps must match number of points (or be empty)");
    }
    
    // Validate all signal vectors have same length as points
    for (const auto& pair : signal_values_map) {
        if (pair.second.size() != transformed_points.rows()) {
            throw std::invalid_argument(
                "Signal '" + pair.first + "' has " + std::to_string(pair.second.size()) +
                " values, but points has " + std::to_string(transformed_points.rows())
            );
        }
    }
    
    // Delete existing data if requested (prevents duplicates)
    if (delete_existing) {
        deleteProcessedData(model_id, source_type, processing_run_id);
        // Note: deleted count could be 0 if no existing data, which is fine
    }
    
    // Compute bounds of the transformed points (source-specific bounds)
    BoundingBox source_bounds;
    if (transformed_points.rows() > 0) {
        source_bounds.min_x = source_bounds.max_x = transformed_points(0, 0);
        source_bounds.min_y = source_bounds.max_y = transformed_points(0, 1);
        source_bounds.min_z = source_bounds.max_z = transformed_points(0, 2);
        for (int i = 1; i < transformed_points.rows(); ++i) {
            source_bounds.min_x = std::min(source_bounds.min_x, transformed_points(i, 0));
            source_bounds.max_x = std::max(source_bounds.max_x, transformed_points(i, 0));
            source_bounds.min_y = std::min(source_bounds.min_y, transformed_points(i, 1));
            source_bounds.max_y = std::max(source_bounds.max_y, transformed_points(i, 1));
            source_bounds.min_z = std::min(source_bounds.min_z, transformed_points(i, 2));
            source_bounds.max_z = std::max(source_bounds.max_z, transformed_points(i, 2));
        }
    }
    
    // Get collection with "Processed_" prefix
    std::string collection_name = getProcessedCollectionName(source_type);
    auto collection = db_[collection_name];
    
    std::string default_timestamp = currentTimestampString("%Y-%m-%dT%H:%M:%S");
    
    int total_points = transformed_points.rows();
    int total_inserted = 0;
    
    // Process in batches
    for (int batch_start = 0; batch_start < total_points; batch_start += batch_size) {
        int batch_end = std::min(batch_start + batch_size, total_points);
        std::vector<bsoncxx::document::value> batch_docs;
        batch_docs.reserve(batch_end - batch_start);
        
        for (int i = batch_start; i < batch_end; ++i) {
            // Build one document per point with all signals
            bsoncxx::builder::stream::document doc_builder;
            doc_builder << "model_id" << model_id
                        << "layer_index" << layer_indices[i]
                        << "timestamp" << (timestamps.empty() ? default_timestamp : timestamps[i])
                        << "spatial_coordinates" << bsoncxx::builder::stream::open_array
                            << static_cast<double>(transformed_points(i, 0))
                            << static_cast<double>(transformed_points(i, 1))
                            << static_cast<double>(transformed_points(i, 2))
                        << bsoncxx::builder::stream::close_array
                        << "coordinate_system" << "unified";
            
            // Add processing_run_id if provided (helps track different processing runs)
            if (!processing_run_id.empty()) {
                doc_builder << "processing_run_id" << processing_run_id;
            }
            
            // Add all signal values as individual fields
            for (const auto& pair : signal_values_map) {
                doc_builder << pair.first << static_cast<double>(pair.second(i));
            }
            
            // Add transformation metadata (stored once per batch)
            if (i == batch_start) {
                doc_builder << "transformation_metadata" << bsoncxx::builder::stream::open_document
                    << "transformation_matrix" << matrix4dToBSONArray(transformation_matrix)
                    << "quality_metrics" << bsoncxx::builder::stream::open_document
                        << "rms_error" << quality_metrics.rms_error
                        << "alignment_quality" << quality_metrics.alignment_quality
                        << "confidence" << quality_metrics.confidence
                        << "max_error" << quality_metrics.max_error
                        << "mean_error" << quality_metrics.mean_error
                    << bsoncxx::builder::stream::close_document
                    << "source_bounds" << bsoncxx::builder::stream::open_document
                        << "min_x" << source_bounds.min_x
                        << "min_y" << source_bounds.min_y
                        << "min_z" << source_bounds.min_z
                        << "max_x" << source_bounds.max_x
                        << "max_y" << source_bounds.max_y
                        << "max_z" << source_bounds.max_z
                        << "width" << source_bounds.width()
                        << "height" << source_bounds.height()
                        << "depth" << source_bounds.depth()
                    << bsoncxx::builder::stream::close_document
                    << "unified_bounds" << bsoncxx::builder::stream::open_document
                        << "min_x" << unified_bounds.min_x
                        << "min_y" << unified_bounds.min_y
                        << "min_z" << unified_bounds.min_z
                        << "max_x" << unified_bounds.max_x
                        << "max_y" << unified_bounds.max_y
                        << "max_z" << unified_bounds.max_z
                        << "width" << unified_bounds.width()
                        << "height" << unified_bounds.height()
                        << "depth" << unified_bounds.depth()
                    << bsoncxx::builder::stream::close_document
                << bsoncxx::builder::stream::close_document;
            }
            
            batch_docs.push_back(doc_builder.extract());
        }
        
        // Upsert batch (overwrites existing documents with same model_id + layer_index + spatial_coordinates)
        if (!batch_docs.empty()) {
            // Use bulk_write with replace_one operations (upsert=true)
            // Keep filter documents alive so views passed to replace_one remain valid during bulk_write
            mongocxx::options::bulk_write bulk_opts;
            std::vector<mongocxx::model::write> writes;
            std::vector<bsoncxx::document::value> filter_values;
            writes.reserve(batch_docs.size());
            filter_values.reserve(batch_docs.size());
            
            for (const auto& doc : batch_docs) {
                // Create filter: model_id + layer_index + spatial_coordinates (unique compound key)
                bsoncxx::builder::stream::document filter_builder;
                filter_builder << "model_id" << model_id;
                
                // Extract layer_index and spatial_coordinates from document
                auto doc_view = doc.view();
                if (doc_view["layer_index"]) {
                    filter_builder << "layer_index" << doc_view["layer_index"].get_int32();
                }
                if (doc_view["spatial_coordinates"]) {
                    filter_builder << "spatial_coordinates" << doc_view["spatial_coordinates"].get_array();
                }
                
                filter_values.push_back(filter_builder.extract());
                const auto& filter = filter_values.back();
                
                // Create replace_one operation with upsert (filter and doc views valid until bulk_write returns)
                mongocxx::model::replace_one replace_op(filter.view(), doc_view);
                replace_op.upsert(true);
                writes.push_back(replace_op);
            }
            
            // Execute bulk write
            collection.bulk_write(writes, bulk_opts);
            total_inserted += batch_docs.size();
        }
    }
}

// Save transformation metadata
void MongoDBWriter::saveTransformationMetadata(
    const std::string& model_id,
    const std::string& source_type,
    const std::string& target_type,
    const Eigen::Matrix4d& transformation_matrix,
    const TransformationQuality& quality_metrics
) {
    auto collection = db_["transformation_metadata"];
    
    std::string timestamp = currentTimestampString("%Y-%m-%d %H:%M:%S");
    
    // Build document
    bsoncxx::builder::stream::document doc_builder;
    doc_builder << "model_id" << model_id
                << "source_type" << source_type
                << "target_type" << target_type
                << "timestamp" << timestamp
                << "transformation_matrix" << matrix4dToBSONArray(transformation_matrix)
                << "quality_metrics" << bsoncxx::builder::stream::open_document
                    << "rms_error" << quality_metrics.rms_error
                    << "alignment_quality" << quality_metrics.alignment_quality
                    << "confidence" << quality_metrics.confidence
                    << "max_error" << quality_metrics.max_error
                    << "mean_error" << quality_metrics.mean_error
                << bsoncxx::builder::stream::close_document;
    
    auto doc = doc_builder.extract();
    
    // Insert document
    collection.insert_one(doc.view());
}

// Save unified bounds metadata
void MongoDBWriter::saveUnifiedBounds(
    const std::string& model_id,
    const BoundingBox& unified_bounds,
    const std::vector<std::string>& source_types
) {
    auto collection = db_["unified_bounds"];
    
    std::string timestamp = currentTimestampString("%Y-%m-%d %H:%M:%S");
    
    // Build document
    bsoncxx::builder::stream::document doc_builder;
    doc_builder << "model_id" << model_id
                << "timestamp" << timestamp
                << "bounds" << bsoncxx::builder::stream::open_document
                    << "min_x" << unified_bounds.min_x
                    << "min_y" << unified_bounds.min_y
                    << "min_z" << unified_bounds.min_z
                    << "max_x" << unified_bounds.max_x
                    << "max_y" << unified_bounds.max_y
                    << "max_z" << unified_bounds.max_z
                << bsoncxx::builder::stream::close_document
                << "width" << unified_bounds.width()
                << "height" << unified_bounds.height()
                << "depth" << unified_bounds.depth();
    
    // Add source types array
    bsoncxx::builder::stream::array sources_builder;
    for (const auto& source : source_types) {
        sources_builder << source;
    }
    doc_builder << "source_types" << sources_builder;
    
    auto doc = doc_builder.extract();
    
    // Insert document
    collection.insert_one(doc.view());
}

} // namespace io
} // namespace am_qadf_native

#endif // EIGEN_AVAILABLE
