#ifndef AM_QADF_NATIVE_IO_MONGODB_WRITER_HPP
#define AM_QADF_NATIVE_IO_MONGODB_WRITER_HPP

#ifdef EIGEN_AVAILABLE
#include <Eigen/Dense>
#include <mongocxx/client.hpp>
#include <mongocxx/database.hpp>
#include <mongocxx/collection.hpp>
#include <bsoncxx/builder/stream/document.hpp>
#include <bsoncxx/builder/stream/array.hpp>
#include "am_qadf_native/synchronization/point_bounds.hpp"
#include "am_qadf_native/synchronization/point_transformation_estimate.hpp"
#include <string>
#include <vector>
#include <array>
#include <map>

namespace am_qadf_native {
namespace io {

// MongoDBWriter: Saves processed/transformed data to MongoDB with "Processed_" prefix
// This is the data warehouse layer - stores transformed points for downstream processing
class MongoDBWriter {
private:
    mongocxx::client client_;
    mongocxx::database db_;
    
public:
    MongoDBWriter(const std::string& uri, const std::string& db_name);
    
    // Delete existing processed data for a model_id (prevents duplicates)
    // Returns number of documents deleted
    int deleteProcessedData(
        const std::string& model_id,
        const std::string& source_type,
        const std::string& processing_run_id = ""  // If empty, deletes all processed data for model_id
    );
    
    // Save processed points (transformed points with unified coordinate system)
    // Collection name: "Processed_<source_type>_data" (e.g., "Processed_thermal_data")
    // Saves one document per point (batched for efficiency) - matches raw data structure
    // 
    // OVERWRITE BEHAVIOR: Uses upsert to overwrite existing documents with same:
    //   - model_id + layer_index + spatial_coordinates (exact point match)
    // This ensures same sample from same data source gets overwritten, no duplication.
    //
    // batch_size: Number of documents to upsert per batch (default: 10000)
    // delete_existing: If true, deletes ALL existing processed data for this model_id before upserting
    void saveProcessedPoints(
        const std::string& model_id,
        const std::string& source_type,  // "thermal", "hatching", "acoustic", etc.
        const Eigen::MatrixXd& transformed_points,
        const Eigen::VectorXd& signal_values,
        const std::vector<int>& layer_indices,  // Layer index for each point (required)
        const std::vector<std::string>& timestamps,  // ISO timestamp strings for each point (optional, can be empty)
        const synchronization::BoundingBox& unified_bounds,
        const Eigen::Matrix4d& transformation_matrix,
        const synchronization::TransformationQuality& quality_metrics,
        const std::string& processing_run_id = "",  // Optional: track processing run
        bool delete_existing = false,  // If true, delete ALL existing processed data for model_id before upserting
        int batch_size = 10000  // Batch size for upsert
    );

    // Zero-copy variant: read from raw buffers (e.g. numpy). Use for billions of points.
    // points_xyz: row-major n_pts*3 (x,y,z per point). signal: n_pts. layer_indices: n_pts ints.
    // timestamps_iso: optional, n_pts * timestamp_stride bytes (ISO string per point); if null use default.
    void saveProcessedPointsFromBuffers(
        const std::string& model_id,
        const std::string& source_type,
        const double* points_xyz,
        const double* signal,
        const int* layer_indices,
        std::size_t n_pts,
        const char* timestamps_iso,
        std::size_t timestamp_stride,
        const synchronization::BoundingBox& unified_bounds,
        const Eigen::Matrix4d& transformation_matrix,
        const synchronization::TransformationQuality& quality_metrics,
        const std::string& processing_run_id = "",
        bool delete_existing = false,
        int batch_size = 10000
    );

    // Save processed points with multiple signals
    // Each point becomes one document with all signal values
    // 
    // OVERWRITE BEHAVIOR: Uses upsert to overwrite existing documents with same:
    //   - model_id + layer_index + spatial_coordinates (exact point match)
    // This ensures same sample from same data source gets overwritten, no duplication.
    void saveProcessedPointsMultipleSignals(
        const std::string& model_id,
        const std::string& source_type,
        const Eigen::MatrixXd& transformed_points,
        const std::map<std::string, Eigen::VectorXd>& signal_values_map,
        const std::vector<int>& layer_indices,  // Layer index for each point (required)
        const std::vector<std::string>& timestamps,  // ISO timestamp strings for each point (optional, can be empty)
        const synchronization::BoundingBox& unified_bounds,
        const Eigen::Matrix4d& transformation_matrix,
        const synchronization::TransformationQuality& quality_metrics,
        const std::string& processing_run_id = "",  // Optional: track processing run
        bool delete_existing = false,  // If true, delete ALL existing processed data for model_id before upserting
        int batch_size = 10000  // Batch size for upsert
    );
    
    // Save transformation metadata
    // Stores transformation matrix and quality metrics for reference
    void saveTransformationMetadata(
        const std::string& model_id,
        const std::string& source_type,
        const std::string& target_type,  // Reference coordinate system
        const Eigen::Matrix4d& transformation_matrix,
        const synchronization::TransformationQuality& quality_metrics
    );
    
    // Save unified bounds metadata
    void saveUnifiedBounds(
        const std::string& model_id,
        const synchronization::BoundingBox& unified_bounds,
        const std::vector<std::string>& source_types  // Which sources contributed to these bounds
    );

private:
    // Helper: Convert Eigen::MatrixXd to BSON array
    bsoncxx::builder::stream::array pointsToBSONArray(const Eigen::MatrixXd& points);
    
    // Helper: Convert Eigen::VectorXd to BSON array
    bsoncxx::builder::stream::array vectorToBSONArray(const Eigen::VectorXd& vec);
    
    // Helper: Convert Eigen::Matrix4d to BSON array
    bsoncxx::builder::stream::array matrix4dToBSONArray(const Eigen::Matrix4d& matrix);
    
    // Helper: Get collection name with "Processed_" prefix
    std::string getProcessedCollectionName(const std::string& source_type);
};

} // namespace io
} // namespace am_qadf_native

#endif // EIGEN_AVAILABLE
#endif // AM_QADF_NATIVE_IO_MONGODB_WRITER_HPP
